# ga.py
#
#elif METHOD == "ga":
#    ga = GeneticAlgorithm(
#        population_size=10,
#        user_df=df_users,
#        access_matrix=df_access.to_dict(orient="records"),
#        W=W,
#        path_loss=path_loss,
#        sat_channel_dict=copy.deepcopy(sat_channel_dict_backup),
#        params=params,
#        seed=123456  # ✅ 固定一個整數 seed；不想固定就拿掉這行
#    )
#    ga.evolve(generations=5)  # 訓練 5 輪(5輪大概要1小時)，可調整為 20、50 等等
#    results_df, all_user_paths, load_by_time, df_data_rates = ga.export_best_result()
# src/ga.py
import random                         # 亂數用於初始化、洗牌、突變等
import copy                           # 深拷貝，用來複製巢狀 dict/list
import ast                            # 安全地把字串轉成 Python 物件（例如 "['S1','S2']" -> ['S1','S2']）
import pandas as pd                   # 資料表處理
from collections import defaultdict   # 預設字典，省去不存在鍵時的初始化
from src.dp import run_dp_path_for_user  # 變異時可用 DP 幫單一 user 重算路徑
from src.utils import (               # 工具函式：算速率/分數/負載/可視性
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility,
)

# === Blocking reasons ===
BLOCK_NO_PAIR      = "no_pair_available"          # 此時刻完全沒有可用的 (sat, ch)
BLOCK_NO_FEASIBLE  = "no_feasible_W_or_conflict"  # 有可見/空閒，但無法串滿 [t_start, t_end]

def _mark_block(rows_list, uid, t, reason):
    """把單一時槽的 blocked 記錄推進 rows_list。"""
    rows_list.append((uid, t, None, None, 0.0, True, reason))  # (user, time, sat=None, ch=None, rate=0, blocked=True, reason)

class Individual:
    """
    一個個體，代表所有 user 的路徑配置方案。
    內含：
      - position: dict[user_id] -> List[(sat, ch, t)]  各 user 的完整路徑
      - data_rates: List[tuple]  每個 (user, t) 的資料率與是否 blocked
      - reward: float            總分
    """
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.user_df = user_df                                   # 使用者資料表（含 t_start/t_end）
        self.access_matrix = access_matrix                        # 每個 time slot 的可視衛星資訊（list[dict]）
        self.df_access = pd.DataFrame(access_matrix)              # 若 check_visibility 需要 DataFrame
        self.W = W                                                # 換手時需連續保留的 slots（非換手時每次 1 slot）
        self.path_loss = path_loss                                # (sat, t) -> path loss
        self.sat_channel_dict = sat_channel_dict                  # 全域通道狀態（0 可用；非 0 不可分配）
        self.params = params                                      # 系統參數（EIRP、雜訊等）
        self.rng = random.Random(seed)                            # 固定亂數種子保重現性

        self.position = {}                                        # user_id -> 路徑[(sat, ch, t)]
        self.data_rates = []                                      # (user_id, time, sat, channel, data_rate, blocked, reason)
        self.reward = 0.0                                         # 個體總分

        self.generate_fast_path()                                 # 用「先到先得的 greedy-like」初始化（含 blocking）

    # ---------- 取得某時槽的可見衛星（支援字串/清單 & per-user 欄位） ----------
    def _visible_sats_for_user(self, t, user_id):
        row = self.access_matrix[t]                               # 取第 t 個 time slot 的資料
        if "visible_sats_for_user" in row:                        # 若有 per-user 可視衛星欄位
            vs = row["visible_sats_for_user"].get(user_id, row.get("visible_sats", []))  # 先取 per-user，否則退回全域
        else:
            vs = row.get("visible_sats", [])                      # 只有全域可視衛星
        if isinstance(vs, str):                                   # 若是字串（例如 "['S1','S2']"）
            try:
                vs = ast.literal_eval(vs)                         # 轉回 Python list
            except Exception:
                vs = []                                           # 解析失敗就當作不可見
        return vs                                                 # 回傳 list[str]

    # ---------- 初始化個體（先到先得 + 無法覆蓋整段就 block） ----------
    def generate_fast_path(self):
        self.position = {}                                        # 清空路徑
        self.data_rates = []                                      # 清空資料率紀錄（含 blocked）
        self.reward = 0.0                                         # 清空總分

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)       # 通道狀態快照（0 可分配；非 0 視為被占或保留）
        total_reward = 0.0                                        # 累積本個體的總分

        active_user_paths = []                                    # 已經「落地」且還沒到期的 user（用來到期釋放資源）

        # 依 t_start 排序，再以批次（相同 t_start）亂數打散
        for t_val, group_df in self.user_df.sort_values("t_start").groupby("t_start"):
            users = list(group_df.itertuples(index=False))        # 取該批次的 users
            self.rng.shuffle(users)                               # 打亂這個批次的處理順序

            for user in users:                                    # 逐一處理 user（先到先得）
                user_id = int(user.user_id)                       # 取得 user_id
                t_start = int(user.t_start)                       # 該 user 的起始時間
                t_end   = int(user.t_end)                         # 該 user 的結束時間

                # 釋放「已完成且早於本使用者 t_start」的資源
                to_remove = []                                     # 暫存要移除的 active user
                for old in active_user_paths:                      # 走訪目前活躍中的使用者
                    if old["t_end"] < t_start:                     # 若他們的服務期已經在此 user 前結束
                        for s, c in set((s, c) for s, c, _ in old["path"]):  # 取其用過的 (sat, ch)
                            tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)  # 把通道計數 -1（釋放）
                        to_remove.append(old)                      # 記錄後續要移除
                for u in to_remove:                                # 從活躍清單移除這些已到期者
                    active_user_paths.remove(u)

                # ===== 從 t_start 開始為此 user 建 path；若無法覆蓋整段則整段 blocked =====
                t = t_start                                        # 當前時刻
                current_sat, current_ch = None, None               # 當前使用的衛星/通道
                last_ho_time = t_start                             # 上次換手的時刻
                is_first_handover = True                           # 是否第一次換手（允許立即換）

                user_path = []                                     # 該 user 的路徑 (sat, ch, t) 列表
                data_rate_rows = []                                # 該 user 的資料率列（含 blocked=False）
                user_reward = 0.0                                  # 該 user 的分數累計

                # --- 第一次挑選（在 t_start） ---
                best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0.0  # 目前最佳候選
                visible_sats = self._visible_sats_for_user(t_start, user_id)                    # 取可見衛星清單
                for sat in visible_sats:                               # 走訪每顆可見衛星
                    for ch, state in tmp_sat_dict.get(sat, {}).items():# 走訪該衛星所有通道
                        if state != 0:                                  # 非 0（包含 1 或 2）視為不可分配
                            continue
                        _, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)  # 算速率
                        if not data_rate:                               # 無速率或無法計算就跳過
                            continue
                        m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())  # 依快照計每顆衛星負載
                        score = compute_score(self.params, m_s_t, data_rate, sat) + 1e-9 * self.rng.random()  # 算分（加微小亂數避平手）
                        if score > best_score:                          # 更新最佳候選
                            best_score = score
                            best_sat, best_ch, best_data_rate = sat, ch, data_rate

                # 若在 t_start 就沒有任何可用 pair → 整段 blocked（no_pair）
                if best_sat is None:
                    for tt in range(t_start, t_end + 1):                # 把該 user 全段標成 blocked
                        _mark_block(self.data_rates, user_id, tt, BLOCK_NO_PAIR)
                    self.position[user_id] = []                         # 該 user 路徑為空
                    continue                                            # 換下一個 user

                # 有候選 → 先落入第一個 slot
                current_sat, current_ch = best_sat, best_ch             # 確認當前衛星/通道
                user_path.append((current_sat, current_ch, t_start))    # 紀錄路徑
                data_rate_rows.append(                                   # 紀錄資料率（非 blocked）
                    (user_id, t_start, current_sat, current_ch, best_data_rate, False, None)
                )
                m_s_t0 = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())         # 依快照計負載
                user_reward += compute_score(self.params, m_s_t0, best_data_rate, current_sat)  # 累積分數

                # --- 後續時間（t_start+1..t_end） ---
                t = t_start + 1                                         # 下一個時槽
                covered_ok = True                                       # 先假設可覆蓋完整段

                while t <= t_end:                                       # 直到 user 的 t_end
                    can_handover = is_first_handover or (t - last_ho_time >= self.W)  # 確認是否允許換手
                    did_handover = False                                # 本輪是否真的換手

                    best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")  # 本輪最佳候選（預設沿用原通道）

                    if can_handover:                                    # 若允許換手，試著換到更好且能撐 W 的通道
                        vsats = list(self._visible_sats_for_user(t, user_id))  # 當前可見衛星
                        self.rng.shuffle(vsats)                         # 打散順序
                        for sat in vsats:                               # 逐顆可見衛星
                            ch_list = list(tmp_sat_dict.get(sat, {}).keys())  # 取該衛星所有通道
                            self.rng.shuffle(ch_list)                    # 打散通道順序
                            for ch in ch_list:                           # 逐通道檢查
                                if tmp_sat_dict[sat][ch] != 0:           # 通道不可分配就跳過
                                    continue
                                # 換手時要求未來能連續撐滿 W（或到 t_end）
                                end_req = min(t_end, t + self.W - 1)     # 需要覆蓋的最後一個時槽
                                can_cover = True
                                for tt in range(t, end_req + 1):         # 檢查 t..end_req 的可視性
                                    if sat not in self._visible_sats_for_user(tt, user_id):
                                        can_cover = False                # 其中有不可見 → 不可用
                                        break
                                if not can_cover:
                                    continue
                                # 計算換到此 (sat,ch) 的即時速率與分數
                                _, dr0 = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                                if not dr0:
                                    continue
                                m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                score0 = compute_score(self.params, m_s_t, dr0, sat) + 1e-9 * self.rng.random()
                                if score0 > best_score:                  # 取分數最高者
                                    best_score = score0
                                    best_sat, best_ch = sat, ch

                        # 若找到比現在更好的 (sat,ch) → 執行換手
                        if (best_sat is not None) and (best_sat != current_sat or best_ch != current_ch):
                            current_sat, current_ch = best_sat, best_ch  # 改成新的衛星/通道
                            last_ho_time = t                              # 更新上次換手時間
                            is_first_handover = False                     # 第一次換手已發生
                            did_handover = True                           # 本輪確實換手

                    # 鎖定長度：換手 → 一次鎖 W；續用 → 每次只鎖 1 slot
                    step = self.W if did_handover else 1

                    for w in range(step):                                 # 逐 slot 寫入
                        tt = t + w
                        if tt > t_end:
                            break                                         # 不要超出使用者的 t_end
                        # 若未換手，至少要在當下可見；否則失敗（無法完整覆蓋）
                        if not did_handover and (current_sat not in self._visible_sats_for_user(tt, user_id)):
                            covered_ok = False                            # 宣告覆蓋失敗
                            break

                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)  # 算速率
                        user_path.append((current_sat, current_ch, tt))   # 寫路徑
                        data_rate_rows.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0, False, None))  # 寫資料率
                        if dr and dr > 0:
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            user_reward += compute_score(self.params, m_s_t, dr, current_sat)  # 累分

                    if not covered_ok:                                    # 若中途失敗，跳出
                        break
                    t += step                                             # 前進到下一個決策時刻

                # ===== 成功/失敗落地 =====
                full_len = (t_end - t_start + 1)                          # 需要覆蓋的總長度
                covered_len = len({tt for _, _, tt in user_path})         # 真正覆蓋到的時槽數（用 set 防重複）

                if covered_len != full_len:                                # 沒覆蓋完整段 → 視為不可行
                    for tt in range(t_start, t_end + 1):                   # 把整段標 blocked(no_feasible)
                        _mark_block(self.data_rates, user_id, tt, BLOCK_NO_FEASIBLE)
                    self.position[user_id] = []                            # 該 user 無路徑
                    continue                                               # 換下一個 user

                # 覆蓋完整段 → 落地：鎖資源、寫資料率、累分、加入活躍池
                self.position[user_id] = user_path                         # 記錄路徑
                for s, c in set((s, c) for s, c, _ in user_path):          # 對所有用到的 (sat, ch)
                    tmp_sat_dict[s][c] += 1                                # 設為被占用（直到該 user 釋放）
                self.data_rates.extend(data_rate_rows)                     # 加入本 user 的資料率列
                total_reward += user_reward                                # 累計個體總分

                active_user_paths.append({                                 # 加到活躍池（之後按 t_end 釋放）
                    "user_id": user_id,
                    "path": user_path,
                    "t_end": t_end
                })

        self.reward = total_reward                                         # 設定本個體的 reward

    # 可保留：在 mutate() 或其他地方想單跑 greedy path 用；目前沒有直接呼叫
    def _run_greedy_path(self, user_id, t_start, t_end, tmp_sat_dict):
        """對單一使用者跑 greedy 路徑（和 generate_fast_path 內部邏輯一致的簡化版）。"""
        user_path = []                                                     # 回傳路徑
        data_rate_records = []                                             # 回傳資料率列
        user_reward = 0                                                    # 回傳 reward
        t = t_start                                                        # 當下時刻
        current_sat, current_ch = None, None                               # 當前 (sat, ch)
        last_ho_time = t_start                                             # 上次換手時刻
        is_first_handover = True                                           # 是否第一次換手

        # === 第一次選擇 ===
        best_sat, best_ch, best_score, best_dr = None, None, float("-inf"), 0.0
        for sat in self._visible_sats_for_user(t_start, user_id):          # 取可見衛星
            for ch, state in tmp_sat_dict.get(sat, {}).items():            # 該衛星所有通道
                if state != 0:                                             # 非 0 不可用
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)  # 算速率
                if dr and dr > 0:
                    score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)  # 算分
                    score += 1e-9 * self.rng.random()                      # 微調亂數避平手
                    if score > best_score:                                 # 取最大分
                        best_score = score
                        best_sat, best_ch, best_dr = sat, ch, dr

        if best_sat is None:                                               # 沒有起點可用 pair → 失敗
            return [], [], 0

        current_sat, current_ch = best_sat, best_ch                        # 寫入第一個 slot
        user_path.append((current_sat, current_ch, t_start))
        data_rate_records.append((user_id, t_start, current_sat, current_ch, best_dr, False, None))
        user_reward += best_score

        # === 後續時間 ===
        t += 1                                                             # 下一個時槽
        while t <= t_end:                                                  # 直到 t_end
            can_ho = is_first_handover or (t - last_ho_time >= self.W)     # 是否可換手
            did_ho = False                                                 # 本輪是否換手
            best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")  # 預設沿用

            if can_ho:                                                     # 可換手 → 嘗試找能撐 W 的更好 pair
                vsats = list(self._visible_sats_for_user(t, user_id))
                self.rng.shuffle(vsats)
                for sat in vsats:
                    chs = list(tmp_sat_dict.get(sat, {}).keys())
                    self.rng.shuffle(chs)
                    for ch in chs:
                        if tmp_sat_dict[sat][ch] != 0:                     # 通道不可分配
                            continue
                        if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):  # 檢查未來 W 可視
                            continue
                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                        if dr and dr > 0:
                            score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)
                            score += 1e-9 * self.rng.random()
                            if score > best_score:                          # 取分高者
                                best_score = score
                                best_sat, best_ch = sat, ch
                if (best_sat != current_sat) or (best_ch != current_ch):    # 若找到更好者 → 換手
                    current_sat, current_ch = best_sat, best_ch
                    last_ho_time = t
                    is_first_handover = False
                    did_ho = True

            step = self.W if did_ho else 1                                  # 換手鎖 W，否則 1
            for w in range(step):                                            # 寫入 step 個 slot
                tt = t + w
                if tt > t_end:
                    break
                if not did_ho and current_sat not in self._visible_sats_for_user(tt, user_id):  # 續用時須可見
                    break
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)
                user_path.append((current_sat, current_ch, tt))
                data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0, False, None))
                if dr and dr > 0:
                    user_reward += compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, current_sat)
            t += step                                                        # 跳到下一次決策時刻

        return user_path, data_rate_records, user_reward                     # 回傳路徑、資料率、分數

    # ---------- 重新計分（用於 crossover/mutation 後） ----------
    def rebuild_from_position(self):
        """依照 self.position 重建 data_rates 與 reward（只算成功的 rows；不補 blocked）。"""
        self.data_rates = []                                                 # 重新建立資料率清單
        self.reward = 0.0                                                    # 重新計總分
        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)                  # 通道狀態快照
        total_reward = 0.0                                                   # 總分累積

        df_ts = self.user_df[["user_id", "t_start", "t_end"]].copy().sort_values("t_start")  # 依 t_start 排序
        active_user_paths = []                                               # 活躍池（釋放資源用）

        for _, row in df_ts.iterrows():                                      # 逐 user 重新計分
            user_id = int(row["user_id"])                                    # 取 user_id
            t_start = int(row["t_start"])                                    # 起始
            t_end = int(row["t_end"])                                        # 結束

            # 釋放已完成的使用者資源
            to_remove = []                                                   # 暫存要移除的活躍者
            for old in active_user_paths:                                    # 走訪活躍池
                if old["t_end"] < t_start:                                   # 若其服務在本 user t_start 前已結束
                    for s, c in set((s, c) for s, c, _ in old["path"]):      # 還原他用過的 (s,c)
                        tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)  # -1
                    to_remove.append(old)                                    # 記錄要移除
            for u in to_remove:                                              # 移出活躍池
                active_user_paths.remove(u)

            path = self.position.get(user_id, [])                            # 取本 user 的路徑
            if not path:                                                     # 若沒有路徑就略過（此函式不補 blocked）
                continue
            path = sorted(path, key=lambda x: x[2])                          # 依時間排序

            user_reward = 0.0                                                # 該 user 的分數
            used_pairs = set()                                               # 該 user 用到的 (s,c)

            for sat, ch, t in path:                                          # 逐時槽計分
                if sat not in self._visible_sats_for_user(t, user_id):       # 該時槽不可見就不計分
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)  # 算速率
                self.data_rates.append((user_id, t, sat, ch, dr if dr else 0.0, False, None))         # 寫結果列
                if dr and dr > 0:                                            # 有效速率才計分與鎖資源
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t, dr, sat) # 累分
                    used_pairs.add((sat, ch))                                 # 紀錄用到的通道

            for s, c in used_pairs:                                          # 鎖住 (s,c)（直到該 user 結束）
                tmp_sat_dict[s][c] = tmp_sat_dict[s].get(c, 0) + 1

            total_reward += user_reward                                      # 累計個體總分
            active_user_paths.append({"user_id": user_id, "path": path, "t_end": t_end})  # 加回活躍池

        self.reward = total_reward                                           # 更新個體總分

class GeneticAlgorithm:
    """GA 主流程：初始化 → 選擇 → 交配 → 變異 → 重新計分 → 更新最佳個體。"""
    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.population_size = population_size                               # 族群大小
        self.user_df = user_df                                               # 使用者資料
        self.access_matrix = access_matrix                                   # 可視矩陣
        self.W = W                                                           # 換手鎖定長度
        self.path_loss = path_loss                                           # path loss 表
        self.params = params                                                 # 系統參數
        self.seed_base = seed or random.randint(0, 999999)                   # 初始化亂數種子

        # 初始化族群（每個個體用 greedy-like 生成初始解；其中已含 blocked 記錄）
        self.population = [
            Individual(
                user_df, access_matrix, W, path_loss, copy.deepcopy(sat_channel_dict),
                params, seed=self.seed_base + i * 7919                       # 每個個體用不同種子
            )
            for i in range(population_size)
        ]
        self.population.sort(key=lambda x: x.reward, reverse=True)           # 依 reward 由大到小排序
        self.best_individual = copy.deepcopy(self.population[0])             # 目前最佳個體

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        for gen in range(generations):                                       # 進化多代
            next_gen = self.population[:elite_size]                          # 精英保留（前幾名直接複製進下一代）
            while len(next_gen) < self.population_size:                      # 補齊下一代
                p1, p2 = self.tournament_selection(), self.tournament_selection()  # 競賽選擇兩個父母
                child = self.crossover(p1, p2)                               # 交配產生子代
                self.mutate(child, mutation_rate)                            # 以一定機率突變
                next_gen.append(child)                                       # 放入下一代

            # 交配/突變後需要「依 position 重建」data_rates/reward
            for ind in next_gen:
                ind.rebuild_from_position()

            self.population = sorted(next_gen, key=lambda x: x.reward, reverse=True)  # 重新排序
            if self.population[0].reward > self.best_individual.reward:      # 若本代最強超過歷史最佳
                self.best_individual = copy.deepcopy(self.population[0])     # 更新最佳個體

    def tournament_selection(self, k=3):
        return max(random.sample(self.population, k), key=lambda x: x.reward)  # 從族群隨機抽 k 個，回傳 reward 最大者

    def crossover(self, p1, p2):
        child = copy.deepcopy(p1)                                            # 以父 1 為基底
        for uid in child.position:                                           # 逐 user 拿父 2 的路徑（機率 0.5）
            if random.random() < 0.5:
                child.position[uid] = copy.deepcopy(p2.position.get(uid, []))
        child.rebuild_from_position()                                        # 重建子代的 data_rates/reward
        return child

    def mutate(self, individual, mutation_rate):
        mutated = False                                                      # 標記是否有突變發生
        for user in individual.user_df.itertuples():                         # 逐 user 嘗試突變
            if random.random() < mutation_rate:                              # 以 mutation_rate 機率突變一位 user
                uid = int(user.user_id)                                      # user_id
                t_start, t_end = int(user.t_start), int(user.t_end)          # 起訖
                snapshot = self._build_snapshot(individual, uid)             # 構造「排除該 user 現有路徑」後的快照
                path, reward, success, data_rates = run_dp_path_for_user(    # 用 DP 幫這位 user 重算路徑
                    uid, t_start, t_end, self.W, self.access_matrix,
                    self.path_loss, snapshot, self.params
                )
                if success:                                                  # 若 DP 找到完整路徑
                    individual.position[uid] = path                          # 用新路徑覆蓋
                    mutated = True                                           # 標記突變成功
        if mutated:
            individual.rebuild_from_position()                               # 有變動才重建 data_rates/reward

    def _build_snapshot(self, individual, exclude_user_id):
        tmp = copy.deepcopy(individual.sat_channel_dict)                     # 以原始通道狀態為底
        for uid, path in individual.position.items():                        # 把其他使用者的路徑占用加上去
            if uid == exclude_user_id:                                       # 排除要突變的 user
                continue
            for s, c, _ in path:                                             # 只看 (sat, ch)，不管 time（全段視為占用）
                tmp[s][c] += 1
        return tmp                                                           # 回傳快照

    def export_best_result(self):
        """
        匯出：
          - results_df：每位使用者成功/失敗（user 粒度）
          - all_user_paths：展開 (sat, ch, t) 的路徑（給你存 paths.csv）
          - load_by_time：每個時槽每顆衛星的使用人數（畫負載用）
          - df_data_rates：成功 rows + 對每位 user 補齊缺的時槽（blocked=True, reason=no_feasible）
                            → 保證與 DP/greedy 的 blocking 計算口徑一致
        """
        best = self.best_individual                                          # 取最佳個體

        # 先把 best.data_rates 變成 DataFrame（注意：rebuild 之後的 data_rates 通常不含 blocked rows）
        cols = ["user_id", "time", "sat", "channel", "data_rate", "blocked", "reason"]
        df_rates = pd.DataFrame(best.data_rates, columns=cols)               # 成功 rows（可能缺少 blocked）

        # 準備輸出容器
        load_by_time = defaultdict(lambda: defaultdict(int))                 # t -> {sat: count}
        all_user_paths = []                                                  # 各 user 的路徑摘要
        results = []                                                         # 各 user 的成功/失敗

        # 幫助快速檢查「某 (uid, time) 是否已有成功 row」
        have_row = set(zip(df_rates["user_id"], df_rates["time"])) if not df_rates.empty else set()

        # 逐 user 補齊 df_rates 缺少的時槽為 blocked，以對齊 DP/greedy 的統計口徑
        for row in self.user_df.itertuples(index=False):
            uid = int(row.user_id)                                           # 使用者 ID
            t_start = int(row.t_start)                                       # 起始時刻
            t_end   = int(row.t_end)                                         # 結束時刻

            path = best.position.get(uid, [])                                # 該 user 的路徑（可能為空）
            # success 定義：有 path 且覆蓋完整段
            success = False
            if path:
                times = {t for _, _, t in path}                              # 此路徑覆蓋的所有時槽
                success = len(times) == (t_end - t_start + 1)                # 完全覆蓋才算成功

            # 對 [t_start, t_end] 中「缺失的時槽」補 blocked(no_feasible)
            missing_rows = []
            for tt in range(t_start, t_end + 1):
                if (uid, tt) not in have_row:                                # 若此 (uid, tt) 沒有成功列
                    missing_rows.append((uid, tt, None, None, 0.0, True, BLOCK_NO_FEASIBLE))
            if missing_rows:                                                 # 有缺才補
                df_rates = pd.concat([df_rates, pd.DataFrame(missing_rows, columns=cols)], ignore_index=True)

            # 輸出路徑與 load_by_time，以及 user-level 的 results
            if path:                                                         # 若有路徑
                for s, c, t in path:                                        # 計算衛星負載
                    load_by_time[t][s] += 1
                t_begin = min(t for _, _, t in path)                         # 路徑最早時槽
                t_end2  = max(t for _, _, t in path)                         # 路徑最晚時槽
                all_user_paths.append({                                      # 路徑摘要
                    "user_id": uid,
                    "path": path,
                    "t_begin": t_begin,
                    "t_end": t_end2,
                    "success": success,
                    "reward": None
                })
                results.append({"user_id": uid, "reward": None, "success": success})  # user 粒度結果
            else:                                                            # 無路徑（全部 blocked）
                all_user_paths.append({
                    "user_id": uid,
                    "path": [],
                    "t_begin": t_start,
                    "t_end": t_end,
                    "success": False,
                    "reward": 0.0
                })
                results.append({"user_id": uid, "reward": None, "success": False})

        # 排序 df_rates（方便與其他方法對齊）
        if not df_rates.empty:
            df_rates = df_rates.sort_values(["user_id", "time"]).reset_index(drop=True)

        # 回傳四個輸出（與你 main.py 預期介面一致）