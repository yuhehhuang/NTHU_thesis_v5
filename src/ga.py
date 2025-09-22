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
import random
import copy
import ast
import pandas as pd
from collections import defaultdict
from src.dp import run_dp_path_for_user
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility,
)

# === Blocking reasons ===
BLOCK_NO_PAIR = "no_pair_available"
BLOCK_NO_FEASIBLE = "no_feasible_W_or_conflict"

def _mark_block(rows_list, uid, t, reason):
    """把單一時槽的 blocked 記錄推進 rows_list。"""
    rows_list.append((uid, t, None, None, 0.0, True, reason))

class Individual:
    """
    一個個體，代表所有 user 的路徑配置方案。
    內含：
      - position: dict[user_id] -> List[(sat, ch, t)]  各 user 的完整路徑
      - data_rates: List[tuple]  每個 (user, t) 的資料率與是否 blocked
      - reward: float            總分
    """
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.df_access = pd.DataFrame(access_matrix)
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params
        self.rng = random.Random(seed)

        self.position = {}
        self.data_rates = []
        self.reward = 0.0

        self.generate_fast_path()

    # ---------- 取得某時槽的可見衛星（支援字串/清單 & per-user 欄位） ----------
    def _visible_sats_for_user(self, t, user_id):
        row = self.access_matrix[t]
        if "visible_sats_for_user" in row:
            vs = row["visible_sats_for_user"].get(user_id, row.get("visible_sats", []))
        else:
            vs = row.get("visible_sats", [])
        if isinstance(vs, str):
            try:
                vs = ast.literal_eval(vs)
            except Exception:
                vs = []
        return vs

    # ---------- 初始化個體（先到先得 + 無法覆蓋整段就 block） ----------
    def generate_fast_path(self):
        self.position = {}
        self.data_rates = []
        self.reward = 0.0

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        active_user_paths = []

        for t_val, group_df in self.user_df.sort_values("t_start").groupby("t_start"):
            users = list(group_df.itertuples(index=False))
            self.rng.shuffle(users)

            for user in users:
                user_id = int(user.user_id)
                t_start = int(user.t_start)
                t_end = int(user.t_end)

                to_remove = []
                for old in active_user_paths:
                    if old["t_end"] < t_start:
                        for s, c in set((s, c) for s, c, _ in old["path"]):
                            tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)
                        to_remove.append(old)
                for u in to_remove:
                    active_user_paths.remove(u)

                t = t_start
                current_sat, current_ch = None, None
                last_ho_time = t_start
                is_first_handover = True

                user_path = []
                data_rate_rows = []
                user_reward = 0.0

                # --- 第一次挑選（在 t_start） ---
                best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0.0
                visible_sats = self._visible_sats_for_user(t_start, user_id)
                
                # 新增邏輯：檢查每個候選在未來 W 個時槽內的可視性
                candidates = []
                for sat in visible_sats:
                    for ch, state in tmp_sat_dict.get(sat, {}).items():
                        if state != 0:
                            continue
                        
                        # 檢查從 t_start 開始，是否能連續服務至少 W 個時槽
                        can_cover_W = True
                        end_check = min(t_end, t_start + self.W - 1)
                        for tt in range(t_start, end_check + 1):
                            if sat not in self._visible_sats_for_user(tt, user_id):
                                can_cover_W = False
                                break
                        
                        if can_cover_W:
                            _, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                            if data_rate:
                                m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                score = compute_score(self.params, m_s_t, data_rate, sat) + 1e-9 * self.rng.random()
                                candidates.append({'sat': sat, 'ch': ch, 'score': score, 'rate': data_rate})

                if not candidates:
                    for tt in range(t_start, t_end + 1):
                        _mark_block(self.data_rates, user_id, tt, BLOCK_NO_PAIR)
                    self.position[user_id] = []
                    continue
                
                best_candidate = max(candidates, key=lambda x: x['score'])
                best_sat, best_ch, best_data_rate = best_candidate['sat'], best_candidate['ch'], best_candidate['rate']
                best_score = best_candidate['score']

                # 有候選 → 先落入第一個 slot
                current_sat, current_ch = best_sat, best_ch
                
                # --- 後續時間（t_start+1..t_end） ---
                covered_ok = True
                
                # 因為第一次選用的 (sat,ch) 已經確保了 W 時槽的可見性，所以直接分配
                step = self.W
                t = t_start
                while t <= t_end:
                    for w in range(step):
                        tt = t + w
                        if tt > t_end:
                            break
                        
                        # 此處只需檢查是否超過 t_end，可見性在選擇時已保證
                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)
                        user_path.append((current_sat, current_ch, tt))
                        data_rate_rows.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0, False, None))
                        if dr and dr > 0:
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            user_reward += compute_score(self.params, m_s_t, dr, current_sat)

                    if t + step > t_end:
                        t += step
                        break
                    
                    t += step
                    
                    # 檢查是否允許換手，並嘗試找更好的選擇
                    can_handover = (t - last_ho_time >= self.W)
                    did_handover = False

                    best_sat_ho, best_ch_ho, best_score_ho = current_sat, current_ch, float("-inf")
                    
                    if can_handover:
                        vsats = list(self._visible_sats_for_user(t, user_id))
                        self.rng.shuffle(vsats)
                        for sat in vsats:
                            ch_list = list(tmp_sat_dict.get(sat, {}).keys())
                            self.rng.shuffle(ch_list)
                            for ch in ch_list:
                                if tmp_sat_dict[sat][ch] != 0:
                                    continue
                                end_req = min(t_end, t + self.W - 1)
                                can_cover = True
                                for tt in range(t, end_req + 1):
                                    if sat not in self._visible_sats_for_user(tt, user_id):
                                        can_cover = False
                                        break
                                if not can_cover:
                                    continue
                                
                                _, dr0 = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                                if not dr0:
                                    continue
                                m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                score0 = compute_score(self.params, m_s_t, dr0, sat) + 1e-9 * self.rng.random()
                                if score0 > best_score_ho:
                                    best_score_ho = score0
                                    best_sat_ho, best_ch_ho = sat, ch

                        if (best_sat_ho is not None) and (best_sat_ho != current_sat or best_ch_ho != current_ch):
                            current_sat, current_ch = best_sat_ho, best_ch_ho
                            last_ho_time = t
                            is_first_handover = False
                            did_handover = True
                            
                # ===== 成功/失敗落地 =====
                full_len = (t_end - t_start + 1)
                covered_len = len({tt for _, _, tt in user_path})

                if covered_len != full_len:
                    for tt in range(t_start, t_end + 1):
                        _mark_block(self.data_rates, user_id, tt, BLOCK_NO_FEASIBLE)
                    self.position[user_id] = []
                    continue

                self.position[user_id] = user_path
                for s, c in set((s, c) for s, c, _ in user_path):
                    tmp_sat_dict[s][c] += 1
                self.data_rates.extend(data_rate_rows)
                total_reward += user_reward

                active_user_paths.append({
                    "user_id": user_id,
                    "path": user_path,
                    "t_end": t_end
                })

        self.reward = total_reward
        
    def _run_greedy_path(self, user_id, t_start, t_end, tmp_sat_dict):
        """對單一使用者跑 greedy 路徑（和 generate_fast_path 內部邏輯一致的簡化版）。"""
        user_path = []
        data_rate_records = []
        user_reward = 0
        t = t_start
        current_sat, current_ch = None, None
        last_ho_time = t_start
        is_first_handover = True

        # === 第一次選擇 ===
        best_sat, best_ch, best_score, best_dr = None, None, float("-inf"), 0.0
        for sat in self._visible_sats_for_user(t_start, user_id):
            for ch, state in tmp_sat_dict.get(sat, {}).items():
                if state != 0:
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                if dr and dr > 0:
                    score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)
                    score += 1e-9 * self.rng.random()
                    if score > best_score:
                        best_score = score
                        best_sat, best_ch, best_dr = sat, ch, dr

        if best_sat is None:
            return [], [], 0

        current_sat, current_ch = best_sat, best_ch
        user_path.append((current_sat, current_ch, t_start))
        data_rate_records.append((user_id, t_start, current_sat, current_ch, best_dr, False, None))
        user_reward += best_score

        # === 後續時間 ===
        t += 1
        while t <= t_end:
            can_ho = is_first_handover or (t - last_ho_time >= self.W)
            did_ho = False
            best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

            if can_ho:
                vsats = list(self._visible_sats_for_user(t, user_id))
                self.rng.shuffle(vsats)
                for sat in vsats:
                    chs = list(tmp_sat_dict.get(sat, {}).keys())
                    self.rng.shuffle(chs)
                    for ch in chs:
                        if tmp_sat_dict[sat][ch] != 0:
                            continue
                        if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):
                            continue
                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                        if dr and dr > 0:
                            score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)
                            score += 1e-9 * self.rng.random()
                            if score > best_score:
                                best_score = score
                                best_sat, best_ch = sat, ch
                if (best_sat != current_sat) or (best_ch != current_ch):
                    current_sat, current_ch = best_sat, best_ch
                    last_ho_time = t
                    is_first_handover = False
                    did_ho = True

            step = self.W if did_ho else 1
            for w in range(step):
                tt = t + w
                if tt > t_end:
                    break
                if not did_ho and current_sat not in self._visible_sats_for_user(tt, user_id):
                    break
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)
                user_path.append((current_sat, current_ch, tt))
                data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0, False, None))
                if dr and dr > 0:
                    user_reward += compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, current_sat)
            t += step

        return user_path, data_rate_records, user_reward

    # ---------- 重新計分（用於 crossover/mutation 後） ----------
    def rebuild_from_position(self):
        """依照 self.position 重建 data_rates 與 reward（只算成功的 rows；不補 blocked）。"""
        self.data_rates = []
        self.reward = 0.0
        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        df_ts = self.user_df[["user_id", "t_start", "t_end"]].copy().sort_values("t_start")
        active_user_paths = []

        for _, row in df_ts.iterrows():
            user_id = int(row["user_id"])
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])

            to_remove = []
            for old in active_user_paths:
                if old["t_end"] < t_start:
                    for s, c in set((s, c) for s, c, _ in old["path"]):
                        tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)
                    to_remove.append(old)
            for u in to_remove:
                active_user_paths.remove(u)

            path = self.position.get(user_id, [])
            if not path:
                continue
            path = sorted(path, key=lambda x: x[2])

            user_reward = 0.0
            used_pairs = set()

            for sat, ch, t in path:
                if sat not in self._visible_sats_for_user(t, user_id):
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                self.data_rates.append((user_id, t, sat, ch, dr if dr else 0.0, False, None))
                if dr and dr > 0:
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t, dr, sat)
                    used_pairs.add((sat, ch))

            for s, c in used_pairs:
                tmp_sat_dict[s][c] = tmp_sat_dict[s].get(c, 0) + 1

            total_reward += user_reward
            active_user_paths.append({"user_id": user_id, "path": path, "t_end": t_end})

        self.reward = total_reward

class GeneticAlgorithm:
    """GA 主流程：初始化 → 選擇 → 交配 → 變異 → 重新計分 → 更新最佳個體。"""
    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.population_size = population_size
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.W = W
        self.path_loss = path_loss
        self.params = params
        self.sat_channel_dict = sat_channel_dict  # 加上 sat_channel_dict 參數
        self.seed_base = seed or random.randint(0, 999999)

        # 初始化族群（每個個體用 greedy-like 生成初始解；其中已含 blocked 記錄）
        self.initialize_population()
        
    def initialize_population(self):
        """初始化族群，每個個體都使用不同的隨機初始解。"""
        self.population = []
        for i in range(self.population_size):
            individual = Individual(
                self.user_df, self.access_matrix, self.W, self.path_loss,
                copy.deepcopy(self.sat_channel_dict),  # 確保每個個體使用獨立的通道狀態副本
                self.params, seed=self.seed_base + i * 7919
            )
            self.population.append(individual)

        self.population.sort(key=lambda x: x.reward, reverse=True)
        self.best_individual = copy.deepcopy(self.population[0])

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        for gen in range(generations):
            next_gen = self.population[:elite_size]
            while len(next_gen) < self.population_size:
                p1, p2 = self.tournament_selection(), self.tournament_selection()
                child = self.crossover(p1, p2)
                self.mutate(child, mutation_rate)
                next_gen.append(child)

            for ind in next_gen:
                ind.rebuild_from_position()

            self.population = sorted(next_gen, key=lambda x: x.reward, reverse=True)
            if self.population[0].reward > self.best_individual.reward:
                self.best_individual = copy.deepcopy(self.population[0])

    def tournament_selection(self, k=3):
        return max(random.sample(self.population, k), key=lambda x: x.reward)

    def crossover(self, p1, p2):
        child = copy.deepcopy(p1)
        for uid in child.position:
            if random.random() < 0.5:
                child.position[uid] = copy.deepcopy(p2.position.get(uid, []))
        # 子代不需要立即 rebuild，因為會在 evolve 迴圈中統一處理
        return child

    def mutate(self, individual, mutation_rate):
        mutated = False
        for user in individual.user_df.itertuples():
            if random.random() < mutation_rate:
                uid = int(user.user_id)
                t_start, t_end = int(user.t_start), int(user.t_end)
                snapshot = self._build_snapshot(individual, uid)
                
                # 使用 DP 重新計算路徑
                path, reward, success, data_rates = run_dp_path_for_user(
                    uid, t_start, t_end, self.W, self.access_matrix,
                    self.path_loss, snapshot, self.params
                )
                
                if success:
                    individual.position[uid] = path
                    mutated = True
        # 突變後不需要立即 rebuild，因為會在 evolve 迴圈中統一處理
        return mutated # 可選：返回是否發生突變

    def _build_snapshot(self, individual, exclude_user_id):
        tmp = copy.deepcopy(self.sat_channel_dict)
        for uid, path in individual.position.items():
            if uid == exclude_user_id:
                continue
            for s, c, _ in path:
                tmp[s][c] += 1
        return tmp

    def export_best_result(self):
        """
        All-or-nothing（含可見性驗證）：
        - 對每位 user：若路徑未覆蓋 [t_start, t_end] 任一時槽，或任一格的 (sat,t) 對該 user 不可見，
            則整段視為失敗 → 全段 blocked（no_feasible）。
        - 完全覆蓋且每格可見 → 視為成功，逐格輸出成功列（data_rate 先放 0，之後由 recompute_all_data_rates 覆寫）。
        """
        best = self.best_individual

        cols = ["user_id", "time", "sat", "channel", "data_rate", "blocked", "reason"]
        rows = []
        load_by_time = defaultdict(lambda: defaultdict(int))
        all_user_paths = []
        results = []

        # 小工具：拿某 user 在時槽 t 的可見衛星清單（支援字串/清單 & per-user 欄位）
        def _visible_list_at_t(uid, t_abs):
            row = self.access_matrix[t_abs]
            if "visible_sats_for_user" in row:
                vs = row["visible_sats_for_user"].get(uid, row.get("visible_sats", []))
            else:
                vs = row.get("visible_sats", [])
            if isinstance(vs, str):
                try:
                    vs = ast.literal_eval(vs)
                except Exception:
                    vs = []
            return vs

        for row in self.user_df.itertuples(index=False):
            uid     = int(row.user_id)
            t_start = int(row.t_start)
            t_end   = int(row.t_end)
            need_len = t_end - t_start + 1

            # 這位使用者在最優個體的路徑（(sat, ch, t)）
            path = best.position.get(uid, []) or []
            path_sorted = sorted(path, key=lambda x: x[2])

            # 建 t -> (sat, ch) 對應表
            t2sc = {t: (s, c) for (s, c, t) in path_sorted}

            # (1) 覆蓋檢查：是否每個 t 都有分配
            full_cover = (len(t2sc) == need_len) and all(t in t2sc for t in range(t_start, t_end + 1))

            # (2) 可見性檢查：每個 t 的衛星 s 是否在該 user 的可見清單
            visible_ok = full_cover and all(
                (t2sc[t][0] in _visible_list_at_t(uid, t)) for t in range(t_start, t_end + 1)
            )

            if full_cover and visible_ok:
                # 成功：逐格輸出（data_rate 先放 0；稍後 recompute_all_data_rates 會覆寫）
                for tt in range(t_start, t_end + 1):
                    s, c = t2sc[tt]
                    rows.append((uid, tt, s, c, 0.0, False, None))
                    load_by_time[tt][s] += 1

                all_user_paths.append({
                    "user_id": uid,
                    "path": path_sorted,
                    "t_begin": t_start,
                    "t_end": t_end,
                    "success": True,
                    "reward": None
                })
                results.append({"user_id": uid, "reward": None, "success": True})
            else:
                # 失敗：只要缺一格或任一格不可見 → 全段 blocked
                for tt in range(t_start, t_end + 1):
                    rows.append((uid, tt, None, None, 0.0, True, BLOCK_NO_FEASIBLE))

                all_user_paths.append({
                    "user_id": uid,
                    "path": [],
                    "t_begin": t_start,
                    "t_end": t_end,
                    "success": False,
                    "reward": 0.0
                })
                results.append({"user_id": uid, "reward": None, "success": False})

        df_rates = pd.DataFrame(rows, columns=cols).sort_values(["user_id", "time"]).reset_index(drop=True)
        return pd.DataFrame(results), pd.DataFrame(all_user_paths), load_by_time, df_rates
