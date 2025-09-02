import networkx as nx                      # 匯入 NetworkX，用來建圖與跑最短路（Dijkstra）
import pandas as pd                        # 匯入 pandas，最後整理輸出用
import ast                                 # 匯入 ast，將字串型態的可見衛星列表安全轉為 Python 物件
from collections import defaultdict        # 匯入 defaultdict，方便建立巢狀 dict
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels
# 從工具模組匯入：計 SINR/速率、算分數、與由 channel 狀態計每顆衛星當前負載的工具

# === Blocking 理由 ===
BLOCK_NO_PAIR      = "no_pair_available"           # 當前快照（此 t）下沒有任何可用的 (sat, ch) 可以起段
BLOCK_NO_FEASIBLE  = "no_feasible_W_or_conflict"   # 有可見衛星/空閒通道，但無法串成完整 [t_begin, t_end] 的可行路徑

def mark_block(rows, uid, t, reason):
    rows.append({                                    # 在 data rate 列表增加一筆「被擋」紀錄
        "user_id": uid, "time": t, "sat": None, "channel": None,  # 無衛星/通道
        "data_rate": 0.0, "blocked": True, "reason": reason       # 速率為 0，blocked=True，附上原因
    })

def run_mslb_sequential(
    user_df,                                         # 使用者表（欄位至少包含 user_id, t_start, t_end）
    access_matrix: list,                              # 每個 time slot 的可見資訊（list[dict]）
    path_loss: dict,                                  # Path loss 查表：(sat, t) -> PL
    sat_channel_dict: dict,                           # 全域衛星通道狀態：0=可用, 1=本 cell 佔用, 2=背景保留(不可分配/不干擾)
    params: dict,                                     # 系統參數（EIRP、雜訊等）
    W: int,                                           # 交接最短持續長度
    require_first_W: bool = False                     # ✅ 若為 True，第一段也必須 L >= W（否則第一段允許 L>=1）
):
    """
    逐一處理（sequential）版 MSLB：
      - 逐時槽 t 迭代
      - 每個 t 先釋放 t_end+1 到期的使用者（把其用過的 (sat,ch) 從 1 → 0）
      - 對於 t_start==t 的新進使用者，照 user_df 的順序一個一個嘗試：
          * 用當前快照建圖 + Dijkstra 找該 user 的最佳完整路徑
          * 找不到 → 立刻標 blocked（no_pair / no_feasible），繼續下一個 user
          * 找得到 → 立刻落地：把整段用到的 (sat,ch) 設為 1；寫入 data_rate/load/path
      - 不做同批“贏家挑選”，完全是 for 迴圈先到先得
    """
    T = len(access_matrix)                             # 總 time slots 數

    # 保證處理順序穩定：同時刻進入者，依 user_id 排序
    user_df = user_df.sort_values(by=["t_start", "user_id"]).reset_index(drop=True)

    results = []                                       # 存每位使用者的成功/失敗與 reward
    all_user_paths = []                                # 存每位使用者的實際路徑（展成 (s,c,t)）
    active_users = []                                  # 正在服務中的使用者（到期需釋放資源）
    all_user_data_rates = []                           # 每個 slot 的資料率記錄（也包含 blocked 標記）
    load_by_time = defaultdict(lambda: defaultdict(int))  # 各時槽各衛星的使用人數累計

    # 便捷索引：user_id -> 該列在 user_df 的 index
    row_of = {int(row["user_id"]): i for i, row in user_df.iterrows()}

    # ---- helpers ------------------------------------------------------------
    def _visible_list_at_t(uid, t_abs):
        """回傳 list[str] 的可見衛星；若 access_matrix[t]['visible_sats'] 是字串，轉為 Python list。"""
        vs = access_matrix[t_abs].get("visible_sats_for_user", {}).get(
            uid, access_matrix[t_abs]["visible_sats"]                 # 若沒有 per-user，可見集合用全域 visible_sats
        )
        if isinstance(vs, str):                                       # 有些資料存成字串（例如 "['S1','S2']"）
            try:
                vs = ast.literal_eval(vs)                             # 安全地轉回 list 物件
            except Exception:
                vs = []                                               # 解析失敗則視為不可見
        return vs

    def _visible_pairs_at_t(uid, t_abs, snap):
        vis = []                                                      # 收集此時刻可用的 (sat, ch)
        visible_sats = _visible_list_at_t(uid, t_abs)                 # 先拿可見衛星清單
        for s in visible_sats:                                        # 逐一檢查每顆可見衛星
            if s not in snap:                                         # 衛星名不在快照中（理論上不會），跳過
                continue
            for ch, state in snap[s].items():                         # 走訪該衛星所有通道
                if state == 0:                                        # 只有 state==0（可用）才能拿來當起始段
                    vis.append((s, ch))                               # 收集可用 pair
        return vis

    def _run_length(uid, s, ch, t0, t_end, snap):
        """從 t0 起，用 (s,ch) 的最長連續可用長度（需同時滿足：此 user 可見、通道 state==0）。"""
        L = 0                                                         # 目前累積長度
        for tt in range(t0, t_end + 1):                               # 從 t0 一直試到 t_end
            visible_sats = _visible_list_at_t(uid, tt)                # 在 tt 時刻的可見衛星
            if (s not in visible_sats) or (snap.get(s, {}).get(ch, 1) != 0):
                # 若該衛星在此時不可見，或該通道在此時不是 0（被占或背景保留），就停止延伸
                break
            L += 1                                                    # 可以延伸，長度 +1
        return L

    def _segment_score(uid, s, ch, t0, L, snap):
        """把一段長度 L 的連續使用（自 t0 起）的總分數加總。m_s_t 使用當下快照。"""
        total = 0.0                                                   # 累積分數
        m_s_t = update_m_s_t_from_channels(snap, snap.keys())         # 由快照計出每顆衛星的當前負載（非 0 算 1）
        for tt in range(t0, t0 + L):                                  # 逐 slot 計分，較保守但簡潔
            _, dr = compute_sinr_and_rate(params, path_loss, s, tt, snap, ch)  # 計算此 slot 的資料率
            if dr is None:                                            # 無法計算（path_loss 缺？）則當 0
                dr = 0.0
            total += compute_score(params, m_s_t, dr, s)              # (1 - alpha*load)*data_rate
        return total

    def build_graph_for_user(uid, t_begin, t_end, snap, require_first_W=False, Lambda=1e9):
        """建立 user 的 segment DAG 圖：
           - 若 require_first_W=True，第一段長度也需 L>=W；否則第一段允許 L>=1
           - 其餘段長度一律 L>=W
           - 只允許 state==0 的通道
        """
        G = nx.DiGraph()                                              # 建立有向圖
        G.add_node("START"); G.add_node("END")                        # 加上起點與終點

        segments = []                                                 # 收集所有可用的 segment 節點
        for t0 in range(t_begin, t_end + 1):                          # 嘗試從每個起始時間 t0 建段
            # ✅ 第一段門檻可開關
            min_L = (W if (require_first_W or t0 != t_begin) else 1)
            for (s, ch) in _visible_pairs_at_t(uid, t0, snap):        # 先拿此刻可見且 state==0 的所有 (sat,ch)
                L = _run_length(uid, s, ch, t0, t_end, snap)          # 往後延伸計算最長連續長度
                if L >= min_L:                                        # 若長度達門檻，就是一個合法 segment
                    segments.append((s, ch, t0, L))                   # 記錄 segment 參數
                    G.add_node(("SEG", s, ch, t0, L))                 # 圖中加一個節點

        # 早退：若沒有以 t_begin 起始的段，或沒有剛好覆蓋到 t_end 的段，無法湊成完整路徑
        has_start = any(t0 == t_begin for _, _, t0, L in segments)    # 是否有起點段
        has_end   = any(t0 + L - 1 == t_end for _, _, t0, L in segments)  # 是否有段正好覆蓋到 t_end
        if not has_start or not has_end:                              # 只要缺一側，代表不可達
            return G                                                  # 回傳僅有 START/END 的空圖（Dijkstra 會因此失敗）

        # START → 起始段：把所有 t0==t_begin 的段，接在 START 後面（權重為 Lambda - 段分數）
        for (s, ch, t0, L) in segments:
            if t0 != t_begin:                                         # 只連到真正從 t_begin 開始的段
                continue
            seg_score = _segment_score(uid, s, ch, t0, L, snap)       # 計算此段的累計分數
            G.add_edge("START", ("SEG", s, ch, t0, L), weight=Lambda - seg_score)  # 最短路 => 最大化分數

        # 段接段（只有換衛星才接；同衛星換通道不允許）
        by_t0 = defaultdict(list)                                     # 將段以起始時間 t0 分組
        for s, ch, t0, L in segments:
            by_t0[t0].append((s, ch, L))

        for (s1, ch1, t0, L) in segments:                             # 對每一個段，找它結束後 t_next 可接的下一段
            t_next = t0 + L                                           # 下一段的起始時間必須正好是本段結束的下一個 slot
            if t_next > t_end:                                        # 超過範圍就不用接
                continue
            for (s2, ch2, L2) in by_t0.get(t_next, []):               # 找所有起於 t_next 的段
                if s2 == s1:                                          # 同衛星不接（避免同衛星換 ch）
                    continue
                seg_score = _segment_score(uid, s2, ch2, t_next, L2, snap)  # 該下一段的分數
                G.add_edge(("SEG", s1, ch1, t0, L),                   # 從前段連到後段
                           ("SEG", s2, ch2, t_next, L2),
                           weight=Lambda - seg_score)                 # 權重同理（越大分越小權）

        # 能覆蓋到 t_end 的段 → 接到 END（權重 0）
        for (s, ch, t0, L) in segments:
            if t0 + L - 1 == t_end:                                   # 恰好覆蓋到最後一個 slot
                G.add_edge(("SEG", s, ch, t0, L), "END", weight=0.0)  # 接到終點

        return G                                                      # 回傳建好的圖

    def solve_one(uid, snap):
        """用當前快照 snap 幫單一使用者求完整最短路；回傳展開路徑與記錄。"""
        r = user_df.loc[row_of[uid]]                                  # 找出該使用者在 user_df 的列
        t_begin = int(r["t_start"]); t_end = int(r["t_end"])          # 取該使用者的起訖時刻

        G = build_graph_for_user(uid, t_begin, t_end, snap, require_first_W=require_first_W)  # ✅ 傳入參數
        try:
            dist, paths = nx.single_source_dijkstra(G, "START", weight="weight")  # 用 Dijkstra 找 START→所有點的最短路
        except Exception:                                             # 若建圖或演算法有例外
            return [], 0.0, [], t_begin, t_end, set(), set()          # 當作不可達

        if "END" not in paths or not paths["END"]:                   # 若沒有到 END 的路徑
            return [], 0.0, [], t_begin, t_end, set(), set()          # 視為不可達

        nodes = paths["END"]                                          # 取出最短路上的節點序列

        total_reward = 0.0                                            # 該使用者的總分
        data_rows = []                                                # 該使用者的每 slot 速率記錄
        expanded = []                                                 # 該使用者展開後的 (s,c,tt) 路徑
        used_pairs = set()                                            # 該使用者使用過的 (s,c) 集合（用來鎖資源與最後釋放）
        used_slots = set()                                            # 該使用者使用過的 (s,c,tt) 集合（若要做衝突檢查可用）

        for node in nodes:                                            # 走訪最短路上的每個節點
            if node in ("START", "END"):                              # 起點/終點略過
                continue
            _, s, ch, t0, L = node                                    # 拆解 segment 節點的內容
            used_pairs.add((s, ch))                                   # 紀錄此段所用 (s,c)
            for tt in range(t0, t0 + L):                              # 依段長 L 展開到每個 time slot
                m_s_t = update_m_s_t_from_channels(snap, snap.keys()) # 用快照計負載（非 0 算 1）
                _, dr = compute_sinr_and_rate(params, path_loss, s, tt, snap, ch)  # 算本 slot 的資料率
                if dr is None:                                        # 無法算時視為 0
                    dr = 0.0
                total_reward += compute_score(params, m_s_t, dr, s)   # 分數累加
                data_rows.append({                                    # 寫入本 slot 的資料率記錄
                    "user_id": uid, "time": tt, "sat": s, "channel": ch,
                    "data_rate": dr, "blocked": False, "reason": None
                })
                expanded.append((s, ch, tt))                          # 展開路徑（之後會累積到 load_by_time）
                used_slots.add((s, ch, tt))                           # 記錄此 slot 有使用（若要做衝突判斷）

        return expanded, total_reward, data_rows, t_begin, t_end, used_pairs, used_slots
    # ------------------------------------------------------------------------

    # === 主迴圈：逐時槽 ===
    for t in range(T):                                                # 逐一走訪每個 time slot
        # 1) 釋放到期（t == t_end+1）
        to_remove = []                                                # 暫存此刻要移除的 active users
        for au in active_users:                                       # 檢查目前所有活躍使用者
            if t == au["t_end"] + 1:                                  # 若到了他們的釋放時刻（服務剛結束的下一個 slot）
                for (s, ch) in au["used_pairs"]:                      # 把他們占用過的每個 (s,c)
                    if s in sat_channel_dict and ch in sat_channel_dict[s]:
                        sat_channel_dict[s][ch] = 0                   # 直接還原為 0（釋放）
                to_remove.append(au)                                  # 標記這個 user 可移除
        for au in to_remove:                                          # 從活躍清單移除
            active_users.remove(au)

        # 2) 這個 t 的新進使用者（保持 user_df 順序）
        entrants = user_df[user_df["t_start"] == t]["user_id"].astype(int).tolist()
        if not entrants:
            continue

        # 3) 逐一處理（先到先得）
        for uid in entrants:
            snapshot = {s: chs.copy() for s, chs in sat_channel_dict.items()}  # 用“當前”狀態快照試算

            expanded, reward, data_rows, t_b, t_e, used_pairs, used_slots = solve_one(uid, snapshot)

            if not expanded:                                          # 找不到完整路徑 → 當場擋掉
                vis_now = _visible_pairs_at_t(uid, t, snapshot)       # 檢查當前 t 是否有任何候選 pair
                reason = BLOCK_NO_PAIR if len(vis_now) == 0 else BLOCK_NO_FEASIBLE

                # 把整段 [t_b, t_e] 都展開成 blocked（與 DP/greedy 的統計尺度一致）
                for tt in range(t_b, t_e + 1):
                    mark_block(all_user_data_rates, uid, tt, reason)

                all_user_paths.append({
                    "user_id": uid, "path": [],
                    "t_begin": t_b, "t_end": t_e,
                    "success": False, "reward": 0.0
                })
                results.append({"user_id": uid, "success": False, "reward": None})
                continue

            # 找得到 → 立刻落地：鎖通道（固定設為 1）、寫 load/data/path
            for (s, ch) in used_pairs:
                if s in sat_channel_dict and ch in sat_channel_dict[s]:
                    sat_channel_dict[s][ch] = 1       # ✅ 固定設為 1（避免 +=1 造成語義混亂）

            for (s, ch, tt) in expanded:
                load_by_time[tt][s] += 1

            all_user_data_rates.extend(data_rows)

            active_users.append({
                "user_id": uid,
                "t_end": t_e,
                "used_pairs": set(used_pairs)
            })

            all_user_paths.append({
                "user_id": uid, "path": expanded,
                "t_begin": t_b, "t_end": t_e,
                "success": True, "reward": reward
            })
            results.append({"user_id": uid, "success": True, "reward": reward})

    # === 輸出 ===
    df_data_rates = pd.DataFrame(
        all_user_data_rates,
        columns=["user_id", "time", "sat", "channel", "data_rate", "blocked", "reason"]
    )

    results_df = pd.DataFrame(results)
    return results_df, all_user_paths, load_by_time, df_data_rates


# === 向後相容：保留舊名稱 ===
def run_mslb_batch(
    user_df,                                                          # 同 run_mslb_sequential
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict,
    W: int,
    require_first_W: bool = True                                     # ✅ 轉傳參數（預設 False）
):
    """Backward-compatible wrapper: batch 名稱指向 sequential 版本。"""
    return run_mslb_sequential(
        user_df=user_df,
        access_matrix=access_matrix,
        path_loss=path_loss,
        sat_channel_dict=sat_channel_dict,
        params=params,
        W=W,
        require_first_W=require_first_W
    )
