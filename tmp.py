import pandas as pd
from collections import defaultdict
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels
)

# 與其他方法一致的 blocking reason
BLOCK_NO_PAIR      = "no_pair_available"
BLOCK_NO_FEASIBLE  = "no_feasible_W_or_conflict"

####################################################################################################
def run_dp_path_for_user(
    user_id: int,
    t_start: int,
    t_end: int,
    W: int,
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict
):
    """
    DP + state counter，狀態為 (sat, ch)，嘗試滿足「換手後至少連續 W slot」。
    - 僅使用 channel==0（1 或 2 一律不能用）
    - 失敗時針對每個 time 標註 blocked 與原因
    """

    # 先掃一遍「逐時槽的個別可行 pair」（不含 W 連續性，只看當下是否可用）
    feasible_pairs_by_t = {}
    for t_abs in range(t_start, t_end + 1):
        feas = []
        visible_sats = access_matrix[t_abs]["visible_sats"]
        # 若 visible_sats 可能是字串，可在這裡 ast.literal_eval 轉成 list
        for sat in visible_sats:
            if sat not in sat_channel_dict:
                continue
            for ch, occ in sat_channel_dict[sat].items():
                if occ != 0:       # 只允許 0（1、2 都不能分配）
                    continue
                if (sat, t_abs) not in path_loss:
                    continue
                SINR, dr = compute_sinr_and_rate(params, path_loss, sat, t_abs, sat_channel_dict, ch)
                if dr is None:
                    continue
                feas.append((sat, ch, dr))
        feasible_pairs_by_t[t_abs] = feas

    # 收集 t_start~t_end 期間曾經「個別可行」的 (sat, ch) 當作 DP 的全集
    pairs_in_period = {(sat, ch) for t_abs in range(t_start, t_end + 1)
                       for (sat, ch, _) in feasible_pairs_by_t[t_abs]}
    all_pairs = sorted(pairs_in_period)
    P = len(all_pairs)
    T_len = t_end - t_start + 1
    NEG_INF = -1e18

    if P == 0:
        # 整段期間完全沒有任何可用 pair → 全部標記 BLOCK_NO_PAIR
        data_rate_records = []
        for t_abs in range(t_start, t_end + 1):
            data_rate_records.append({
                "user_id": user_id, "time": t_abs, "sat": None, "channel": None,
                "data_rate": 0.0, "blocked": True, "reason": BLOCK_NO_PAIR
            })
        return [], 0.0, False, data_rate_records

    # dp[t_rel][p_idx][c] ； t_rel: 相對於 t_start 的時間索引
    dp = [[[NEG_INF] * (W + 1) for _ in range(P)] for _ in range(T_len)]
    parent = [[[None] * (W + 1) for _ in range(P)] for _ in range(T_len)]
    data_rate_cache = {}

    # 初始化 (t_rel = 0)
    t_abs0 = t_start
    feas0 = feasible_pairs_by_t[t_abs0]
    for p_idx, (sat, ch) in enumerate(all_pairs):
        # 當下這一格是否在「個別可行清單」內
        #（可視/可用/能算速率）
        if not any((sat == s and ch == c) for (s, c, _) in feas0):
            continue
        # 估分
        SINR, dr = compute_sinr_and_rate(params, path_loss, sat, t_abs0, sat_channel_dict, ch)
        if dr is None:
            continue
        m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
        score = compute_score(params, m_s_t, dr, sat)
        dp[0][p_idx][W] = score
        parent[0][p_idx][W] = (-1, -1, -1)  # 回溯用的起點標記
        data_rate_cache[(0, p_idx)] = dr

    # DP 遞推
    for t_rel in range(1, T_len):
        t_abs = t_start + t_rel
        feas_now = feasible_pairs_by_t[t_abs]

        for p_idx, (sat, ch) in enumerate(all_pairs):
            # 不在當下個別可行清單 → 跳過
            if not any((sat == s and ch == c) for (s, c, _) in feas_now):
                continue

            SINR, dr = compute_sinr_and_rate(params, path_loss, sat, t_abs, sat_channel_dict, ch)
            if dr is None:
                continue
            m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
            score = compute_score(params, m_s_t, dr, sat)
            data_rate_cache[(t_rel, p_idx)] = dr

            # 1) 換手（c=0）：上一時槽必須是「其他衛星」且 c=W（滿足 W 之後才能換）
            best_val = NEG_INF
            best_parent = None
            for sp in range(P):
                if all_pairs[sp][0] == sat:  # 同衛星換 ch 不允許
                    continue
                if dp[t_rel - 1][sp][W] != NEG_INF:
                    val = dp[t_rel - 1][sp][W] + score
                    if val > best_val:
                        best_val = val
                        best_parent = (sp, t_rel - 1, W)
            dp[t_rel][p_idx][0] = best_val
            parent[t_rel][p_idx][0] = best_parent

            # 2) 無換手（同一 (sat,ch) 延續）
            #    - 從 c=0 → c=1（剛換手後的第一步）
            if dp[t_rel - 1][p_idx][0] != NEG_INF:
                dp[t_rel][p_idx][1] = dp[t_rel - 1][p_idx][0] + score
                parent[t_rel][p_idx][1] = (p_idx, t_rel - 1, 0)

            #    - 從 c=k-1 或 c=k → c=k（1<k<=W）
            for c in range(2, W + 1):
                candidates = []
                if dp[t_rel - 1][p_idx][c - 1] != NEG_INF:
                    candidates.append((dp[t_rel - 1][p_idx][c - 1], (p_idx, t_rel - 1, c - 1)))
                if dp[t_rel - 1][p_idx][c] != NEG_INF:
                    candidates.append((dp[t_rel - 1][p_idx][c], (p_idx, t_rel - 1, c)))
                if candidates:
                    best_val2, best_parent2 = max(candidates, key=lambda x: x[0])
                    dp[t_rel][p_idx][c] = best_val2 + score
                    parent[t_rel][p_idx][c] = best_parent2

    # 終點：找最大值
    max_reward = NEG_INF
    end_state = None
    for p_idx in range(P):
        for c in range(W + 1):
            if dp[T_len - 1][p_idx][c] > max_reward:
                max_reward = dp[T_len - 1][p_idx][c]
                end_state = (p_idx, T_len - 1, c)

    # 若 DP 失敗：逐時槽標記 blocked（依「是否有個別可行 pair」區分原因）
    if max_reward == NEG_INF:
        data_rate_records = []
        for t_abs in range(t_start, t_end + 1):
            if len(feasible_pairs_by_t[t_abs]) == 0:
                reason = BLOCK_NO_PAIR
            else:
                reason = BLOCK_NO_FEASIBLE
            data_rate_records.append({
                "user_id": user_id, "time": t_abs, "sat": None, "channel": None,
                "data_rate": 0.0, "blocked": True, "reason": reason
            })
        return [], 0.0, False, data_rate_records

    # 回溯成功路徑
    path = []
    data_rate_records = []
    cur = end_state  # (p_idx, t_rel, c)
    while cur and cur[0] != -1:
        p_idx, t_rel, c = cur
        sat, ch = all_pairs[p_idx]
        t_abs = t_start + t_rel
        dr = data_rate_cache.get((t_rel, p_idx), 0.0)

        path.append((sat, ch, t_abs))
        data_rate_records.append({
            "user_id": user_id, "time": t_abs, "sat": sat, "channel": ch,
            "data_rate": dr, "blocked": False, "reason": None
        })

        cur = parent[t_rel][p_idx][c]

    path.reverse()
    # 若你希望 data_rate_records 也按時間排序：
    data_rate_records.sort(key=lambda x: x["time"])
    return path, max_reward, True, data_rate_records


####################################################################################################
def run_dp_per_W(
    user_df: pd.DataFrame,
    access_matrix: list,
    path_loss: dict,
    sat_load_dict_backup: dict,
    params: dict,
    W: int = 2
):
    """
    逐使用者（依 t_start 排序）跑 DP，逐段更新 sat_load_dict。
    - 僅允許 channel==0；1/2 都不能用
    - df_data_rates 會包含 blocked/reason 欄位，方便後續 compute_blocking_stats
    """
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_load_dict_backup.items()}
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True)

    active_user_paths = []
    all_user_paths = []
    results = []
    load_by_time = defaultdict(lambda: defaultdict(int))
    all_user_data_rates = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])

        # 釋放已完成的使用者（維持你原本邏輯）
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path:
                    unique_sat_ch = set((s, c) for s, c, _ in path)
                    for sat, ch in unique_sat_ch:
                        sat_load_dict[sat][ch] = max(0, sat_load_dict[sat][ch] - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # 計算路徑（DP）
        path, reward, success, data_rate_records = run_dp_path_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            W=W,
            access_matrix=access_matrix,
            path_loss=path_loss,
            sat_channel_dict=sat_load_dict,
            params=params
        )

        # 更新負載（只對實際用到的 (sat,ch) 做 +1）
        if path:
            unique_sat_ch = set((s, c) for s, c, _ in path)
            for sat, ch in unique_sat_ch:
                sat_load_dict[sat][ch] += 1
            for s, c, t in path:
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1

        all_user_data_rates.extend(data_rate_records)

        # 紀錄結果（保持你原本格式）
        if success:
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": success,
                "reward": reward
            })

        all_user_paths.append({
            "user_id": user_id,
            "path": path,
            "t_begin": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward
        })

        results.append({
            "user_id": user_id,
            "reward": reward if success else None,
            "success": success
        })

    # 直接用 dict 紀錄（含 blocked/reason），不要指定 columns
    df_data_rates = pd.DataFrame(all_user_data_rates)
    return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
