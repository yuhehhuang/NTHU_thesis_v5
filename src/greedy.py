import json
from collections import defaultdict, Counter
import pandas as pd
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility
)

BLOCK_NO_PAIR      = "no_pair_available"
BLOCK_NO_FEASIBLE  = "no_feasible_W_or_conflict"

def run_greedy_path_for_user(
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
    單一使用者的貪婪路徑規劃：
    - 僅在 channel==0 時才可分配（1 或 2 都不可）
    - 換手時需滿足未來 W-slot 連續可視；若不滿足 → 標 blocked 並 t+=1
    - 會寫入 blocked 紀錄（blocked=True, reason=...）
    """
    def _visible_list(t):
        vs = access_matrix[t]["visible_sats"]
        # 若你的 access_matrix 已是 list 就不需處理；這裡保險用
        if isinstance(vs, str):
            import ast
            return ast.literal_eval(vs)
        return vs

    current_sat, current_ch = None, None
    last_ho_time = t_start - W  # 讓第一個 slot 可以被視為「可換手」
    is_first_handover = True
    path = []
    total_reward = 0.0
    data_rate_records = []

    # ========== 初始 slot：同時檢查 W-slot 可視 ==========
    best_sat, best_ch, best_score, best_data_rate = None, None, -1.0, 0.0

    visible_sats = _visible_list(t_start)
    # 先看當下有沒有任何可分配 pair（==0）
    avail_pairs_exist = any(
        (sat in sat_channel_dict) and any(v == 0 for v in sat_channel_dict[sat].values())
        for sat in visible_sats
    )

    if not avail_pairs_exist:
        # 完全沒空位
        data_rate_records.append({
            "user_id": user_id, "time": t_start, "sat": None, "channel": None,
            "data_rate": 0.0, "blocked": True, "reason": BLOCK_NO_PAIR
        })
        return [], 0.0, False, data_rate_records

    # 有空位 → 挑能支撐 W-slot 的最佳 (sat,ch)
    for sat in visible_sats:
        if sat not in sat_channel_dict:
            continue
        for ch, occ in sat_channel_dict[sat].items():
            if occ != 0:  # 1 or 2 一律不可用
                continue
            # 檢查 W-slot 可視性（或到 t_end）
            if not check_visibility(pd.DataFrame(access_matrix), sat, t_start, min(t_end, t_start + W - 1)):
                continue
            if (sat, t_start) not in path_loss:
                continue
            SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t_start, sat_channel_dict, ch)
            if data_rate is None:
                continue
            m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
            score = compute_score(params, m_s_t, data_rate, sat)
            if score > best_score:
                best_score, best_sat, best_ch, best_data_rate = score, sat, ch, data_rate

    if best_sat is None:
        # 有空位但對這個 user 不可行（W-slot 可視or其他約束失敗）
        data_rate_records.append({
            "user_id": user_id, "time": t_start, "sat": None, "channel": None,
            "data_rate": 0.0, "blocked": True, "reason": BLOCK_NO_FEASIBLE
        })
        return [], 0.0, False, data_rate_records

    # 成功選到第一個 W-window
    current_sat, current_ch = best_sat, best_ch
    last_ho_time = t_start
    is_first_handover = False
    # 一次加入 W 個 slot
    t = t_start
    t_last = min(t + W - 1, t_end)
    for tt in range(t, t_last + 1):
        path.append((current_sat, current_ch, tt))
        _, dr = compute_sinr_and_rate(params, path_loss, current_sat, tt, sat_channel_dict, current_ch)
        data_rate_records.append({
            "user_id": user_id, "time": tt, "sat": current_sat, "channel": current_ch,
            "data_rate": (dr or 0.0), "blocked": False, "reason": None
        })
        total_reward += best_score
    t = t_last + 1

    # ========== 後續 slot ==========
    while t <= t_end:
        can_handover = (t - last_ho_time >= W)

        if not can_handover:
            # 仍在前一個 W-window，照抄
            path.append((current_sat, current_ch, t))
            _, dr = compute_sinr_and_rate(params, path_loss, current_sat, t, sat_channel_dict, current_ch)
            data_rate_records.append({
                "user_id": user_id, "time": t, "sat": current_sat, "channel": current_ch,
                "data_rate": (dr or 0.0), "blocked": False, "reason": None
            })
            total_reward += best_score  # 使用上一個 window 的 best_score
            t += 1
            continue

        # 可以換手：尋找新的 W-window
        best_sat, best_ch, best_score, best_data_rate = None, None, -1.0, 0.0

        visible_sats = _visible_list(t)
        avail_pairs_exist = any(
            (sat in sat_channel_dict) and any(v == 0 for v in sat_channel_dict[sat].values())
            for sat in visible_sats
        )

        if not avail_pairs_exist:
            # 完全沒空位 → 這個 slot 被擋
            data_rate_records.append({
                "user_id": user_id, "time": t, "sat": None, "channel": None,
                "data_rate": 0.0, "blocked": True, "reason": BLOCK_NO_PAIR
            })
            t += 1
            continue

        for sat in visible_sats:
            if sat not in sat_channel_dict:
                continue
            for ch, occ in sat_channel_dict[sat].items():
                if occ != 0:
                    continue
                if not check_visibility(pd.DataFrame(access_matrix), sat, t, min(t_end, t + W - 1)):
                    continue
                if (sat, t) not in path_loss:
                    continue
                SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_channel_dict, ch)
                if data_rate is None:
                    continue
                m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
                score = compute_score(params, m_s_t, data_rate, sat)
                if score > best_score:
                    best_score, best_sat, best_ch, best_data_rate = score, sat, ch, data_rate

        if best_sat is None:
            # 有空位，但無法滿足 W-slot 可視/其他約束 → 這個 slot 被擋
            data_rate_records.append({
                "user_id": user_id, "time": t, "sat": None, "channel": None,
                "data_rate": 0.0, "blocked": True, "reason": BLOCK_NO_FEASIBLE
            })
            t += 1
            continue

        # 成功找到新 window → 寫入 W 個 slot
        current_sat, current_ch = best_sat, best_ch
        last_ho_time = t
        t_last = min(t + W - 1, t_end)
        for tt in range(t, t_last + 1):
            path.append((current_sat, current_ch, tt))
            _, dr = compute_sinr_and_rate(params, path_loss, current_sat, tt, sat_channel_dict, current_ch)
            data_rate_records.append({
                "user_id": user_id, "time": tt, "sat": current_sat, "channel": current_ch,
                "data_rate": (dr or 0.0), "blocked": False, "reason": None
            })
            total_reward += best_score
        t = t_last + 1

    return path, total_reward, True, data_rate_records


def run_greedy_per_W(
    user_df: pd.DataFrame,
    access_matrix: list,
    path_loss: dict,
    sat_load_dict_backup: dict,
    params: dict,
    W: int = 4
):
    """
    批次（依 t_start 排序）Greedy：
    - 不使用 channel==2（不可分配）
    - 將每個 user 未能被服務的 slot 記為 blocked（由 run_greedy_path_for_user 直接產出）
    """
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_load_dict_backup.items()}
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True)

    active_user_paths = []     # 用於「釋放」舊用戶佔用（維持你原邏輯）
    all_user_paths = []
    results = []
    load_by_time = defaultdict(lambda: defaultdict(int))
    all_user_data_rates = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])

        # 釋放已完成的使用者（你原本的策略）
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

        # 路徑分配（已內建 blocked 記錄）
        path, reward, success, data_rate_records = run_greedy_path_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            W=W,
            access_matrix=access_matrix,
            path_loss=path_loss,
            sat_channel_dict=sat_load_dict,
            params=params
        )

        # 更新負載（僅對實際使用到的 (sat,ch) 做一次 +1）
        if path:
            unique_sat_ch = set((s, c) for s, c, _ in path)
            for sat, ch in unique_sat_ch:
                sat_load_dict[sat][ch] += 1
            # load_by_time 逐時槽增加
            for s, c, t in path:
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1

        all_user_data_rates.extend(data_rate_records)

        # 紀錄結果
        if success and path:
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

    # 注意：這裡 DataFrame 直接吃 dict，包含 blocked/reason 欄位
    df_data_rates = pd.DataFrame(all_user_data_rates)
    return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
