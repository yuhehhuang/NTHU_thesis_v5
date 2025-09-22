from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from src.utils import compute_sinr_and_rate

BLOCK_NO_PAIR      = "no_pair_available"
BLOCK_NO_FEASIBLE  = "no_feasible_W_or_conflict"
BLOCK_CAPACITY     = "capacity_shortage"

def mark_block(data_rate_records, uid, t, reason):
    data_rate_records.append({
        "user_id": uid, "time": t, "sat": None, "channel": None,
        "data_rate": 0.0, "blocked": True, "reason": reason
    })

def compute_sat_load(channel_status_dict):
    total = len(channel_status_dict)
    used  = sum(1 for v in channel_status_dict.values() if v != 0)
    return used / total if total > 0 else 0

def _vis_range(df_access, sat, t0, t1, time_slots):
    if t0 > t1: return False
    for tt in range(t0, min(t1, time_slots - 1) + 1):
        row = df_access[df_access["time_slot"] == tt]
        if row.empty: return False
        vis = row["visible_sats"].iloc[0]
        if isinstance(vis, str): vis = ast.literal_eval(vis)
        if sat not in vis: return False
    return True

def run_hungarian_per_W(df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W):
    time_slots  = len(df_access)
    alpha       = params["alpha"]

    # 0=可用, 1=本 cell 佔用, 2=背景保留不可分配（不干擾）
    sat_load_dict = {s: chs.copy() for s, chs in sat_channel_dict_backup.items()}

    # 使用者狀態
    user_assignments   = defaultdict(list)   # uid -> [(t, sat, ch)]
    user_last_pair     = {}                  # uid -> (sat, ch)
    user_last_ho_time  = {}                  # uid -> 上次換手時間 t
    user_next_time     = {}                  # uid -> 下一次可以再被分配的時間
    user_release_time  = {}                  # uid -> 本輪分配到的最後 t（用來釋放 1→0）

    data_rate_records = []
    load_by_time = defaultdict(lambda: defaultdict(int))

    for t in range(time_slots):
        # A) 釋放到期：t == release_t + 1 → 釋放該 user 本輪用到的 (sat,ch) 為 0
        to_free = [uid for uid, t_last in user_release_time.items() if t == t_last + 1]
        for uid in to_free:
            used_pairs = {(s, c) for (tt, s, c) in user_assignments[uid] if tt <= user_release_time[uid]}
            for sat, ch in used_pairs:
                if sat in sat_load_dict and ch in sat_load_dict[sat]:
                    sat_load_dict[sat][ch] = 0
            del user_release_time[uid]

        # B) 候選使用者：新進（t_start==t）或上次鎖窗結束（user_next_time==t）
        new_users    = df_users[df_users["t_start"] == t].index.tolist()
        resume_users = [uid for uid, tnext in user_next_time.items() if tnext == t]
        candidate_users = list({*new_users, *resume_users})
        if not candidate_users:
            continue

        # C) 這個 t 可見衛星
        row = df_access[df_access["time_slot"] == t]
        if row.empty:
            for uid in candidate_users:
                mark_block(data_rate_records, uid, t, BLOCK_NO_PAIR)
            continue
        vis_t = row["visible_sats"].iloc[0]
        if isinstance(vis_t, str): vis_t = ast.literal_eval(vis_t)

        # D) 全域候選 (sat,ch)：state==0 且「此刻 t 可見」
        global_pairs = []
        seen_pairs = set()
        for sat in vis_t:
            if sat not in sat_load_dict: continue
            for ch, state in sat_load_dict[sat].items():
                if state != 0: continue
                if (sat, ch) in seen_pairs: continue
                seen_pairs.add((sat, ch))
                global_pairs.append((sat, ch))
        if not global_pairs:
            for uid in candidate_users:
                mark_block(data_rate_records, uid, t, BLOCK_NO_PAIR)
            continue

        # E) pair 的 t 當下 score
        pair_score = {}
        for j, (sat, ch) in enumerate(global_pairs):
            _, rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_load_dict, ch)
            if rate is None: continue
            load_ratio = compute_sat_load(sat_load_dict[sat])
            pair_score[j] = (1 - alpha * load_ratio) * rate

        # F) cost matrix（關鍵修改：僅在「換手」時鎖 W；續用只鎖 1 slot）
        n_users = len(candidate_users)
        n_pairs = len(global_pairs)
        cost = np.full((n_users, n_pairs), 1e9)
        user_t_last = {}

        for i, uid in enumerate(candidate_users):
            t_end = int(df_users.loc[uid, "t_end"])
            last_ho = user_last_ho_time.get(uid, None)
            can_handover = (last_ho is None) or (t - last_ho >= W)

            if can_handover:
                # 換手：允許所有 global_pairs；t_last = min(t+W-1, t_end)
                t_last = min(t + W - 1, t_end)
                user_t_last[uid] = t_last
                for j, (sat, ch) in enumerate(global_pairs):
                    if j not in pair_score: continue
                    if not _vis_range(df_access, sat, t, t_last, time_slots): continue
                    cost[i, j] = -pair_score[j]
            else:
                # 不能換手：只能續用上一個 (sat,ch)；且只鎖 1 slot → t_last = t
                t_last = t
                user_t_last[uid] = t_last
                last_pair = user_last_pair.get(uid, None)
                if last_pair is None:
                    continue  # 整列 +∞ → 稍後標 no_feasible
                sat0, ch0 = last_pair
                try:
                    j = global_pairs.index((sat0, ch0))
                except ValueError:
                    continue  # 這個 pair 不是 state==0 → 不可行
                if (j in pair_score) and _vis_range(df_access, sat0, t, t_last, time_slots):
                    cost[i, j] = -pair_score[j]

        # G) 標整列不可行 → no_feasible
        row_min = cost.min(axis=1)
        infeasible = {candidate_users[i] for i, v in enumerate(row_min) if v > 1e8}
        for uid in infeasible:
            mark_block(data_rate_records, uid, t, BLOCK_NO_FEASIBLE)

        # H) 匈牙利配對 + 去重
        row_ind, col_ind = linear_sum_assignment(cost)
        claimed = set()
        matched = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] > 1e8: continue
            sat, ch = global_pairs[j]
            if (sat, ch) in claimed: continue
            claimed.add((sat, ch))
            matched.append((i, j))

        # I) 容量閘門（每衛星這個 t 可用 0-channel 上限）
        free_cap = {sat: sum(1 for v in sat_load_dict.get(sat, {}).values() if v == 0)
                    for sat in vis_t}
        by_sat = defaultdict(list)  # sat -> [(i,j,score)]
        for i, j in matched:
            sat, ch = global_pairs[j]
            by_sat[sat].append((i, j, pair_score.get(j, -1e9)))

        kept = []
        for sat, lst in by_sat.items():
            lst.sort(key=lambda x: x[2], reverse=True)
            k = max(0, free_cap.get(sat, 0))
            keep, drop = lst[:k], lst[k:]
            kept.extend((i, j) for (i, j, _) in keep)
            for (i, j, _) in drop:
                uid = candidate_users[i]
                if uid not in infeasible:
                    mark_block(data_rate_records, uid, t, BLOCK_CAPACITY)

        assigned_uids = {candidate_users[i] for (i, j) in kept}
        for i, uid in enumerate(candidate_users):
            if uid in infeasible or uid in assigned_uids: continue
            if row_min[i] <= 1e8:
                mark_block(data_rate_records, uid, t, BLOCK_CAPACITY)

        # J) 落地：鎖 [t..t_last]，僅「換手」更新 last_handover_t；續用不更新
        for i, j in kept:
            uid = candidate_users[i]
            sat, ch = global_pairs[j]
            t_last = user_t_last[uid]

            prev_pair = user_last_pair.get(uid, None)
            did_ho = (prev_pair is None) or (prev_pair != (sat, ch))
            if did_ho:
                user_last_ho_time[uid] = t  # 只在換手時更新
            # 鎖 channel
            sat_load_dict[sat][ch] = 1
            user_last_pair[uid] = (sat, ch)
            user_next_time[uid] = t_last + 1
            user_release_time[uid] = t_last

            for tt in range(t, t_last + 1):
                _, dr = compute_sinr_and_rate(params, path_loss, sat, tt, sat_load_dict, ch)
                if dr is None: dr = 0.0
                data_rate_records.append({
                    "user_id": uid, "time": tt, "sat": sat, "channel": ch,
                    "data_rate": dr, "blocked": False, "reason": None
                })
                load_by_time[tt][sat] += 1
                user_assignments[uid].append((tt, sat, ch))

    # === 輸出 ===
    df_results = pd.DataFrame(data_rate_records)

    paths = []
    for uid, entries in user_assignments.items():
        if not entries: continue
        entries = sorted(entries, key=lambda x: x[0])
        path_list = [(s, c, tt) for (tt, s, c) in entries]
        t_begin, t_end = entries[0][0], entries[-1][0]
        req_s = int(df_users.loc[uid, "t_start"]); req_e = int(df_users.loc[uid, "t_end"])
        assigned_times = {tt for (tt, _, _) in entries}
        success = all(tt in assigned_times for tt in range(req_s, req_e + 1))
        reward = sum(r["data_rate"] for r in data_rate_records if (r.get("user_id")==uid and not r.get("blocked", False)))
        paths.append([uid, str(path_list), t_begin, t_end, success, reward])

    df_paths = pd.DataFrame(paths, columns=["user_id","path","t_begin","t_end","success","reward"])
    return df_results, df_paths, load_by_time
