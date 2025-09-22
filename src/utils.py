import numpy as np
from collections import defaultdict
import pandas as pd
from collections import defaultdict
import copy

def update_m_s_t_from_channels(sat_channel_dict, all_sats):
    """每個 channel 只要非 0 就算佔用 1 個（避免 2 被算成 2、也避免 >1 重複累加）。"""
    return {
        sat: sum(1 for v in sat_channel_dict.get(sat, {}).values() if v != 0)
        for sat in all_sats
    }


def check_visibility(df_access, sat, t_start, t_end):
    """檢查衛星在 t_start~t_end 是否連續可見"""
    for t in range(t_start, t_end + 1):
        row = df_access[df_access["time_slot"] == t]
        if row.empty or sat not in row["visible_sats"].values[0]:
            return False
    return True

def compute_sinr_and_rate(params, path_loss, sat, t, sat_channel_dict, chosen_channel):
    """
    只把 channel 狀態為 1 的當作會造成 interference。
    sat_channel_dict 的格式支援 0/1/2 如上。
    """
    PL = path_loss.get((sat, t))
    if PL is None:
        return None, None

    # 注意：eirp_linear, grx_linear, noise_watt 應該在 params 已經存在
    P_rx = params["eirp_linear"] * params["grx_linear"] / PL

    interference = 0.0
    for other_sat, channels in sat_channel_dict.items():
        if other_sat == sat:
            continue
        # 只有狀態 == 1 的 channel 會被算入干擾
        if channels.get(chosen_channel, 0) == 1:
            PL_other = path_loss.get((other_sat, t))
            if PL_other:
                interference += params["eirp_linear"] * params["grx_linear"] / PL_other

    SINR = P_rx / (params["noise_watt"] + interference)
    data_rate = params["channel_bandwidth_hz"] * np.log2(1 + SINR) / 1e6  # Mbps
    return SINR, data_rate
    
def compute_sat_load(sat_channel_dict):
    # 回傳 dict[sat] = 已佔用 channel 數（0/1/2 都視為佔用）
    load = {}
    for sat, chs in sat_channel_dict.items():
        load[sat] = sum(1 for v in chs.values() if v != 0)
    return load
def compute_score(params, m_s_t, data_rate, sat):
    L = m_s_t[sat] / params["num_channels"]
    return (1 - params["alpha"] * L) * data_rate
#這是每個user都分配完channel後正式去算他們彼此的干擾
#sat_channel_dict_backup是有0,1,2的狀態
def recompute_all_data_rates(all_user_paths, path_loss, params, sat_channel_dict_backup):
    """
    重新計算每個 user 的 data rate，考慮背景干擾 + 所有 user 的分配。
    """
    # 1️⃣ 先複製初始化的背景使用者狀態
    sat_channel_dict = copy.deepcopy(sat_channel_dict_backup)

    # 2️⃣ 建立一個 dict 記錄每個 time slot 的使用者分配
    assignments_by_time = defaultdict(list)  # {time: [(user_id, sat, ch), ...]}
    for entry in all_user_paths:
        user_id = entry["user_id"]
        path = entry["path"]
        if isinstance(path, str):
            try:
                path = eval(path)
            except:
                continue  # 如果 eval 出錯就跳過       
        for sat, ch, t in path:
            assignments_by_time[t].append((user_id, sat, ch))

    all_records = []

    # 3️⃣ 逐個 time slot 重新計算干擾 + data rate
    for t, assignments in assignments_by_time.items():
        # 先把這個 time slot 的使用者佔用標記到 sat_channel_dict
        temp_dict = copy.deepcopy(sat_channel_dict)
        for _, sat, ch in assignments:
            temp_dict[sat][ch] = 1  # 表示這個 slot 這個 channel 有 user 使用

        # 計算每個 user 的 SINR & data rate
        for user_id, sat, ch in assignments:
            SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t, temp_dict, ch)
            all_records.append({
                "user_id": user_id,
                "time": t,
                "sat": sat,
                "channel": ch,
                "data_rate": data_rate if data_rate else 0
            })

    return pd.DataFrame(all_records)


def compute_blocking_stats(df_users: pd.DataFrame, df_results: pd.DataFrame):
    """
    回傳:
      df_blk: 每位使用者的 blocking 統計
      overall_blocking_rate: 以「人」為單位的阻斷率
         = 有至少一次被擋的使用者數 / 服務長度 > 0 的使用者總數
      reason_breakdown: （若 df_results 有 blocked/reason 欄位）各原因分布
    """
    if "user_id" not in df_results or "time" not in df_results:
        raise ValueError("df_results 必須包含 'user_id' 與 'time' 欄位")

    df_results = df_results.copy()
    df_results["user_id"] = df_results["user_id"].astype(int)
    df_results["time"] = df_results["time"].astype(int)

    has_blocked = "blocked" in df_results.columns
    if has_blocked:
        assigned_series = (
            df_results[df_results["blocked"] == False]
            .groupby("user_id")["time"]
            .agg(lambda s: set(s.astype(int)))
        )
    else:
        assigned_series = (
            df_results.groupby("user_id")["time"]
            .agg(lambda s: set(s.astype(int)))
        )

    rows = []
    total_len = 0
    total_blocked = 0  # 仍保留（若你還想要 slot-based 指標可用）

    df_users_local = df_users.copy()
    df_users_local["t_start"] = df_users_local["t_start"].astype(int)
    df_users_local["t_end"] = df_users_local["t_end"].astype(int)

    for uid, urow in df_users_local[["t_start", "t_end"]].iterrows():
        t0, t1 = int(urow["t_start"]), int(urow["t_end"])
        service_set = set(range(t0, t1 + 1))
        assigned_set = assigned_series.get(uid, set())
        assigned_in_window = len(assigned_set & service_set)
        blocked_slots = len(service_set) - assigned_in_window

        rows.append({
            "user_id": uid,
            "service_len": len(service_set),
            "assigned_slots": assigned_in_window,
            "blocked_slots": blocked_slots,
            "user_blocking_rate": (blocked_slots / len(service_set)) if len(service_set) > 0 else 0.0
        })

        total_len += len(service_set)
        total_blocked += blocked_slots

    df_blk = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)

    # NEW: 以「人」為單位的阻斷判定與總體比例
    df_blk["is_blocked_any"] = (df_blk["service_len"] > 0) & (df_blk["blocked_slots"] > 0)  # 至少一次中斷
    eligible_users = int((df_blk["service_len"] > 0).sum())  # 有實際服務區間的人
    overall_blocking_rate = (int(df_blk["is_blocked_any"].sum()) / eligible_users) if eligible_users > 0 else 0.0

    reason_breakdown = None
    if has_blocked and "reason" in df_results.columns:
        df_b = df_results[df_results["blocked"] == True]
        reason_counts = df_b.groupby("reason").size().rename("count")
        if reason_counts.sum() > 0:
            reason_pct = (reason_counts / reason_counts.sum()).rename("pct")
            reason_breakdown = pd.concat([reason_counts, reason_pct], axis=1).reset_index()
        else:
            reason_breakdown = pd.DataFrame(columns=["reason", "count", "pct"])

    return df_blk, overall_blocking_rate, reason_breakdown



def compute_timewise_blocking_rate(df_users: pd.DataFrame, df_results: pd.DataFrame):
    """
    回傳每個 time 的 blocking rate：
      time_block_rate[t] = 被擋的 user 數 / 該 t 需要服務的 user 數
    需要 df_results 具備 blocked 欄位；若沒有，請先用 compute_blocking_stats 的邏輯自行補 blocked。
    """
    if "blocked" not in df_results.columns:
        raise ValueError("df_results 缺少 'blocked' 欄位，無法做時間向度的阻擋率。")

    # 計算每個 t 的 demand（多少 user 應該在服務）
    # 方式：對 df_users 展開或用統計方式計算
    t_min = int(df_users["t_start"].min())
    t_max = int(df_users["t_end"].max())
    demand = {}
    for t in range(t_min, t_max + 1):
        demand[t] = int(((df_users["t_start"] <= t) & (df_users["t_end"] >= t)).sum())

    # 每個 t 被擋的事件數
    blocked_per_t = (
        df_results[df_results["blocked"] == True]
        .groupby("time").size().to_dict()
    )

    rows = []
    for t in range(t_min, t_max + 1):
        d = demand.get(t, 0)
        b = blocked_per_t.get(t, 0)
        rate = (b / d) if d > 0 else 0.0
        rows.append({"time": t, "demand_users": d, "blocked_users": b, "time_blocking_rate": rate})

    return pd.DataFrame(rows)