import pandas as pd
import random
from geopy.distance import distance
from geopy.point import Point

T = 10 * 60 // 15  # 10 min (40 slots)
NUM_USERS = 100
NUM_GROUPS = 20
USERS_PER_GROUP = NUM_USERS // NUM_GROUPS
CENTER_LAT = 40.0386
CENTER_LON = -75.5966

center = Point(CENTER_LAT, CENTER_LON)
users = []

# 產生所有可能的開始時間
possible_starts = list(range(0, max(0, T - 12) + 1))

# 隨機抽取不重複的開始時間
group_starts = random.sample(possible_starts, NUM_GROUPS)

for group_id in range(NUM_GROUPS):
    group_t_start = group_starts[group_id]  # ✅ 取對應的開始時間

    for i in range(USERS_PER_GROUP):
        user_id = group_id * USERS_PER_GROUP + i

        # 每個 user 自己決定服務時長 10~12 slot
        duration = random.randint(10, 12)
        t_end = min(group_t_start + duration, T - 1)  # 確保不超過 T

        # 隨機位置（半徑 0.8 km）
        angle = random.uniform(0, 360)
        radius = random.uniform(0, 0.8)
        point = distance(kilometers=radius).destination(center, bearing=angle)

        users.append({
            "user_id": user_id,
            "t_start": group_t_start,  # ✅ 同一個 group 的開始時間一樣
            "t_end": t_end,            # ✅ 每個 user 的結束時間不同
            "lat": point.latitude,
            "lon": point.longitude
        })

df = pd.DataFrame(users)
df.to_csv("data/user_info.csv", index=False)
print("✅ 產生 user_info.csv")
