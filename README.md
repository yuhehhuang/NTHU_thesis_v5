# NTHU_thesis
sat_load_dict用來當graph上user看到的load狀態(演算法用):sat_load_dict[sat][ch]
load_by_time :紀錄[t][s]時load情況(計算結果用)

匈牙利:你希望將每個時間 
t 進場的 user 當作一個 batch，並且「完整安排這批使用者直到各自的t_end為止」，而中途不插入其他新進來的 user。這個邏輯類似 block scheduling：每批用戶進來就完整處理完再換下一批。

##
JP只要DP/GENE/MSLB/GREEDY/matching

>> git add src/*.pyom/yuhehhuang/NTHU_thesis.git
>> git commit -m "greedy is available"
>> git push origin mains_final>