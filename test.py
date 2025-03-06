import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from dtw_jhbai import dtw_with_lb_keogh

def compute_envelope(seq, radius=1):
    upper_env = maximum_filter1d(seq, size=radius, mode='reflect')
    lower_env = minimum_filter1d(seq, size=radius, mode='reflect')
    return upper_env, lower_env

def main():
    serie = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    query = np.array([2.0, 3.0, 2.0])

    # Envelope 要從query計算（正確的做法）
    upper, lower = compute_envelope(query, radius=1)

    # 一次性呼叫，找出serie裡最好的子序列
    dist, start_idx, end_idx = dtw_with_lb_keogh(serie, query, upper, lower)

    print(f"DTW距離: {dist:.4f}, 子序列範圍: ({start_idx}, {end_idx})")

if __name__ == "__main__":
    main()
