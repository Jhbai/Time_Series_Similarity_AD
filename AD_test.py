import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from dtw_jhbai import dtw_with_lb_keogh

# 計算 envelope for LB_Keogh
def compute_envelope(seq, radius=1):
    upper_env = maximum_filter1d(seq, size=radius, mode='reflect')
    lower_env = minimum_filter1d(seq, size=radius, mode='reflect')
    return upper_env, lower_env

class DTWSelfAnomalyDetector:
    def __init__(self, series, window_size, radius=1):
        self.series = series
        self.w = window_size
        self.radius = radius
        self.distances = []
        self.anomaly_indices = []

    def compute_self_similarity(self):
        n = len(self.series)
        distances = np.zeros(n - self.w + 1)

        for i in range(n - self.w + 1):
            query = self.series[i:i+self.w]
            upper, lower = compute_envelope(query, radius=self.radius)
            best_dist = np.inf

            # 和其他子序列比對
            for j in range(n - self.w + 1):
                if abs(i - j) < self.w:  # 避免與自己或太近的重疊區域比對
                    continue

                candidate = self.series[j:j+self.w]
                lb_dist, _, _ = dtw_with_lb_keogh(candidate, query, upper, lower)
                if lb_dist < best_dist:
                    best_dist = lb_dist

            distances[i] = best_dist

        self.distances = distances

    def detect_anomalies(self, percentile=99):
        threshold = np.percentile(self.distances, percentile)
        anomalies = []

        for idx, dist in enumerate(self.distances):
            if dist >= threshold:
                anomalies.append((idx, idx + self.w - 1))

        self.anomaly_indices = anomalies
        return anomalies

    def run_detection(self, percentile=99):
        print("開始計算自我相似性...")
        self.compute_self_similarity()
        print("計算完成，正在偵測異常...")
        anomalies = self.detect_anomalies(percentile)
        return anomalies

# 範例使用方法：
if __name__ == "__main__":
    # 假設你從外部載入單條序列資料
    # series = np.load("single_series.npy")  
    series = np.sin(np.linspace(0, 10, 2000))
    series[200:400] += 2

    detector = DTWSelfAnomalyDetector(series, window_size=50, radius=1)
    anomalies = detector.run_detection(percentile=99)

    print("找到的異常位置區段：")
    for st, ed in anomalies:
        print(f"異常區段：{st} ~ {ed}")
