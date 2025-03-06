import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from dtw_jhbai import dtw_with_lb_keogh

def paa(series, segments):
    """PAA 降維，確保不會產生 NaN，並保留對應的時間軸"""
    n = len(series)
    segment_size = n // segments
    remainder = n % segments

    # 若不能整除，則用 np.pad() 進行補齊
    if remainder > 0:
        pad_size = segments - remainder
        series = np.pad(series, (0, pad_size), mode='edge')
        n = len(series)

    reduced_series = np.array([np.mean(series[i:i+segment_size]) for i in range(0, n, segment_size)])
    time_axis = np.array([i + segment_size // 2 for i in range(0, n, segment_size)])

    return reduced_series, time_axis

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

            for j in range(n - self.w + 1):
                if abs(i - j) < self.w:
                    continue

                candidate = self.series[j:j+self.w]
                lb_dist, _, _ = dtw_with_lb_keogh(candidate, query, upper, lower)

                if np.isnan(lb_dist) or np.isinf(lb_dist):
                    continue  # 防止 NaN 影響運算

                if lb_dist < best_dist:
                    best_dist = lb_dist

            distances[i] = best_dist

        self.distances = distances

    def detect_anomalies(self, percentile=99):
        if len(self.distances) == 0 or np.isnan(self.distances).any():
            print("⚠️ Warning: DTW 計算失敗，沒有有效數據！")
            return []

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

if __name__ == "__main__":
    series = np.sin(np.linspace(0, 10, 2000))
    series[200:400] += 2

    # 先執行 PAA 降維
    reduced_series, time_axis = paa(series, segments=10)

    # 進行異常檢測
    detector = DTWSelfAnomalyDetector(reduced_series, window_size=5, radius=1)
    anomalies = detector.run_detection(percentile=99)

    # 映射回原始時間軸
    mapped_anomalies = [(time_axis[st], time_axis[ed]) for st, ed in anomalies]

    print("找到的異常位置區段（對應回原始時間）：")
    for st, ed in mapped_anomalies:
        print(f"異常區段：{st} ~ {ed}")
