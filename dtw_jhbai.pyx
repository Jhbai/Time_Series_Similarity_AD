# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cdef extern from "dtw_core.h":
    double subsequence_dtw(const double *serie, int len_serie, const double *query, int len_query, int *start_idx, int *end_idx)
    double lb_keogh(const double *query, const double *upper, const double *lower, int len)

def dtw_with_lb_keogh(cnp.ndarray[double, ndim=1] serie,
                      cnp.ndarray[double, ndim=1] query,
                      cnp.ndarray[double, ndim=1] upper,
                      cnp.ndarray[double, ndim=1] lower):

    cdef int window_size = query.shape[0]
    cdef int len_serie = serie.shape[0]
    cdef double best_dist = np.inf
    cdef int best_start = -1, best_end = -1
    cdef int start_idx, end_idx
    cdef double lb_dist, dtw_dist
    cdef const double [:] serie_mv = serie  # 使用 memoryview 存取

    for i in range(len_serie - window_size + 1):
        # 使用 memoryview 而非 numpy 物件取址
        lb_dist = lb_keogh(&serie_mv[i], &upper[0], &lower[0], window_size)
        if lb_dist >= best_dist:
            continue  # 超過目前最佳，跳過完整DTW計算

        dtw_dist = subsequence_dtw(&serie_mv[i], window_size, &query[0], window_size, &start_idx, &end_idx)
        if dtw_dist < best_dist:
            best_dist = dtw_dist
            best_start = i + start_idx
            best_end = i + end_idx

    return best_dist, best_start, best_end
