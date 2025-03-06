#ifndef DTW_CORE_H
#define DTW_CORE_H

double subsequence_dtw(const double *serie, int len_serie, const double *query, int len_query, int *start_idx, int *end_idx);
double lb_keogh(const double *query, const double *upper, const double *lower, int len);

#endif
