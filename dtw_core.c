#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "dtw_core.h"

double min3(double a, double b, double c) {
    double min = a < b ? a : b;
    return min < c ? min : c;
}

double subsequence_dtw(const double *serie, int len_serie, const double *query, int len_query, int *start_idx, int *end_idx) {
    double **dtw = malloc((len_query + 1) * sizeof(double*));
    for (int i = 0; i <= len_query; i++)
        dtw[i] = malloc((len_serie + 1) * sizeof(double));

    for (int i = 0; i <= len_query; i++)
        for (int j = 0; j <= len_serie; j++)
            dtw[i][j] = DBL_MAX;
    dtw[0][0] = 0;

    for (int i = 1; i <= len_query; i++)
        for (int j = 1; j <= len_serie; j++) {
            double cost = fabs(query[i-1] - serie[j-1]);
            dtw[i][j] = cost + min3(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]);
        }

    double min_dist = DBL_MAX;
    int min_j = -1;
    for (int j = 1; j <= len_serie; j++) {
        if (dtw[len_query][j] < min_dist) {
            min_dist = dtw[len_query][j];
            min_j = j;
        }
    }

    int i = len_query;
    int j = min_j;
    while (i > 1 && j > 1) {
        if (dtw[i-1][j-1] <= dtw[i-1][j] && dtw[i-1][j-1] <= dtw[i][j-1]) {
            i--; j--;
        } else if (dtw[i-1][j] < dtw[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }

    *start_idx = j - 1;
    *end_idx = min_j - 1;

    for (int k = 0; k <= len_query; k++)
        free(dtw[k]);
    free(dtw);

    return min_dist;
}

double lb_keogh(const double *query, const double *upper, const double *lower, int len) {
    double lb_sum = 0;
    for (int i = 0; i < len; i++) {
        if (query[i] > upper[i])
            lb_sum += (query[i] - upper[i]) * (query[i] - upper[i]);
        else if (query[i] < lower[i])
            lb_sum += (query[i] - lower[i]) * (query[i] - lower[i]);
    }
    return sqrt(lb_sum);
}
