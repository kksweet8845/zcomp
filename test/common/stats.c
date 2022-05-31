#include "stats.h"



float sparse_matrix_zero_ratio(double* m, int M, int N) {
    int total = M * N;
    int zeros = 0;

    for(int i=0;i<M * N;i++) {
        zeros += (*(m++) == 0);
    }

    return zeros/total;
}

