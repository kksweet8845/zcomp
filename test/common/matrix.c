#include "matrix.h"
#include <string.h>


void mm(double* c, const double* a, const double * b, int M, int N, int K) {

    double* ap = a;
    double* bp, *cp;

    ap = a;
    for(int i=0;i<M;i++) {
        bp = b;
        for(int j=0;j<N;j++) {
            double val = *(ap++);
            cp = c + i * K;
            for(int k=0;k<K;k++) {
                *(cp++) += *(bp++) * val;
            }
        }
    }
    return;
}


void mmtr(double* in, int M, int N) {


    double* tmp = (double*) malloc(M * N * sizeof(double));

    register int i, j;
    register double*ptr = in;


    for(i=0;i<M;i++) {
        for(j=0;j<N;j++) {
            tmp[j*M+i] = *(ptr++);
        }
    }

    memcpy(in, tmp, M*N*sizeof(double));
    free(tmp);

}





