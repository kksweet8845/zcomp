#include "zcomp.h"

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>



static inline void vkprintf(__m256d a) {
    double tmp[4] __attribute__ ((aligned (32)));
    _mm256_store_pd(tmp, *((__m256d *)&a));
    printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
}



int main(void) {

    double sm[8] = {0, 1, 0, 2, 0, 0, 4, 5};

    double* comp_sm = (double*) calloc(sizeof(double), 8);


    __mmask8* masks = (__mmask8*) malloc(sizeof(__mmask8) * (8/4));

    int mask_idx;

    _mm256_zcomp_storeu_pd(comp_sm, sm, 8, masks, &mask_idx);


    double* cur = comp_sm;
    for(int i = 0; i < mask_idx; i++) {
        printf("%2x ", masks[i]);
        printf("\n");
        for(int j = 0; j < (masks[i]&0xf0)>>4; j++)
            printf("%5.2lf ", *cur++);
        printf("\n");
    }


    __m256d t;
    cur = comp_sm;
    for(int i=0;i<mask_idx;i++) {
        t = _mm256_zcomp_loadu_pd(&cur, masks[i]);
        vkprintf(t);
    }




    return 0;

}










