#include "zcomp.h"
#include <string.h>

#include <immintrin.h>
#include <popcntintrin.h>
#include <stdio.h>


static inline void vkprintf(__m256d a) {
    double tmp[4] __attribute__ ((aligned (32)));
    _mm256_store_pd(&tmp, *((__m256d *)&a));
    printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
}





void _mm256_zcomp_storeu_pd(double* dst,  double* src, int len, __mmask8* mask_header, int *mask_idx_ptr) {

    double* src_cur = src;
    double* dst_cur = dst;

    __m256d zvec = _mm256_set_pd(0, 0, 0, 0);

    int remain = len&~0x3;

    int mask_idx = 0;
    for(int i=0;i<len/4;i++) {
        __m256d tvec = _mm256_loadu_pd(src_cur);
        vkprintf(tvec);
        src_cur += 4;
        __mmask8 mask = _mm256_cmp_pd_mask(tvec, zvec, _CMP_NEQ_US);
        printf("mask %02x\n", mask);
        unsigned int nnz_cnt = _mm_popcnt_u32(mask);
        mask_header[mask_idx++] = mask | ((nnz_cnt & 0x0f) << 4);
        _mm256_mask_compressstoreu_pd((void*)dst_cur, mask, tvec);
        for(int j=0;j<nnz_cnt;j++)
            printf("dst %lf \n", *(dst_cur+j));
        // printf("dst")
        dst_cur += nnz_cnt;
    }

    if(len & 0x3) {
        unsigned char nnz_cnt = 0;
        __mmask8 mask = 0x0;
        for(int i= 0;i<4;i++ ) {
            if(*src_cur != 0.0f && i < remain) {
                nnz_cnt++;
                mask |= (0x1 << (4-i));
                *dst_cur++ = *src_cur++;
            } else {
                *dst_cur++ = 0.0f;
            }
        }
        mask_header[mask_idx++] = mask | (nnz_cnt << 4) ;
    }
    *mask_idx_ptr = mask_idx;
}

__m256d _mm256_zcomp_loadu_pd(double** src, __mmask8 mask) {
    unsigned char nnz_cnt = (mask & 0xf0) >> 4;
    __m256d tvec = _mm256_maskz_expandloadu_pd(mask&0x0f, (void*) *src);
    *src += nnz_cnt;
    return tvec;
}