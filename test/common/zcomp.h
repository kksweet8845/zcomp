#ifndef ZCOMP_H
#define ZCOMP_H



#include <immintrin.h>


// zcomp_header_t* zcomp_store_pd(double* m, int memb);
// int zcomp_load_pd(double* m, zcomp_header_t* header);
void _mm256_zcomp_storeu_pd(double* dst,  double* src, int len, __mmask8* mask_header, int *mask_idx_ptr);
__m256d _mm256_zcomp_loadu_pd(double** src, __mmask8 mask);




#endif