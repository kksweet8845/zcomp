#ifndef MM_AVX256_H
#define MM_AVX256_h


#include <immintrin.h>

#define ALIGNED_32B(addr) (!((unsigned long)addr & 0x1f))

int _mm_avx256(double** o, double** m, double** n, int mh, int mw, int nh, int nw);
int _mm_avx256m(double* o, double* m, double* n, int mh, int mw, int nh, int nw);
int _mm256_zcomp_store_ps(double* m, int len);

int _mm_avx256m_zcomp(double* o, double* m, double* n, int mh, int mw, int nh, int nw,
                      __mmask8* mmasks, int mmasks_num, int* mmasks_off);





#endif