#include "mm_avx256.h"
#include "matrix.h"
#include "zcomp.h"


#include <stdio.h>
#include <immintrin.h>


static void vkprintf(__m256d a) {
    double tmp[4] __attribute__ ((aligned (32)));
    _mm256_store_pd(&tmp, *((__m256d *)&a));
    printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
}




int _mm_avx256m(double* o, double* m, double* n, int mh, int mw, int nh, int nw) {
    // if( !ALIGNED_32B(o) || !ALIGNED_32B(m) || !ALIGNED_32B(n))
    //     return 1;

    if( (mh & 0x3) && (mw & 0x3) && (nh & 0x3) && (nw & 0x3))
        return 2;

    
    if (mw != nh)
        return 3;


    mmtr(n, nh, nw);

    int tilew = 4;

    __m256d i0, i1, i2, i3;
    __m256d i4, i5, i6, i7;
    __m256d i8, i9, i10, i11;
    __m256d t0, t1, t2, t3;
    __m256d tr0, tr1, tr2, tr3;
    __m256d res;

    int noff, moff;

    for(int mrt=0;mrt<mh;mrt+=tilew) {
        for(int mct=0;mct<mw;mct+=tilew) {
            for(int nct=0; nct<nw;nct += tilew) {
                //* read col based
                i4 = _mm256_loadu_pd(( n + nct*nw + mct ));
                i5 = _mm256_loadu_pd(( n + (nct + 1)*nw + mct));
                i6 = _mm256_loadu_pd(( n + (nct + 2)*nw + mct));
                i7 = _mm256_loadu_pd(( n + (nct + 3)*nw + mct));


                for(int t=0;t<tilew;t++) {

                    res = _mm256_loadu_pd( ( o + (mrt+t)*nw + nct ) );
                    // res = _mm256_loadu_pd(&z[mrt+t][nct]);
                    i0 =  _mm256_loadu_pd( ( m + (mrt+t)*mw + mct) );
                    // i0 = _mm256_loadu_pd((&m[mrt+t][mct]));
                    
                    printf("mmidx %d, %d\n",  mrt/4+t, mct/4);
                    vkprintf(i0);

                    // printf("i\n");
                    // vkprintf(i0);
                    // printf("\n");
                    
                    i8 = _mm256_mul_pd(i4, i0);
                    i9 = _mm256_mul_pd(i5, i0);
                    i10 = _mm256_mul_pd(i6, i0);
                    i11 = _mm256_mul_pd(i7, i0);

                    

                    // printf("rt %d ct %d t %d\n", mrt, mct, t);
                    // vkprintf(i8);
                    // vkprintf(i9);
                    // vkprintf(i10);
                    // vkprintf(i11);
                    // printf("\n");

                    //* transpose

                    t0 = _mm256_unpacklo_pd(i8, i9);
                    t1 = _mm256_unpackhi_pd(i8, i9);
                    t2 = _mm256_unpacklo_pd(i10, i11);
                    t3 = _mm256_unpackhi_pd(i10, i11);

                    tr0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    tr1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    tr2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    tr3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    // _mm256_store_pd(&tmp, tr0);
                    // printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // _mm256_store_pd(&tmp, tr1);
                    // printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // _mm256_store_pd(&tmp, tr2);
                    // printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // _mm256_store_pd(&tmp, tr3);
                    // printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);

                    tr0 = _mm256_add_pd(tr0, tr1);
                    tr0 = _mm256_add_pd(tr0, tr2);
                    tr0 = _mm256_add_pd(tr0, tr3);
                    res = _mm256_add_pd(res, tr0);

                    // _mm256_store_pd(&z[mrt+t][nct], res);
                    _mm256_storeu_pd( ( o + (mrt+t)*nw + nct), res);
                }
            }
        }
    }
    return 0;
}

int _mm_avx256(double**z, double**m, double** n, int mh, int mw, int nh, int nw) {


    if( !ALIGNED_32B(z) || !ALIGNED_32B(m) || !ALIGNED_32B(n))
        return 1;

    if( (mh & 0x3) && (mw & 0x3) && (nh & 0x3) && (nw & 0x3))
        return 2;

    if (mw != nh)
        return 3;


    for(int i=0;i<nh;i++) {
        for(int j=i;j<nw;j++) {
            double t;
            if(i != j){
                t = n[j][i];
                n[j][i] = n[i][j];
                n[i][j] = t;
            }   
        }
    }


    int tilew = 4;

    __m256d i0, i1, i2, i3;
    __m256d i4, i5, i6, i7;
    __m256d i8, i9, i10, i11;
    __m256d t0, t1, t2, t3;
    __m256d tr0, tr1, tr2, tr3;
    __m256d res;


    for(int mrt=0;mrt<mh;mrt+=tilew) {
        for(int mct=0;mct<mw;mct+=tilew) {
            for(int nct=0; nct<nw;nct += tilew) {
                //* read col based
                i4 = _mm256_loadu_pd((&n[nct][mct]));
                i5 = _mm256_loadu_pd((&n[nct+1][mct]));
                i6 = _mm256_loadu_pd((&n[nct+2][mct]));
                i7 = _mm256_loadu_pd((&n[nct+3][mct]));


                for(int t=0;t<tilew;t++) {
                    res = _mm256_loadu_pd(&z[mrt+t][nct]);
                    i0 = _mm256_loadu_pd((&m[mrt+t][mct]));

                    printf("mmidx %d, %d\n",  mrt/4+t, mct/4);
                    vkprintf(i0);

                    // printf("i\n");
                    // vkprintf(i0);
                    // printf("\n");
                    
                    i8 = _mm256_mul_pd(i4, i0);
                    i9 = _mm256_mul_pd(i5, i0);
                    i10 = _mm256_mul_pd(i6, i0);
                    i11 = _mm256_mul_pd(i7, i0);

                    //* transpose

                    t0 = _mm256_unpacklo_pd(i8, i9);
                    t1 = _mm256_unpackhi_pd(i8, i9);
                    t2 = _mm256_unpacklo_pd(i10, i11);
                    t3 = _mm256_unpackhi_pd(i10, i11);

                    tr0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    tr1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    tr2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    tr3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    tr0 = _mm256_add_pd(tr0, tr1);
                    tr0 = _mm256_add_pd(tr0, tr2);
                    tr0 = _mm256_add_pd(tr0, tr3);
                    res = _mm256_add_pd(res, tr0);

                    _mm256_store_pd(&z[mrt+t][nct], res);
                }
            }
        }
    }


    return 0;
}



int _mm_avx256m_zcomp(double* o, double* m, double* n, int mh, int mw, int nh, int nw,
                      __mmask8* mmasks, int mmasks_num, int* mmasks_off) {


     if( (mh & 0x3) && (mw & 0x3) && (nh & 0x3) && (nw & 0x3))
        return 2;

    
    if (mw != nh)
        return 3;


    mmtr(n, nh, nw);

    int tilew = 4;

    __m256d i0, i1, i2, i3;
    __m256d i4, i5, i6, i7;
    __m256d i8, i9, i10, i11;
    __m256d t0, t1, t2, t3;
    __m256d tr0, tr1, tr2, tr3;
    __m256d res;

    int noff, moff;

    int mmask_w = mw >> 2;
    int mmask_h = mh >> 2;

    double* tmp;

    for(int mrt=0;mrt<mh;mrt+=tilew) {
        for(int mct=0;mct<mw;mct+=tilew) {
            for(int nct=0; nct<nw;nct += tilew) {
                //* read col based
                i4 = _mm256_loadu_pd(( n + nct*nw + mct ));
                i5 = _mm256_loadu_pd(( n + (nct + 1)*nw + mct));
                i6 = _mm256_loadu_pd(( n + (nct + 2)*nw + mct));
                i7 = _mm256_loadu_pd(( n + (nct + 3)*nw + mct));


                for(int t=0;t<tilew;t++) {
                    res = _mm256_loadu_pd( ( o + (mrt+t)*nw + nct ) );
                    int mmasks_off_off = (mrt+t) * mmask_w + (mct/4);
                    tmp = (void*)m + (mmasks_off[mmasks_off_off]);
                    i0 = _mm256_zcomp_loadu_pd(&tmp, mmasks[mmasks_off_off]);
                    
                    i8 = _mm256_mul_pd(i4, i0);
                    i9 = _mm256_mul_pd(i5, i0);
                    i10 = _mm256_mul_pd(i6, i0);
                    i11 = _mm256_mul_pd(i7, i0);

                    //* transpose

                    t0 = _mm256_unpacklo_pd(i8, i9);
                    t1 = _mm256_unpackhi_pd(i8, i9);
                    t2 = _mm256_unpacklo_pd(i10, i11);
                    t3 = _mm256_unpackhi_pd(i10, i11);

                    tr0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    tr1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    tr2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    tr3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    tr0 = _mm256_add_pd(tr0, tr1);
                    tr0 = _mm256_add_pd(tr0, tr2);
                    tr0 = _mm256_add_pd(tr0, tr3);
                    res = _mm256_add_pd(res, tr0);

                    _mm256_storeu_pd( ( o + (mrt+t)*nw + nct), res);
                }
            }
        }
    }
    return 0;
}


