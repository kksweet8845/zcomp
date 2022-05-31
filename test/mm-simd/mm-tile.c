#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PAGE_SIZE               4096
#define NUM_CACHE_LINES        65536
#define LOG2_LINE_SIZE             4
#define PI                         3.1416
#define DEFAULT_M                 10
#define DEFAULT_P                  1


#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>

// #include "sim_api.h"

#define DIM 100


void vkprintf(__m256d a) {
    double tmp[4] __attribute__ ((aligned (32)));
    _mm256_store_pd(&tmp, *((__m256d *)&a));
    printf("%lf %lf %lf %lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
}



main(int argc, char* argv[]) {




    double m[DIM][DIM] __attribute__ ((aligned (32)));
    double n[DIM][DIM] __attribute__ ((aligned (32)));
    double z[DIM][DIM] __attribute__ ((aligned (32)));
    
    __m256d v1_d;
    __m256d v2_d;
    __m256d v3_d;
    __m256d sumd;
    __m256d i0, i1, i2, i3;
    __m256d i4, i5, i6, i7;
    __m256d i8, i9, i10, i11;
    __m256d t0, t1, t2, t3;
    __m256d tr0, tr1, tr2, tr3;
    __m256d res;

    unsigned int clocktime1, clocktime2;

    struct timeval fulltime;




    int packed = 4;
    int tilew = 4;

    //* initialize m,n

    for(int i=0;i<DIM;i++) {
        for(int j=0;j<DIM;j++) {
            m[i][j] = ((double)i*DIM+j);
            n[i][j] = ((double)i*DIM+j);
            z[i][j] = 0;
        }
    }

    for(int i=0;i<DIM;i++) {
        for(int j=i;j<DIM;j++) {
            double t;
            if(i != j){
                t = n[j][i];
                n[j][i] = n[i][j];
                n[i][j] = t;
            }   
        }
    }


    //* Perform the tile-based matrix multiplication
    //* P = M * N
    //* M = size[I*K]
    //* N = size[K*J]
    

    int ITER = DIM / packed;


    //* perform matrix multiplication


    double* m_addr;
    double* n_addr;
    double* z_addr;

    double tmp[4] __attribute__ ((aligned (32)));

    int bin = DIM / tilew;


    gettimeofday(&fulltime, NULL);
    (clocktime1) = (unsigned long)(fulltime.tv_usec + fulltime.tv_sec * 1000000);

    for(int mrt=0;mrt<DIM;mrt+=tilew) {
        for(int mct=0;mct<DIM;mct+=tilew) {
            for(int nct=0;nct<DIM;nct+=tilew) {

                //* read col based
                i4 = _mm256_loadu_pd((&n[nct][mct]));
                i5 = _mm256_loadu_pd((&n[nct+1][mct]));
                i6 = _mm256_loadu_pd((&n[nct+2][mct]));
                i7 = _mm256_loadu_pd((&n[nct+3][mct]));

                for(int t=0;t<tilew;t++) {
                    res = _mm256_loadu_pd(&z[mrt+t][nct]);
                    i0 = _mm256_loadu_pd((&m[mrt+t][mct]));

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

                    _mm256_store_pd(&z[mrt+t][nct], res);
                }
            }
        }
    }


    gettimeofday(&fulltime, NULL);
    (clocktime2) = (unsigned long)(fulltime.tv_usec + fulltime.tv_sec * 1000000);

    // SimNamedMarker(5, "end");
    // SimRoiEnd();

    for(int i=0;i<DIM;i++) {
        for(int j=0;j<DIM;j++) {
            printf("%10.2lf ", z[i][j]);
        }
        printf("\n");
    }


    printf("total ns: %ld\n", clocktime2 - clocktime1);


    exit(0);

}


