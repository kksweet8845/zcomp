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

#include "sim_api.h"

#define DIM 100


main(int argc, char* argv[]) {


    SimSetThreadName("main");



    double m[DIM][DIM];
    double n[DIM][DIM];
    double z[DIM][DIM];

    //* initialize m,n

    for(int i=0;i<DIM;i++) {
        for(int j=0;j<DIM;j++) {
            m[i][j] = ((double)i*DIM+j);
            n[i][j] = ((double)i*DIM+j);
            z[i][j] = 0;
        }
    }

    //* perform matrix multiplication

    SimRoiStart();
    SimNamedMarker(4, "begin");

    for(int i=0;i<DIM;i++) {
        for(int j=0;j<DIM;j++) {
            double tmp = 0.0;
            for(int k=0;k<DIM;k++) {
                tmp += m[i][k] * n[k][j];
            }
            z[i][j] = tmp;
        }
    }

    SimNamedMarker(5, "end");
    SimRoiEnd();


    FILE* fp = fopen("ans.txt", "w");

    for(int i=0;i<DIM;i++) {
        for(int j=0;j<DIM;j++) {
            fprintf(fp, "%lf ", z[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);





    exit(0);

}


