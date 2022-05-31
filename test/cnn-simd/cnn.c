#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#include <sys/time.h>
#include <immintrin.h>
#include "conv.h"
#include "matrix.h"


#define I_CHANNEL 3
#define I_HEIGHT 3
#define I_WIDTH 3

#define O_CHANNEL 3
#define K_SIZE 3

#define STRIDE 1
#define PADDING 1



int main(int argc, char* argv[]) {

    conv_op op;
    op.in_channels = I_CHANNEL;
    op.out_channels = O_CHANNEL;
    op.kernel_size = K_SIZE;
    op.padding = PADDING;
    op.stride = STRIDE;
    op.in_w = I_WIDTH;
    op.in_h = I_HEIGHT;
    op.out_w = ((op.in_w + 2 * op.padding - op.kernel_size) / op.stride) + 1;
    op.out_h = ((op.in_h + 2 * op.padding - op.kernel_size) / op.stride) + 1;

    double *img_m = (double*) malloc(sizeof(double) * op.in_channels * op.in_h * op.in_w);
    for(int i=0;i<op.in_channels;i++) {
        for(int j=0;j<op.in_h;j++) {
            for(int k=0;k<op.in_w;k++) {
                int off = i* op.in_h * op.in_w + j * op.in_w + k;
                img_m[off] = 1;
            }
        }
    }
    double *wei_m = (double*) malloc(sizeof(double) * op.out_channels * op.in_channels * op.kernel_size * op.kernel_size);
    for(int i=0;i<op.out_channels * op.in_channels * op.kernel_size * op.kernel_size;i++) {
        wei_m[i] = 1;
    }

    double* out_m = malloc(sizeof(double) * op.out_w* op.out_h* op.in_channels* op.kernel_size * op.kernel_size);
    for(int i=0;i<op.out_w*op.out_h;i++) {
        for(int j=0;j<op.in_channels*op.kernel_size*op.kernel_size;j++) {
            int off = i * op.in_channels*op.kernel_size*op.kernel_size + j;
            out_m[off] = 0;
        }
    }

    double *res = (double*) calloc((op.out_h * op.out_w * op.out_channels), sizeof(double));

    img2colm(img_m, out_m, &op);
    //* perform matrix multiply
    mm(res, out_m, wei_m, op.out_w*op.out_h, op.in_channels * op.kernel_size * op.kernel_size, op.out_channels);
    //* matrix transpose
    mmtr(res, op.out_w*op.out_h, op.out_channels);

    for(int i=0;i<op.out_w*op.out_h*op.out_channels;i++) {
        printf("%5.2f ", res[i]);
        if((i+1) % (op.out_h*op.out_w) == 0)
            printf("\n\n");
    }
    printf("\n");

    return 0;

}





