#ifndef CONV_H
#define CONV_H



#include <immintrin.h>

typedef struct conv_op {
    double *input;   double *d_input;
    double *output;  double *d_output;
    double *weights; double *d_weights;
    double *bias;    double *d_bias;
    double *input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    //* The number of value in one input, channel*input_w*input_h
    int in_units, out_units;

    short batchsize;
} conv_op;

void img2colm(const double* in, double* out, const conv_op* op);
void img2col_zcomp(const double* in, double* out, const conv_op *op, __mmask8* masks, int *masks_idx, int* masks_off);






#endif