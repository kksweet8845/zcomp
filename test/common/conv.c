#include "conv.h"



void img2colm(const double* in, double* out, const conv_op *op) {

    int kk = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;

    for(int oh=0;oh<op->out_h;oh+=op->stride) {
        for(int ow=0;ow<op->out_w;ow+=op->stride) {
            int owoh_off = oh*op->out_w + ow;
            for(int ic=0;ic<op->in_channels;ic++) {
                int coff = kk * ic;
                for(int kh=0;kh<op->kernel_size;kh++) {
                    for(int kw=0;kw<op->kernel_size;kw++) {
                        int koff = coff + kh * op->kernel_size + kw;
                        int outoff = owoh_off * ikk + koff;
                        int inoff = ic * (op->in_h) * (op->in_w) + (oh+kh-op->padding) * op->in_h + (ow+kw-op->padding);
                        //* padding
                        if( (oh+kh - op->padding) < 0 ||
                            (ow+kw - op->padding) < 0) {
                                out[outoff] = 0;
                                continue;
                        }

                        out[outoff] = in[inoff];
                    }
                }
            }
        }
    }
}

void img2col_zcomp(const double* in, double* out, const conv_op *op, __mmask8* masks, int *masks_idx, int* masks_off) {

    int kk = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;

    int mask_w = ikk / 4;
    int mask_h = op->out_h*op->out_w / 4;
    double tmp[4];
    int tmp_idx = 0;
    int _mask_idx = 0;
    int cum_cnt = 0;

    double* out_cur = out;

    __m256d zvec = _mm256_set_pd(0, 0, 0, 0);

    for(int oh=0;oh<op->out_h;oh+=op->stride) {
        for(int ow=0;ow<op->out_w;ow+=op->stride) {
            for(int ic=0;ic<op->in_channels;ic++) {
                for(int kh=0;kh<op->kernel_size;kh++) {
                    for(int kw=0;kw<op->kernel_size;kw++) {
                        int inoff = ic * (op->in_h) * (op->in_w) + (oh+kh-op->padding) * op->in_w + (ow+kw-op->padding);
                        //* padding
                        if( (oh+kh - op->padding) < 0 ||
                            (ow+kw - op->padding) < 0) {
                            tmp[tmp_idx++] = 0;
                        } else {
                            tmp[tmp_idx++] = in[inoff];
                        }
                        if(tmp_idx == 4) { //* store to src
                            __m256d tvec = _mm256_loadu_pd(tmp);
                            __mmask8 tmask = _mm256_cmp_pd_mask(tvec, zvec, _CMP_NEQ_US);
                            unsigned int nnz_cnt = _mm_popcnt_u32(tmask);
                            masks[_mask_idx] = tmask | ((nnz_cnt & 0x0f) << 4);
                            _mm256_mask_compressstoreu_pd((void*)out_cur, tmask, tvec);
                            masks_off[_mask_idx++] = (int)out_cur - (int)out;
                            out_cur += nnz_cnt;
                            tmp_idx = 0;
                        }
                    }
                }
            }
        }
    }

    *masks_idx = _mask_idx;

}



