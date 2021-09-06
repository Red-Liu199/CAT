#ifndef RNNT_CORE_H
#define RNNT_CORE_H

#include <cuda_runtime.h>

typedef enum
{
    GATHER_STATUS_SUCCESS = 0,
    GATHER_STATUS_FAILED = 1
} rnntStatus_t;

#ifdef __cplusplus
#include <cstddef>
extern "C"
{
#endif

    rnntStatus_t run_gather_sum(cudaStream_t stream, const float *xs, const float *ys, const unsigned int *xn, const unsigned int *yn,
                                float *gather_xs, const unsigned int *memPref,
                                const unsigned int *framePref, const unsigned int *labelPref,
                                unsigned int N, unsigned int T, unsigned int U, unsigned int V);

    rnntStatus_t run_scatter_grad(cudaStream_t stream, const float *grad_sum, float *grad_x, float *grad_y,
                                  const unsigned int *lx, unsigned int *ly,
                                  unsigned int *sumPref, unsigned int *xCumSum, unsigned int *yCumSum,
                                  unsigned int V, unsigned int lx_max, unsigned int ly_max, unsigned int N);

    rnntStatus_t run_gather_cat(cudaStream_t stream, const float *x_padded, const unsigned int *lx,
                                float *x_gather, const unsigned int *memPref,
                                unsigned int N, unsigned int T, unsigned int V);

    rnntStatus_t run_pad_grad(cudaStream_t stream, const float *grad_gather, const unsigned int *lx,
                              float *grad_padded, const unsigned int *memPref,
                              unsigned int N, unsigned int T, unsigned int V);

#ifdef __cplusplus
}
#endif

#endif //RNNT_CORE_H
