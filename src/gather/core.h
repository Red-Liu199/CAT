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

#ifdef __cplusplus
}
#endif

#endif //RNNT_CORE_H
