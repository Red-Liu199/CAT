#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 64
#define H 16

__global__ void kernel_fill_gather(const float *xs, const float *ys, const unsigned int *xn, const unsigned int *yn,
                                   float *gather_xs, const unsigned int *memPref,
                                   const unsigned int *framePref, const unsigned int *labelPref,
                                   unsigned int V)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int u = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int actual_t = xn[n];
    unsigned int actual_u = yn[n];

    if (t >= actual_t || u >= actual_u)
        return;
    float *ptr_gather = gather_xs + memPref[n] + (t * actual_u + u) * V;
    const float *ptr_x = xs + framePref[n] + t * V;
    const float *ptr_y = ys + labelPref[n] + u * V;
    for (int i = 0; i < V; i++, ptr_gather++, ptr_x++, ptr_y++)
    {
        // gather_xs(n, t, u, i) = xs(n, t, i) + ys(n, u, i)
        *ptr_gather = *ptr_x + *ptr_y;
    }
}

rnntStatus_t run_gather_sum(cudaStream_t stream, const float *xs, const float *ys, const unsigned int *xn, const unsigned int *yn,
                            float *gather_xs, const unsigned int *memPref,
                            const unsigned int *framePref, const unsigned int *labelPref,
                            unsigned int N, unsigned int T, unsigned int U, unsigned int V)
{

    dim3 threads1(W, H);
    dim3 blocks1((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks1, threads1, 0, stream>>>(xs, ys, xn, yn, gather_xs, memPref, framePref, labelPref, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}
