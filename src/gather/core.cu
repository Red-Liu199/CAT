#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 64
#define H 16

__global__ void kernel_fill_gather(const float *xs, const float *ys, const unsigned int *lx, const unsigned int *ly,
                                   float *gather_xs, const unsigned int *memPref,
                                   const unsigned int *framePref, const unsigned int *labelPref,
                                   unsigned int V)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int u = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int actual_t = lx[n];
    unsigned int actual_u = ly[n];

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

rnntStatus_t run_gather_sum(cudaStream_t stream, const float *xs, const float *ys, const unsigned int *lx, const unsigned int *ly,
                            float *gather_xs, const unsigned int *memPref,
                            const unsigned int *framePref, const unsigned int *labelPref,
                            unsigned int N, unsigned int T, unsigned int U, unsigned int V)
{

    dim3 threads1(W, H);
    dim3 blocks1((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks1, threads1, 0, stream>>>(xs, ys, lx, ly, gather_xs, memPref, framePref, labelPref, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}

__global__ void kernel_fill_grad_x(const float *grad_sum, const unsigned int *ly,
                                   float *grad_x, const unsigned int *memPref,
                                   const unsigned int *xCumSum, unsigned int V)
{ // (Tm, V, N)
    unsigned int xi = blockIdx.x * W + threadIdx.x;
    unsigned int v = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    if (xi >= xCumSum[n] || (n > 0 && xi < xCumSum[n - 1]) || v >= V)
        return;

    const float *ptr_grad_sum = grad_sum + memPref[n] + v;
    float *ptr_x = grad_x + xi * V + v;
    for (int i = 0; i < ly[n]; i++, ptr_grad_sum++)
    {
        *ptr_x += *ptr_grad_sum;
    }
}
__global__ void kernel_fill_grad_y(const float *grad_sum, const unsigned int *lx,
                                   float *grad_y, const unsigned int *memPref,
                                   const unsigned int *yCumSum, unsigned int V)
{ // (Um, V, N)
    unsigned int xi = blockIdx.x * W + threadIdx.x;
    unsigned int v = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    if (xi >= yCumSum[n] || (n > 0 && xi < yCumSum[n - 1]) || v >= V)
        return;

    const float *ptr_grad_sum = grad_sum + memPref[n] + v;
    float *ptr_y = grad_y + xi * V + v;
    for (int i = 0; i < lx[n]; i++, ptr_grad_sum++)
    {
        *ptr_y += *ptr_grad_sum;
    }
}

rnntStatus_t run_scatter_grad(cudaStream_t stream, const float *grad_sum, float *grad_x, float *grad_y,
                              const unsigned int *lx, unsigned int *ly,
                              unsigned int *sumPref, unsigned int *xCumSum, unsigned int *yCumSum,
                              unsigned int V, unsigned int lx_max, unsigned int ly_max, unsigned int N)
{
    dim3 threads1(W, H);
    dim3 blocks1((lx_max + W - 1) / W, (V + H - 1) / H, N);

    kernel_fill_grad_x<<<blocks1, threads1, 0, stream>>>(grad_sum, ly, grad_x, sumPref, xCumSum, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    dim3 threads2(W, H);
    dim3 blocks2((ly_max + W - 1) / W, (V + H - 1) / H, N);
    kernel_fill_grad_y<<<blocks2, threads2, 0, stream>>>(grad_sum, lx, grad_y, sumPref, yCumSum, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}