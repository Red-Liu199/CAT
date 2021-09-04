#include <tuple>
#include <string>
// #include <iostream>

#include <THC/THC.h>

#include <torch/types.h>
#include <torch/extension.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)          \
    TORCH_CHECK((x).is_contiguous(), \
                #x " must be contiguous")

#define CHECK_CUDA(x)                   \
    TORCH_CHECK((x).device().is_cuda(), \
                #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, \
                #x " must be a Float tensor")

#define CHECK_INT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Int, \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

at::Tensor gather_sum(
    const at::Tensor &xs, const at::Tensor &ys,
    const at::Tensor &xn, const at::Tensor &yn)
{
    // Check contiguous
    CHECK_CONTIGUOUS(xs);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(xn);
    CHECK_CONTIGUOUS(yn);
    // Check types
    CHECK_FLOAT(xs);
    CHECK_FLOAT(ys);
    CHECK_INT(xn);
    CHECK_INT(yn);
    // Check device
    CHECK_CUDA(xs);
    CHECK_CUDA(ys);
    CHECK_CUDA(xn);
    CHECK_CUDA(yn);
    // Check number of dimensions and elements
    TORCH_CHECK(xs.dim() == 2, "xs must have 2 dimensions")
    TORCH_CHECK(ys.dim() == 2, "ys must have 2 dimensions")
    TORCH_CHECK(xn.size(0) == yn.size(0), "xn and yn shape must be equal (N,)")
    TORCH_CHECK(xs.size(0) == xn.sum().item<int64_t>(), "xs shape must be equal to (sum(xn), )")
    TORCH_CHECK(ys.size(0) == yn.sum().item<int64_t>(), "ys shape must be equal to (sum(yn), )")

    const auto N = xn.size(0);
    const auto T = xn.max().item<int64_t>(); // max of {T_i}
    const auto U = yn.max().item<int64_t>(); // max of {U_i}
    const auto V = xs.size(1);

    auto memPref = (xn * yn).cumsum(0, at::ScalarType::Int);
    auto labelPref = yn.cumsum(0, at::ScalarType::Int);
    auto framePref = xn.cumsum(0, at::ScalarType::Int);

    int64_t STU = memPref[-1].item<int64_t>();

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}) * V;
        auto cumsumFramePref = framePref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
        labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
        framePref.index_put_({Slice(1, None, None)}, cumsumFramePref);
    }
    memPref[0] = 0;
    labelPref[0] = 0;
    framePref[0] = 0;

    auto stream = at::cuda::getCurrentCUDAStream(xs.device().index());

    rnntStatus_t status;

    at::TensorOptions gather_xs_opts(xs.device());
    gather_xs_opts = gather_xs_opts.dtype(at::ScalarType::Float);

    auto gather_xs_shape = {STU, V}; // (\sum_{T_i*(U_i+1)}, 2)
    at::Tensor gather_xs = at::empty(gather_xs_shape, gather_xs_opts);

    status = run_gather_sum(stream, xs.data_ptr<float>(), ys.data_ptr<float>(),
                            (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
                            gather_xs.data_ptr<float>(), (unsigned int *)memPref.data_ptr<int>(),
                            (unsigned int *)framePref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                            N, T, U, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather sum status " + std::to_string(status));

    return gather_xs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "gather_sum",
        &gather_sum,
        "CUDA based gather sum.",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("xn"),
        pybind11::arg("yn"));
}
