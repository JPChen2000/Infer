#include "src/kernel/matmul.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<size_t>(axis)] =
            strides[static_cast<size_t>(axis + 1)] * dims[static_cast<size_t>(axis + 1)];
    }
    return strides;
}

struct MatMulShape {
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
    int64_t batch_count{0};
    size_t batch_rank{0};
    size_t a_gap{0};
    size_t b_gap{0};
    bool singleton_batch{false};
};

bool ValidateBatchedMatMulX86(const feather::operators::MatMulParam* param, MatMulShape* shape) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr || shape == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || !param->out->IsInitialized()) {
        return false;
    }

    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const auto& out_dims = param->out->dims().data();
    const size_t a_rank = a_dims.size();
    const size_t b_rank = b_dims.size();
    const size_t out_rank = out_dims.size();
    if (a_rank < 2 || b_rank < 2 || out_rank < 2 || out_rank != std::max(a_rank, b_rank) ||
        a_dims[a_rank - 1] != b_dims[b_rank - 2]) {
        return false;
    }

    shape->m = a_dims[a_rank - 2];
    shape->k = a_dims[a_rank - 1];
    shape->n = b_dims[b_rank - 1];
    if (shape->m <= 0 || shape->k <= 0 || shape->n <= 0 || out_dims[out_rank - 2] != shape->m ||
        out_dims[out_rank - 1] != shape->n) {
        return false;
    }

    shape->batch_rank = out_rank - 2;
    const size_t a_batch_rank = a_rank - 2;
    const size_t b_batch_rank = b_rank - 2;
    shape->a_gap = shape->batch_rank - a_batch_rank;
    shape->b_gap = shape->batch_rank - b_batch_rank;
    shape->batch_count = 1;
    shape->singleton_batch = true;
    for (size_t axis = 0; axis < shape->batch_rank; ++axis) {
        const int64_t a_dim = axis < shape->a_gap ? 1 : a_dims[axis - shape->a_gap];
        const int64_t b_dim = axis < shape->b_gap ? 1 : b_dims[axis - shape->b_gap];
        const int64_t out_dim = out_dims[axis];
        if (a_dim <= 0 || b_dim <= 0 || out_dim <= 0 ||
            (a_dim != b_dim && a_dim != 1 && b_dim != 1) || out_dim != std::max(a_dim, b_dim) ||
            shape->batch_count > std::numeric_limits<int64_t>::max() / out_dim) {
            return false;
        }
        shape->batch_count *= out_dim;
        shape->singleton_batch = shape->singleton_batch && out_dim == 1;
    }

    if (shape->batch_count <= 0 || shape->batch_count > std::numeric_limits<int64_t>::max() / shape->m ||
        shape->batch_count * shape->m > std::numeric_limits<int64_t>::max() / shape->n ||
        param->out->numel() != shape->batch_count * shape->m * shape->n) {
        return false;
    }
    return true;
}

template <typename T, typename MatrixCompute>
int32_t ComputeBatchedMatMulX86(feather::operators::MatMulParam* param, MatrixCompute matrix_compute) {
    MatMulShape shape;
    if (!ValidateBatchedMatMulX86(param, &shape)) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const auto& out_dims = param->out->dims().data();
    const size_t batch_rank = shape.batch_rank;
    const int64_t m = shape.m;
    const int64_t k = shape.k;
    const int64_t n = shape.n;

    // Decode uses singleton batch dimensions (typically [1, 1, K] x [K, N]).
    // Avoid rebuilding stride vectors and walking batch coordinates when the
    // broadcasted batch has exactly one matrix.
    if (shape.singleton_batch && shape.batch_count == 1) {
        const auto* lhs = reinterpret_cast<const T*>(param->a->raw_data());
        const auto* rhs = reinterpret_cast<const T*>(param->b->raw_data());
        auto* out = reinterpret_cast<T*>(param->out->raw_data());
        return matrix_compute(lhs, rhs, m, k, n, out);
    }

    const size_t a_gap = shape.a_gap;
    const size_t b_gap = shape.b_gap;
    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    std::vector<int64_t> batch_dims(out_dims.begin(), out_dims.begin() + static_cast<int64_t>(batch_rank));
    const auto batch_strides = ComputeStrides(batch_dims);
    const int64_t batch_count = shape.batch_count;

    const auto* lhs = reinterpret_cast<const T*>(param->a->raw_data());
    const auto* rhs = reinterpret_cast<const T*>(param->b->raw_data());
    auto* out = reinterpret_cast<T*>(param->out->raw_data());
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t remaining = batch;
        int64_t a_batch_offset = 0;
        int64_t b_batch_offset = 0;
        for (size_t axis = 0; axis < batch_rank; ++axis) {
            const int64_t coordinate = remaining / batch_strides[axis];
            remaining %= batch_strides[axis];
            if (axis >= a_gap && a_dims[axis - a_gap] != 1) {
                a_batch_offset += coordinate * a_strides[axis - a_gap];
            }
            if (axis >= b_gap && b_dims[axis - b_gap] != 1) {
                b_batch_offset += coordinate * b_strides[axis - b_gap];
            }
        }
        if (matrix_compute(lhs + a_batch_offset, rhs + b_batch_offset, m, k, n, out + batch * m * n) != 0) {
            return -1;
        }
    }
    return 0;
}

template <DataType dtype>
int32_t ComputeMatMulX86(feather::operators::MatMulParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        param->a->data_type() != dtype || param->b->data_type() != dtype) {
        return -1;
    }
    param->out->set_data_type(dtype);
    if constexpr (dtype == DataType::FP32) {
        return ComputeBatchedMatMulX86<float>(
            param, [](const float* lhs, const float* rhs, int64_t m, int64_t k, int64_t n, float* out) {
                return x86::ComputeLinearRowMajorX86Fp32(lhs, rhs, nullptr, m, k, n, x86::LinearBiasType::kNone,
                                                         out);
            });
    } else if constexpr (dtype == DataType::FP16) {
        return ComputeBatchedMatMulX86<uint16_t>(
            param, [](const uint16_t* lhs, const uint16_t* rhs, int64_t m, int64_t k, int64_t n, uint16_t* out) {
                return x86::ComputeLinearRowMajorX86Fp16(lhs, rhs, nullptr, m, k, n, x86::LinearBiasType::kNone,
                                                         out);
            });
    } else {
        return ComputeBatchedMatMulX86<uint16_t>(
            param, [](const uint16_t* lhs, const uint16_t* rhs, int64_t m, int64_t k, int64_t n, uint16_t* out) {
                return x86::ComputeLinearRowMajorX86Bf16(lhs, rhs, nullptr, m, k, n, x86::LinearBiasType::kNone,
                                                         out);
            });
    }
}

int32_t ComputeBf16MatMulX86(feather::operators::MatMulParam* param, x86::Bf16LinearWorkspace* workspace) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        param->a->data_type() != DataType::BF16 || param->b->data_type() != DataType::BF16) {
        return -1;
    }
    MatMulShape shape;
    if (!ValidateBatchedMatMulX86(param, &shape)) {
        return -1;
    }
    param->out->set_data_type(DataType::BF16);

    return ComputeBatchedMatMulX86<uint16_t>(
        param, [workspace](const uint16_t* lhs, const uint16_t* rhs, int64_t m, int64_t k, int64_t n,
                            uint16_t* out) {
            return x86::ComputeLinearRowMajorX86Bf16(lhs, rhs, nullptr, m, k, n, x86::LinearBiasType::kNone, out,
                                                     workspace);
        });
}

}  // namespace

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::MatMul::FP32");
    return ComputeMatMulX86<DataType::FP32>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::MatMul::FP16");
    return ComputeMatMulX86<DataType::FP16>(static_cast<feather::operators::MatMulParam*>(param_));
}

int32_t MatMulKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::MatMul::BF16");
    return ComputeBf16MatMulX86(static_cast<feather::operators::MatMulParam*>(param_), &workspace_);
}

void EnsureX86MatMulKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
