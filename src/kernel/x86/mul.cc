#include "src/kernel/mul.h"

#include <immintrin.h>

#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/fp8_host.h"
#include "src/kernel/x86/elementwise.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                         std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
        const int64_t lhs_dim = i < out_rank - lhs_dims.size() ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = i < out_rank - rhs_dims.size() ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

int64_t ComputeBroadcastOffset(const std::vector<int64_t>& out_coords, const std::vector<int64_t>& in_dims,
                               const std::vector<int64_t>& in_strides) {
    const size_t rank_gap = out_coords.size() - in_dims.size();
    int64_t offset = 0;
    for (size_t i = 0; i < in_dims.size(); ++i) {
        const int64_t coord = in_dims[i] == 1 ? 0 : out_coords[i + rank_gap];
        offset += coord * in_strides[i];
    }
    return offset;
}

}  // namespace

template <DataType dtype>
int32_t ComputeMulFallback(feather::operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }

    std::vector<int64_t> out_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &out_dims)) {
        return -1;
    }
    param->out->set_data_type(dtype);
    const auto out_strides = ComputeStrides(out_dims);
    const auto lhs_strides = ComputeStrides(param->lhs->dims().data());
    const auto rhs_strides = ComputeStrides(param->rhs->dims().data());
    std::vector<int64_t> out_coords(out_dims.size(), 0);

    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        const int64_t lhs_offset = ComputeBroadcastOffset(out_coords, param->lhs->dims().data(), lhs_strides);
        const int64_t rhs_offset = ComputeBroadcastOffset(out_coords, param->rhs->dims().data(), rhs_strides);
        const float lhs = TensorIO<dtype>::Read(param->lhs.get(), lhs_offset);
        const float rhs = TensorIO<dtype>::Read(param->rhs.get(), rhs_offset);
        TensorIO<dtype>::Write(param->out.get(), linear, lhs * rhs);
    }
    return 0;
}

template <>
int32_t MulKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Mul::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }

    if (x86::elementwise_detail::TryComputeScalarBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kMul>(param)) {
        return 0;
    }
    if (x86::elementwise_detail::TryComputeLastDimensionBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kMul>(param)) {
        return 0;
    }
    return ComputeMulFallback<DataType::FP32>(param);
}

template <>
int32_t MulKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Mul::FP16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }

    if (param->lhs->dims().data() != param->rhs->dims().data() ||
        param->lhs->dims().data() != param->out->dims().data() || param->lhs->data_type() != DataType::FP16) {
        return ComputeMulFallback<DataType::FP16>(param);
    }

    param->out->set_data_type(DataType::FP16);
    const uint16_t* lhs = param->lhs->data<uint16_t>();
    const uint16_t* rhs = param->rhs->data<uint16_t>();
    uint16_t* out = param->out->mutable_data<uint16_t>();
    const int64_t numel = param->out->numel();

    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        const __m128i lhs_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs + i));
        const __m128i rhs_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs + i));
        const __m256 lhs_vec = _mm256_cvtph_ps(lhs_half);
        const __m256 rhs_vec = _mm256_cvtph_ps(rhs_half);
        const __m256 out_vec = _mm256_mul_ps(lhs_vec, rhs_vec);
        const __m128i out_half = _mm256_cvtps_ph(out_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), out_half);
    }
    for (; i < numel; ++i) {
        out[i] = FloatToHalf(HalfToFloat(lhs[i]) * HalfToFloat(rhs[i]));
    }
    return 0;
}

template <>
int32_t MulKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Mul::BF16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }
    if (x86::elementwise_detail::TryComputeLastDimensionBroadcast<
            x86::elementwise_detail::BinaryOperation::kMul>(param)) {
        return 0;
    }
    return ComputeMulFallback<DataType::BF16>(param);
}

template <DataType dtype>
int32_t ComputeX86Fp8Mul(feather::operators::BinaryParam* param) {
    return fp8_host::Binary<dtype, fp8_host::BinaryOp::kMul>(param);
}

template <>
int32_t MulKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Mul::FP8E4M3");
    return ComputeX86Fp8Mul<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t MulKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Mul::FP8E5M2");
    return ComputeX86Fp8Mul<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

void EnsureX86MulKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
