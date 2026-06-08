#include "src/kernel/identity.h"

#include <immintrin.h>

#include <cstring>

#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

template <DataType dtype>
int32_t ComputeIdentityFallback(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    std::memcpy(param->out->raw_data(), param->input->raw_data(), bytes);
    return 0;
}

template <typename T>
void CopyRawBlock(const T* src, T* dst, int64_t count) {
    std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(T));
}

template <>
void CopyRawBlock<float>(const float* src, float* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 vec = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, vec);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(float));
    }
}

template <>
void CopyRawBlock<uint16_t>(const uint16_t* src, uint16_t* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vec);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(uint16_t));
    }
}

template <typename T>
int32_t ComputeIdentityRaw(feather::operators::UnaryParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    CopyRawBlock(param->input->data<T>(), param->out->mutable_data<T>(), param->input->numel());
    return 0;
}

}  // namespace

template <>
int32_t IdentityKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Identity::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeIdentityFallback<DataType::FP32>(param);
    }
    return ComputeIdentityRaw<float>(param, DataType::FP32);
}

template <>
int32_t IdentityKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Identity::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeIdentityFallback<DataType::FP16>(param);
    }
    return ComputeIdentityRaw<uint16_t>(param, DataType::FP16);
}

void EnsureX86IdentityKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Identity",
            []() { return std::make_unique<IdentityKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Identity",
            []() { return std::make_unique<IdentityKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
