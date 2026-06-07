#include "src/kernel/relu.h"

#include <immintrin.h>

#include <algorithm>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

template <DataType dtype>
int32_t ComputeReluFallback(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, std::max(0.0f, TensorIO<dtype>::Read(param->input.get(), i)));
    }
    return 0;
}

}  // namespace

template <>
int32_t ReluKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::ReLU::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP32) {
        return ComputeReluFallback<DataType::FP32>(param);
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();
    const __m256 zero = _mm256_setzero_ps();

    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        const __m256 vec = _mm256_loadu_ps(input + i);
        _mm256_storeu_ps(output + i, _mm256_max_ps(vec, zero));
    }
    for (; i < numel; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    return 0;
}

template <>
int32_t ReluKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::ReLU::FP16");
    return ComputeReluFallback<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureX86ReluKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReLU",
            []() { return std::make_unique<ReluKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "ReLU",
            []() { return std::make_unique<ReluKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
