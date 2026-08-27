#include "src/kernel/sqrt.h"

#include <immintrin.h>

#include <cmath>

#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t SqrtKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Sqrt::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != DataType::FP32) ||
        param->input->dims().data() != param->out->dims().data()) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();
    int64_t index = 0;
    for (; index + 8 <= numel; index += 8) {
        _mm256_storeu_ps(output + index, _mm256_sqrt_ps(_mm256_loadu_ps(input + index)));
    }
    for (; index < numel; ++index) {
        output[index] = std::sqrt(input[index]);
    }
    return 0;
}

void EnsureX86SqrtKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sqrt",
            []() { return std::make_unique<SqrtKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
