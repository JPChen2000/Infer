#include "src/kernel/gemm.h"

#include "src/operator/params.h"
#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t GemmKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Gemm::FP32");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* lhs = param->a->data<float>();
    const float* rhs = param->b->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    float* out = param->out->mutable_data<float>();

    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];

    const x86::LinearBiasType bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);
    return x86::ComputeLinearRowMajorX86Fp32(lhs, rhs, bias, m, k, n, bias_type, out);
}

template <>
int32_t GemmKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Gemm::FP16");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const uint16_t* lhs = param->a->data<uint16_t>();
    const uint16_t* rhs = param->b->data<uint16_t>();
    const uint16_t* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<uint16_t>() : nullptr;
    uint16_t* out = param->out->mutable_data<uint16_t>();

    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];

    const x86::LinearBiasType bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);
    return x86::ComputeLinearRowMajorX86Fp16(lhs, rhs, bias, m, k, n, bias_type, out);
}

void EnsureX86GemmKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
