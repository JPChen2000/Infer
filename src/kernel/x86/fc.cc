#include "src/kernel/fc.h"

#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_fc_x86_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP32, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP16, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t FcKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::FC::FP32");
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* input = param->input->data<float>();
    const float* weight = param->w->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    const auto bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);

    return x86::ComputeLinearRowMajorX86Fp32(input, weight, bias, m, k, n, bias_type, param->out->mutable_data<float>());
}

template <>
int32_t FcKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::FC::FP16");
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    const uint16_t* input = param->input->data<uint16_t>();
    const uint16_t* weight = param->w->data<uint16_t>();
    const uint16_t* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<uint16_t>() : nullptr;
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    const auto bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);

    return x86::ComputeLinearRowMajorX86Fp16(input, weight, bias, m, k, n, bias_type,
                                             param->out->mutable_data<uint16_t>());
}

void EnsureX86FcKernelsRegistered() { (void)g_fc_x86_kernels_registered; }

}  // namespace kernel
}  // namespace feather
