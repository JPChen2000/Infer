#include "src/kernel/matmul.h"

#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::MatMul::FP32");
    auto* param = static_cast<feather::operators::MatMulParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    return x86::ComputeLinearRowMajorX86Fp32(param->a->data<float>(), param->b->data<float>(), nullptr,
                                             param->a->dims()[0], param->a->dims()[1], param->b->dims()[1],
                                             x86::LinearBiasType::kNone, param->out->mutable_data<float>());
}

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::MatMul::FP16");
    auto* param = static_cast<feather::operators::MatMulParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    return x86::ComputeLinearRowMajorX86Fp16(param->a->data<uint16_t>(), param->b->data<uint16_t>(), nullptr,
                                             param->a->dims()[0], param->a->dims()[1], param->b->dims()[1],
                                             x86::LinearBiasType::kNone, param->out->mutable_data<uint16_t>());
}

void EnsureX86MatMulKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
