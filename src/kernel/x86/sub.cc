#include "src/kernel/sub.h"

#include "src/kernel/common/elementwise_broadcast.h"
#include "src/kernel/x86/elementwise.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Sub::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }
    if (x86::elementwise_detail::TryComputeScalarBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kSub>(param) ||
        x86::elementwise_detail::TryComputeLastDimensionBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kSub>(param)) {
        return 0;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs - rhs; });
}

template <DataType dtype>
int32_t ComputeX86Fp8Sub(feather::operators::BinaryParam* param) {
    if (param == nullptr) return -1;
    return common_detail::RunBinary<dtype>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                           [](float lhs, float rhs) { return lhs - rhs; });
}

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Sub::FP8E4M3");
    return ComputeX86Fp8Sub<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Sub::FP8E5M2");
    return ComputeX86Fp8Sub<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

void EnsureX86SubKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
