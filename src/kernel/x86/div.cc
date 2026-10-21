#include "src/kernel/div.h"

#include <memory>

#include "src/kernel/common/elementwise_broadcast.h"
#include "src/kernel/x86/elementwise.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t DivKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Div::FP32");
    auto* param = ParamAs<feather::operators::BinaryParam>();
    if (param == nullptr) {
        return -1;
    }
    if (x86::elementwise_detail::TryComputeScalarBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kDiv>(param) ||
        x86::elementwise_detail::TryComputeLastDimensionBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kDiv>(param)) {
        return 0;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs / rhs; });
}

template <DataType dtype>
int32_t ComputeX86Fp8Div(feather::operators::BinaryParam* param) {
    if (param == nullptr) return -1;
    return common_detail::RunBinary<dtype>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                           [](float lhs, float rhs) { return lhs / rhs; });
}

template <>
int32_t DivKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Div::FP8E4M3");
    return ComputeX86Fp8Div<DataType::FP8E4M3>(ParamAs<feather::operators::BinaryParam>());
}

template <>
int32_t DivKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Div::FP8E5M2");
    return ComputeX86Fp8Div<DataType::FP8E5M2>(ParamAs<feather::operators::BinaryParam>());
}



}  // namespace kernel
}  // namespace feather
