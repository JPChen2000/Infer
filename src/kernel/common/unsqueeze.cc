#include "src/kernel/unsqueeze.h"

#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_unsqueeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Unsqueeze::FP32");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Unsqueeze::FP16");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Unsqueeze::INT64");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Unsqueeze::BOOL");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

void EnsureCommonUnsqueezeKernelsRegistered() { (void)g_common_unsqueeze_kernels_registered; }

void EnsureUnsqueezeKernelsRegistered() { EnsureCommonUnsqueezeKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
