#include "src/kernel/squeeze.h"

#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_squeeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Squeeze", []() {
        return std::make_unique<SqueezeKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Squeeze::FP32");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Squeeze::FP16");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Squeeze::BF16");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Squeeze::INT64");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Squeeze::BOOL");
    return common_tensor_detail::CopyTensor(static_cast<feather::operators::AxesParam*>(param_));
}

void EnsureCommonSqueezeKernelsRegistered() { (void)g_common_squeeze_kernels_registered; }

void EnsureSqueezeKernelsRegistered() { EnsureCommonSqueezeKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
