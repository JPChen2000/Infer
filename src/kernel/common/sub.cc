#include "src/kernel/sub.h"

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_sub_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Sub", []() {
        return std::make_unique<SubKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SubKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Sub::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs - rhs; });
}

template <>
int32_t SubKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Sub::FP16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunBinary<DataType::FP16>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs - rhs; });
}

void EnsureSubKernelsRegistered() { (void)g_sub_kernels_registered; }

}  // namespace kernel
}  // namespace feather
