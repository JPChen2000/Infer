#include "src/kernel/tanh.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_tanh_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t TanhKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Tanh::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP32>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::tanh(value); });
}

template <>
int32_t TanhKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Tanh::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::tanh(value); });
}

void EnsureTanhKernelsRegistered() { (void)g_tanh_kernels_registered; }

}  // namespace kernel
}  // namespace feather
