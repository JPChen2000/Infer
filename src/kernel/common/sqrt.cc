#include "src/kernel/sqrt.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_sqrt_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Sqrt", []() {
        return std::make_unique<SqrtKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Sqrt", []() {
        return std::make_unique<SqrtKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Sqrt", []() {
        return std::make_unique<SqrtKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t SqrtKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Sqrt::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP32>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::sqrt(value); });
}

template <>
int32_t SqrtKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Sqrt::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::sqrt(value); });
}

template <>
int32_t SqrtKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Sqrt::BF16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::BF16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::sqrt(value); });
}

void EnsureSqrtKernelsRegistered() { (void)g_sqrt_kernels_registered; }

}  // namespace kernel
}  // namespace feather
