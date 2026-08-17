#include "src/kernel/erf.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_erf_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ErfKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Erf::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP32>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::erf(value); });
}

template <>
int32_t ErfKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Erf::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::erf(value); });
}

void EnsureErfKernelsRegistered() { (void)g_erf_kernels_registered; }

}  // namespace kernel
}  // namespace feather
