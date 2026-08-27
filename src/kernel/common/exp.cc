#include "src/kernel/exp.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_exp_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunExp(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::exp(value); });
}

}  // namespace

template <> int32_t ExpKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Exp::FP32");
    return RunExp<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t ExpKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Exp::FP16");
    return RunExp<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t ExpKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Exp::BF16");
    return RunExp<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureExpKernelsRegistered() {
    (void)g_exp_kernels_registered;
    EnsureX86ExpKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
