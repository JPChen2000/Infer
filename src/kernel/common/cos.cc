#include "src/kernel/cos.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_cos_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunCos(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::cos(value); });
}

}  // namespace

template <> int32_t CosKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Cos::FP32");
    return RunCos<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t CosKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Cos::FP16");
    return RunCos<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t CosKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Cos::BF16");
    return RunCos<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureCosKernelsRegistered() { (void)g_cos_kernels_registered; }

}  // namespace kernel
}  // namespace feather
