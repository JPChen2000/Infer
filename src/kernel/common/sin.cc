#include "src/kernel/sin.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_sin_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunSin(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::sin(value); });
}

}  // namespace

template <> int32_t SinKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Sin::FP32");
    return RunSin<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t SinKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Sin::FP16");
    return RunSin<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t SinKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Sin::BF16");
    return RunSin<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureSinKernelsRegistered() { (void)g_sin_kernels_registered; }

}  // namespace kernel
}  // namespace feather
