#include "src/kernel/softplus.h"

#include <algorithm>
#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_softplus_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Softplus", []() { return std::make_unique<SoftplusKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Softplus", []() { return std::make_unique<SoftplusKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Softplus", []() { return std::make_unique<SoftplusKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunSoftplus(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(), [](float value) {
        return std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
    });
}

}  // namespace

template <> int32_t SoftplusKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Softplus::FP32");
    return RunSoftplus<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t SoftplusKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Softplus::FP16");
    return RunSoftplus<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t SoftplusKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Softplus::BF16");
    return RunSoftplus<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureSoftplusKernelsRegistered() { (void)g_softplus_kernels_registered; }

}  // namespace kernel
}  // namespace feather
