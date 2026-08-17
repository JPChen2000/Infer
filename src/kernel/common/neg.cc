#include "src/kernel/neg.h"

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_neg_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Neg", []() { return std::make_unique<NegKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Neg", []() { return std::make_unique<NegKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Neg", []() { return std::make_unique<NegKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunNeg(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return -value; });
}

}  // namespace

template <> int32_t NegKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Neg::FP32");
    return RunNeg<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t NegKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Neg::FP16");
    return RunNeg<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}
template <> int32_t NegKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Neg::BF16");
    return RunNeg<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

void EnsureNegKernelsRegistered() { (void)g_neg_kernels_registered; }

}  // namespace kernel
}  // namespace feather
