#include "src/kernel/div.h"

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_div_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Div", []() {
        return std::make_unique<DivKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t DivKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Div::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs / rhs; });
}

template <>
int32_t DivKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Div::FP16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunBinary<DataType::FP16>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs / rhs; });
}

void EnsureDivKernelsRegistered() { (void)g_div_kernels_registered; }

}  // namespace kernel
}  // namespace feather
