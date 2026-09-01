#include "src/kernel/quantize_linear.h"

#include "src/kernel/common/quantize_linear_utils.h"

namespace feather {
namespace kernel {
namespace {

bool g_quantize_linear_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    return true;
}();

}  // namespace

int32_t CommonQuantizeLinearKernel::compute() {
    auto* param = static_cast<operators::QuantizeLinearParam*>(param_);
    if (param == nullptr || param->input == nullptr ||
        (param->input->data_type() != DataType::FP32 && param->input->data_type() != DataType::FP16 &&
         param->input->data_type() != DataType::BF16)) return -1;
    return quantize_linear_detail::ComputeQuantizeLinear(param);
}

void EnsureQuantizeLinearKernelsRegistered() { (void)g_quantize_linear_kernels_registered; }

}  // namespace kernel
}  // namespace feather
