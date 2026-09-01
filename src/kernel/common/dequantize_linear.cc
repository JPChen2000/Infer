#include "src/kernel/dequantize_linear.h"

#include "src/kernel/common/quantize_linear_utils.h"

namespace feather {
namespace kernel {
namespace {

bool g_dequantize_linear_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT8, "DequantizeLinear",
                                                []() { return std::make_unique<CommonDequantizeLinearKernel>(); });
    return true;
}();

}  // namespace

int32_t CommonDequantizeLinearKernel::compute() {
    auto* param = static_cast<operators::DequantizeLinearParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::INT8) return -1;
    return quantize_linear_detail::ComputeDequantizeLinear(param);
}

void EnsureDequantizeLinearKernelsRegistered() { (void)g_dequantize_linear_kernels_registered; }

}  // namespace kernel
}  // namespace feather
