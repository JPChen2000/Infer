#include "src/kernel/relu.h"

#include <algorithm>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {
bool g_relu_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "ReLU",
                                               []() { return std::make_unique<ReluKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "ReLU",
                                               []() { return std::make_unique<ReluKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();
}  // namespace

template <DataType dtype>
int32_t ComputeReluKernel(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, std::max(0.0f, TensorIO<dtype>::Read(param->input.get(), i)));
    }
    return 0;
}

template <>
int32_t ReluKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::ReLU::FP32");
    return ComputeReluKernel<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}

template <>
int32_t ReluKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::ReLU::FP16");
    return ComputeReluKernel<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}

typedef feather::kernel::ReluKernel<DeviceType::COMMON, DataType::FP32> ReluCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, ReLU, ReluCommonFP32Kernel);

typedef feather::kernel::ReluKernel<DeviceType::COMMON, DataType::FP16> ReluCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, ReLU, ReluCommonFP16Kernel);

void EnsureCommonReluKernelsRegistered() { (void)g_relu_kernels_registered; }

void EnsureReluKernelsRegistered() {
    EnsureCommonReluKernelsRegistered();
    EnsureX86ReluKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
