#include "src/kernel/reshape.h"

#include <cstring>

#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_reshape_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Reshape",
                                                []() { return std::make_unique<ReshapeKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Reshape",
                                                []() { return std::make_unique<ReshapeKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BOOL, "Reshape",
                                                []() { return std::make_unique<ReshapeKernel<DeviceType::COMMON, DataType::BOOL>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeReshapeKernel(feather::operators::ReshapeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    std::memcpy(param->out->raw_data(), param->input->raw_data(), bytes);
    return 0;
}

template <>
int32_t ReshapeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Reshape::FP32");
    return ComputeReshapeKernel<DataType::FP32>(static_cast<feather::operators::ReshapeParam*>(param_));
}

template <>
int32_t ReshapeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Reshape::FP16");
    return ComputeReshapeKernel<DataType::FP16>(static_cast<feather::operators::ReshapeParam*>(param_));
}

template <>
int32_t ReshapeKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Reshape::BOOL");
    return ComputeReshapeKernel<DataType::BOOL>(static_cast<feather::operators::ReshapeParam*>(param_));
}

typedef feather::kernel::ReshapeKernel<DeviceType::COMMON, DataType::FP32> ReshapeCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Reshape, ReshapeCommonFP32Kernel);

typedef feather::kernel::ReshapeKernel<DeviceType::COMMON, DataType::FP16> ReshapeCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Reshape, ReshapeCommonFP16Kernel);

typedef feather::kernel::ReshapeKernel<DeviceType::COMMON, DataType::BOOL> ReshapeCommonBoolKernel;
REGISTER_KERNEL(COMMON, BOOL, Reshape, ReshapeCommonBoolKernel);

void EnsureCommonReshapeKernelsRegistered() { (void)g_reshape_kernels_registered; }

void EnsureReshapeKernelsRegistered() {
    EnsureCommonReshapeKernelsRegistered();
    EnsureX86ReshapeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
