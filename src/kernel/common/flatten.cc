#include "src/kernel/flatten.h"

#include <cstring>

#include "util/types.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_flatten_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Flatten",
                                                []() { return std::make_unique<FlattenKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Flatten",
                                                []() { return std::make_unique<FlattenKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeFlattenKernel(feather::operators::FlattenParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    std::memcpy(param->out->raw_data(), param->input->raw_data(), bytes);
    return 0;
}

template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Flatten::FP32");
    return ComputeFlattenKernel<DataType::FP32>(static_cast<feather::operators::FlattenParam*>(param_));
}

template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Flatten::FP16");
    return ComputeFlattenKernel<DataType::FP16>(static_cast<feather::operators::FlattenParam*>(param_));
}

typedef feather::kernel::FlattenKernel<DeviceType::COMMON, DataType::FP32> FlattenCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Flatten, FlattenCommonFP32Kernel);

typedef feather::kernel::FlattenKernel<DeviceType::COMMON, DataType::FP16> FlattenCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Flatten, FlattenCommonFP16Kernel);

void EnsureFlattenKernelsRegistered() {
    (void)g_flatten_kernels_registered;
    EnsureX86FlattenKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
