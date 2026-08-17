#include "src/kernel/identity.h"

#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_identity_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeIdentityKernel(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    std::memcpy(param->out->raw_data(), param->input->raw_data(), bytes);
    return 0;
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Identity::FP32");
    return ComputeIdentityKernel<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Identity::FP16");
    return ComputeIdentityKernel<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Identity::BF16");
    return ComputeIdentityKernel<DataType::BF16>(static_cast<feather::operators::UnaryParam*>(param_));
}

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::FP32> IdentityCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Identity, IdentityCommonFP32Kernel);

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::FP16> IdentityCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Identity, IdentityCommonFP16Kernel);

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::BF16> IdentityCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Identity, IdentityCommonBF16Kernel);

void EnsureIdentityKernelsRegistered() {
    (void)g_identity_kernels_registered;
    EnsureX86IdentityKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
