#include "src/kernel/pow.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_pow_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Pow",
                                                []() { return std::make_unique<PowKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Pow",
                                                []() { return std::make_unique<PowKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Pow",
                                                []() { return std::make_unique<PowKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputePowKernel(feather::operators::PowParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, std::pow(TensorIO<dtype>::Read(param->input.get(), i), exponent));
    }
    return 0;
}

template <>
int32_t PowKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Pow::FP32");
    return ComputePowKernel<DataType::FP32>(static_cast<feather::operators::PowParam*>(param_));
}

template <>
int32_t PowKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Pow::FP16");
    return ComputePowKernel<DataType::FP16>(static_cast<feather::operators::PowParam*>(param_));
}

template <>
int32_t PowKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Pow::BF16");
    return ComputePowKernel<DataType::BF16>(static_cast<feather::operators::PowParam*>(param_));
}

typedef feather::kernel::PowKernel<DeviceType::COMMON, DataType::FP32> PowCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Pow, PowCommonFP32Kernel);

typedef feather::kernel::PowKernel<DeviceType::COMMON, DataType::FP16> PowCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Pow, PowCommonFP16Kernel);

void EnsureCommonPowKernelsRegistered() { (void)g_pow_kernels_registered; }

void EnsurePowKernelsRegistered() {
    EnsureCommonPowKernelsRegistered();
    EnsureX86PowKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
