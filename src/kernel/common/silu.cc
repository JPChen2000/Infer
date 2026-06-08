#include "src/kernel/silu.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_silu_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "SiLU",
                                                []() { return std::make_unique<SiluKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "SiLU",
                                                []() { return std::make_unique<SiluKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

inline float SiluScalar(float value) {
    return value / (1.0f + std::exp(-value));
}

}  // namespace

template <DataType dtype>
int32_t ComputeSiluKernel(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, SiluScalar(TensorIO<dtype>::Read(param->input.get(), i)));
    }
    return 0;
}

template <>
int32_t SiluKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::SiLU::FP32");
    return ComputeSiluKernel<DataType::FP32>(static_cast<feather::operators::UnaryParam*>(param_));
}

template <>
int32_t SiluKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::SiLU::FP16");
    return ComputeSiluKernel<DataType::FP16>(static_cast<feather::operators::UnaryParam*>(param_));
}

typedef feather::kernel::SiluKernel<DeviceType::COMMON, DataType::FP32> SiluCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, SiLU, SiluCommonFP32Kernel);

typedef feather::kernel::SiluKernel<DeviceType::COMMON, DataType::FP16> SiluCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, SiLU, SiluCommonFP16Kernel);

void EnsureCommonSiluKernelsRegistered() { (void)g_silu_kernels_registered; }

void EnsureSiluKernelsRegistered() {
    EnsureCommonSiluKernelsRegistered();
    EnsureX86SiluKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
