#include "src/kernel/gemm.h"

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_gemm_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeGemmCommon(feather::operators::GemmParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];
    param->out->set_data_type(dtype);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                sum += TensorIO<dtype>::Read(param->a.get(), i * k + t) *
                       TensorIO<dtype>::Read(param->b.get(), t * n + j);
            }
            if (param->bias != nullptr && param->bias->IsInitialized()) {
                if (param->bias->dims().size() == 1) {
                    sum += TensorIO<dtype>::Read(param->bias.get(), j);
                } else {
                    sum += TensorIO<dtype>::Read(param->bias.get(), i * n + j);
                }
            }
            TensorIO<dtype>::Write(param->out.get(), i * n + j, sum);
        }
    }

    return 0;
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Gemm::FP32");
    return ComputeGemmCommon<DataType::FP32>(static_cast<feather::operators::GemmParam*>(param_));
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Gemm::FP16");
    return ComputeGemmCommon<DataType::FP16>(static_cast<feather::operators::GemmParam*>(param_));
}

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP32> GemmCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Gemm, GemmCommonFP32Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP16> GemmCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Gemm, GemmCommonFP16Kernel);

void EnsureCommonGemmKernelsRegistered() { (void)g_gemm_kernels_registered; }

void EnsureGemmKernelsRegistered() {
    EnsureCommonGemmKernelsRegistered();
    EnsureX86GemmKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
