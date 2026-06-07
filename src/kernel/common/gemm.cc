#include "src/kernel/gemm.h"

#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_gemm_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP32>>(); });
    return true;
}();

}  // namespace

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Gemm::FP32");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* lhs = param->a->data<float>();
    const float* rhs = param->b->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;

    const int m = static_cast<int>(param->a->dims()[0]);
    const int k = static_cast<int>(param->a->dims()[1]);
    const int n = static_cast<int>(param->b->dims()[1]);

    float* out = param->out->mutable_data<float>();
    memset(static_cast<void*>(out), 0, param->out->memory_size());

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int t = 0; t < k; ++t) {
                out[i * n + j] += lhs[i * k + t] * rhs[t * n + j];
            }
            if (bias != nullptr) {
                if (param->bias->dims().size() == 1) {
                    out[i * n + j] += bias[j];
                } else {
                    out[i * n + j] += bias[i * n + j];
                }
            }
        }
    }

    return 0;
}

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP32> GemmCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Gemm, GemmCommonFP32Kernel);

void EnsureCommonGemmKernelsRegistered() { (void)g_gemm_kernels_registered; }

void EnsureGemmKernelsRegistered() {
    EnsureCommonGemmKernelsRegistered();
    EnsureX86GemmKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
