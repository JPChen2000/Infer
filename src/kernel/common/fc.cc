#include "core/tensor.h"
#include "util/logger.h"
#include "util/timer.h"
#include "util/types.h"
#include "src/kernel/fc.h"
#include "src/operator/params.h"
using feather::DataType;
using feather::Tensor;
using feather::operators::FcParam;

namespace feather {
namespace kernel {

namespace {
bool g_fc_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP32>>(); });
    return true;
}();
}  // namespace

template<>
int32_t FcKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::FC::FP32");
    auto param = static_cast<FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }
    const float* input = param->input->data<float>();
    const float* weight = param->w->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;

    int m = param->input->dims()[0];
    int n = param->input->dims()[1];
    int c = param->w->dims()[1];

    float* out = param->out->mutable_data<float>();
    memset(static_cast<void*>(out), 0, param->out->memory_size());
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < c; ++j) {
            for (auto k = 0; k < n; ++k) {
                out[i * c + j] += input[i * n + k] * weight[k * c + j];
            }
            if (bias) {
                if (param->bias->dims().size() == 1) {
                    out[i * c + j] += bias[j];
                } else {
                    out[i * c + j] += bias[i * c + j];
                }
            }
        }
    }

    return 0;
}
typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP32> FcCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, FC, FcCommonFP32Kernel);

void EnsureFcKernelsRegistered() { (void)g_fc_kernels_registered; }
}  // namespace kernel
}  // namespace feather
