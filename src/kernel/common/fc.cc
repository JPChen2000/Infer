#include "core/tensor.h"
#include "util/logger.h"
#include "src/kernel/common/kernel_io.h"
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
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();
}  // namespace

template <DataType dtype>
int32_t ComputeFcCommon(feather::operators::FcParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    const int64_t m = param->input->dims()[0];
    const int64_t n = param->input->dims()[1];
    const int64_t c = param->w->dims()[1];
    param->out->set_data_type(dtype);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n; ++k) {
                sum += TensorIO<dtype>::Read(param->input.get(), i * n + k) *
                       TensorIO<dtype>::Read(param->w.get(), k * c + j);
            }
            if (param->bias != nullptr && param->bias->IsInitialized()) {
                if (param->bias->dims().size() == 1) {
                    sum += TensorIO<dtype>::Read(param->bias.get(), j);
                } else {
                    sum += TensorIO<dtype>::Read(param->bias.get(), i * c + j);
                }
            }
            TensorIO<dtype>::Write(param->out.get(), i * c + j, sum);
        }
    }

    return 0;
}

template<>
int32_t FcKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::FC::FP32");
    return ComputeFcCommon<DataType::FP32>(static_cast<FcParam*>(param_));
}

template<>
int32_t FcKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::FC::FP16");
    return ComputeFcCommon<DataType::FP16>(static_cast<FcParam*>(param_));
}
typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP32> FcCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, FC, FcCommonFP32Kernel);

typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP16> FcCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, FC, FcCommonFP16Kernel);

void EnsureCommonFcKernelsRegistered() { (void)g_fc_kernels_registered; }

void EnsureFcKernelsRegistered() {
    EnsureCommonFcKernelsRegistered();
    EnsureX86FcKernelsRegistered();
}
}  // namespace kernel
}  // namespace feather
