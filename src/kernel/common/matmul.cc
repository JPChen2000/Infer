#include "src/kernel/matmul.h"

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_matmul_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeMatMulCommon(feather::operators::MatMulParam* param) {
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
            TensorIO<dtype>::Write(param->out.get(), i * n + j, sum);
        }
    }
    return 0;
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::MatMul::FP32");
    return ComputeMatMulCommon<DataType::FP32>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::MatMul::FP16");
    return ComputeMatMulCommon<DataType::FP16>(static_cast<feather::operators::MatMulParam*>(param_));
}

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP32> MatMulCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, MatMul, MatMulCommonFP32Kernel);

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP16> MatMulCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, MatMul, MatMulCommonFP16Kernel);

void EnsureCommonMatMulKernelsRegistered() { (void)g_matmul_kernels_registered; }

void EnsureMatMulKernelsRegistered() {
    EnsureCommonMatMulKernelsRegistered();
    EnsureX86MatMulKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
