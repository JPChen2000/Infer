#include "src/kernel/matmul.h"

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_matmul_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP32>>(); });
    return true;
}();

}  // namespace

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::MatMul::FP32");
    auto* param = static_cast<feather::operators::MatMulParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* lhs = param->a->data<float>();
    const float* rhs = param->b->data<float>();
    float* out = param->out->mutable_data<float>();

    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                sum += lhs[i * k + t] * rhs[t * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    return 0;
}

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP32> MatMulCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, MatMul, MatMulCommonFP32Kernel);

void EnsureCommonMatMulKernelsRegistered() { (void)g_matmul_kernels_registered; }

void EnsureMatMulKernelsRegistered() {
    EnsureCommonMatMulKernelsRegistered();
    EnsureX86MatMulKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
