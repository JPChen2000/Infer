#include "src/kernel/matmul.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

float HorizontalAdd(__m256 vec) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vec);
    float sum = 0.0f;
    for (float value : tmp) {
        sum += value;
    }
    return sum;
}

int32_t ComputeMatMulX86Fp32(feather::operators::MatMulParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* lhs = param->a->data<float>();
    const float* rhs = param->b->data<float>();
    float* out = param->out->mutable_data<float>();

    const int64_t m = param->a->dims()[0];
    const int64_t k = param->a->dims()[1];
    const int64_t n = param->b->dims()[1];

    std::vector<float> rhs_t(static_cast<size_t>(k * n));
    for (int64_t row = 0; row < k; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            rhs_t[static_cast<size_t>(col * k + row)] = rhs[row * n + col];
        }
    }

    for (int64_t i = 0; i < m; ++i) {
        const float* lhs_row = lhs + i * k;
        for (int64_t j = 0; j < n; ++j) {
            const float* rhs_row = rhs_t.data() + j * k;
            __m256 acc = _mm256_setzero_ps();
            int64_t t = 0;
            for (; t + 8 <= k; t += 8) {
                const __m256 lhs_vec = _mm256_loadu_ps(lhs_row + t);
                const __m256 rhs_vec = _mm256_loadu_ps(rhs_row + t);
                acc = _mm256_fmadd_ps(lhs_vec, rhs_vec, acc);
            }
            float sum = HorizontalAdd(acc);
            for (; t < k; ++t) {
                sum += lhs_row[t] * rhs_row[t];
            }
            out[i * n + j] = sum;
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t MatMulKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::MatMul::FP32");
    return ComputeMatMulX86Fp32(static_cast<feather::operators::MatMulParam*>(param_));
}

void EnsureX86MatMulKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
