#ifndef FEATHER_KERNEL_X86_LINEAR_FP32_H
#define FEATHER_KERNEL_X86_LINEAR_FP32_H

#include <cstddef>
#include <cstdint>

namespace feather {
namespace kernel {
namespace x86 {

enum class LinearBiasType {
    kNone,
    kVector,
    kMatrix,
};

size_t Fp32LinearWorkerCount(int64_t m, int64_t k, int64_t n);

// Large row-major matrices use a small row block so the RHS vector is reused
// across several output rows.  Small and single-row workloads keep the
// existing path to avoid paying the blocking overhead.
bool Fp32LinearUsesBlockedKernel(int64_t m, int64_t k, int64_t n);

int32_t ComputeLinearRowMajorX86Fp32(const float* lhs, const float* rhs, const float* bias, int64_t m, int64_t k,
                                     int64_t n, LinearBiasType bias_type, float* out);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_FP32_H
