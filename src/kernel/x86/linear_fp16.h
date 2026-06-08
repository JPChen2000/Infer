#ifndef FEATHER_KERNEL_X86_LINEAR_FP16_H
#define FEATHER_KERNEL_X86_LINEAR_FP16_H

#include <cstdint>

#include "src/kernel/x86/linear_fp32.h"

namespace feather {
namespace kernel {
namespace x86 {

int32_t ComputeLinearRowMajorX86Fp16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t m,
                                     int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_FP16_H
