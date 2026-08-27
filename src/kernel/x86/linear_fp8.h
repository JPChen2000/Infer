#ifndef FEATHER_KERNEL_X86_LINEAR_FP8_H
#define FEATHER_KERNEL_X86_LINEAR_FP8_H

#include <cstdint>
#include <vector>

#include "src/kernel/x86/linear_fp32.h"
#include "util/types.h"

namespace feather {
namespace kernel {
namespace x86 {

class Fp8LinearWorkspace {
   public:
    bool Resize(size_t count);
    float* data() { return values_.data(); }
    const float* data() const { return values_.data(); }

   private:
    std::vector<float> values_;
};

// Stores an immutable [K, N] FP8 RHS in [output_block][K][64] order after
// lossless conversion to BF16. Every finite FP8 value is exactly representable
// in BF16, so this cache removes decode work from the steady-state GEMV while
// retaining the FP8 tensor and quantization scale as the source contract.
class PackedFp8Rhs {
   public:
    bool Pack(DataType dtype, const uint8_t* rhs, int64_t k, int64_t n, uint64_t source_version = 0);
    bool Matches(DataType dtype, const uint8_t* rhs, int64_t k, int64_t n,
                 uint64_t source_version = 0) const;
    const uint16_t* data() const { return data_.data(); }

   private:
    DataType dtype_{DataType::UNKNOWN};
    const uint8_t* source_{nullptr};
    int64_t k_{0};
    int64_t n_{0};
    uint64_t source_version_{0};
    std::vector<uint16_t> data_;
};

// Stores an immutable [N, K] FP8 RHS in [output_block][K][64] order.  The
// packed form makes each output tile contiguous after lossless conversion to
// BF16, avoiding repeated FP8 decode work for the Qwen lm-head.
class PackedFp8TransposedRhs {
   public:
    bool Pack(DataType dtype, const uint8_t* rhs_transposed, int64_t k, int64_t n,
              uint64_t source_version = 0);
    bool Matches(DataType dtype, const uint8_t* rhs_transposed, int64_t k, int64_t n,
                 uint64_t source_version = 0) const;
    const uint16_t* data() const { return data_.data(); }

   private:
    DataType dtype_{DataType::UNKNOWN};
    const uint8_t* source_{nullptr};
    int64_t k_{0};
    int64_t n_{0};
    uint64_t source_version_{0};
    std::vector<uint16_t> data_;
};

// Computes C = A * B for row-major FP8 tensors. The input and output scales
// are supplied separately because the storage format contains only the FP8
// code; all accumulation is performed in FP32.
int32_t ComputeLinearRowMajorX86Fp8(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
    const uint8_t* bias, float bias_scale, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type,
    uint8_t* out, float out_scale, float alpha = 1.0f, float beta = 1.0f);

// Same operation as ComputeLinearRowMajorX86Fp8, using an immutable RHS that
// has already been decoded to BF16 and packed into output tiles. The source
// pointer/version are checked so a stale cache cannot be used after mutation.
int32_t ComputeLinearRowMajorX86Fp8PackedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
    const PackedFp8Rhs& packed_rhs, const uint8_t* bias, float bias_scale, int64_t m, int64_t k, int64_t n,
    LinearBiasType bias_type, uint8_t* out, float out_scale, float alpha = 1.0f, float beta = 1.0f,
    uint64_t source_version = 0, Fp8LinearWorkspace* workspace = nullptr);

// Same packed computation for an ONNX Gemm with transB=1. The source RHS is
// stored as [N, K], but its immutable cache uses the same output-tile layout
// as PackedFp8Rhs after preparation.
int32_t ComputeLinearRowMajorX86Fp8PackedTransposedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const PackedFp8TransposedRhs& packed_rhs, const uint8_t* bias, float bias_scale, int64_t m, int64_t k,
    int64_t n, LinearBiasType bias_type, uint8_t* out, float out_scale, float alpha = 1.0f, float beta = 1.0f,
    uint64_t source_version = 0, Fp8LinearWorkspace* workspace = nullptr);

// Computes C = alpha * A * B^T + beta * bias for an ONNX Gemm with transB=1,
// where B is stored row-major as [N, K]. This access pattern is used by the
// Qwen lm-head and keeps each output row of B contiguous in memory.
int32_t ComputeLinearRowMajorX86Fp8TransposedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const uint8_t* bias, float bias_scale, float alpha, float beta, int64_t m, int64_t k, int64_t n,
    LinearBiasType bias_type, uint8_t* out, float out_scale);

// Computes the terminal FP8 lm-head projection and greedy selection without
// materializing the vocabulary logits.  The optional packed RHS is required
// to match the source pointer/version when supplied.
int32_t ComputeLinearRowMajorX86Fp8TransposedRhsArgmax(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const PackedFp8TransposedRhs* packed_rhs, int64_t k, int64_t n, float out_scale, int64_t* token,
    Fp8LinearWorkspace* workspace = nullptr, uint64_t source_version = 0);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_FP8_H
