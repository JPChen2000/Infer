#ifndef FEATHER_KERNEL_X86_LINEAR_H
#define FEATHER_KERNEL_X86_LINEAR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/kernel/x86/linear_fp32.h"

namespace feather {
namespace kernel {
namespace x86 {

size_t Bf16LinearWorkerCount(int64_t m, int64_t k, int64_t n);

// Conversion storage reused by a MatMul/Gemm kernel across decode calls. The
// workspace deliberately owns only activation conversion data; immutable
// weights are kept in the packed RHS objects below.
class Bf16LinearWorkspace {
   public:
    bool Resize(size_t count);
    bool ResizePanel(size_t count);
    float* data() { return values_.data(); }
    const float* data() const { return values_.data(); }
    float* panel_data() { return panel_.data(); }
    size_t size() const { return values_.size(); }

   private:
    std::vector<float> values_;
    std::vector<float> panel_;
};

// Stores a read-only BF16 RHS in output-column blocks. The packed layout is
// [output_block][k][64], which turns the decode-time GEMV access into a
// contiguous stream for each worker.
class PackedBf16Rhs {
   public:
    bool Pack(const uint16_t* rhs, int64_t k, int64_t n, uint64_t source_version = 0);
    bool Matches(const uint16_t* rhs, int64_t k, int64_t n, uint64_t source_version = 0) const;
    const uint16_t* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

   private:
    const uint16_t* source_{nullptr};
    int64_t k_{0};
    int64_t n_{0};
    uint64_t source_version_{0};
    std::vector<uint16_t> data_;
};

// Stores a read-only BF16 RHS supplied as [N, K] for transposed-B Gemm in
// the same [output_block][K][64] layout used by the decode-time GEMV kernel.
// This makes the 64 output weights for each K value contiguous.
class PackedBf16TransposedRhs {
   public:
    bool Pack(const uint16_t* rhs_transposed, int64_t k, int64_t n, uint64_t source_version = 0);
    bool Matches(const uint16_t* rhs_transposed, int64_t k, int64_t n, uint64_t source_version = 0) const;
    const uint16_t* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

   private:
    const uint16_t* source_{nullptr};
    int64_t k_{0};
    int64_t n_{0};
    uint64_t source_version_{0};
    std::vector<uint16_t> data_;
};

int32_t ComputeLinearRowMajorX86Bf16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t m,
                                     int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out,
                                     Bf16LinearWorkspace* workspace = nullptr);

int32_t ComputeLinearRowMajorX86Bf16PackedRhs(const uint16_t* lhs, const uint16_t* rhs,
                                              const PackedBf16Rhs& packed_rhs, const uint16_t* bias, int64_t m,
                                              int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out,
                                              Bf16LinearWorkspace* workspace = nullptr,
                                              uint64_t source_version = 0);

// Single-threaded packed RHS entry point. This keeps the decode hot path out
// of the OpenMP/lambda dispatch wrapper when the workload is below the
// parallelization threshold.
int32_t ComputeLinearRowMajorX86Bf16PackedRhsSingleThread(
    const uint16_t* lhs, const uint16_t* rhs, const PackedBf16Rhs& packed_rhs, const uint16_t* bias, int64_t m,
    int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out, Bf16LinearWorkspace* workspace = nullptr,
    uint64_t source_version = 0);


int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhs(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs,
    const uint16_t* bias, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type, float alpha, float beta,
    uint16_t* out, Bf16LinearWorkspace* workspace = nullptr, uint64_t source_version = 0);

// Single-threaded packed transposed-RHS entry point used by the Qwen lm-head
// to avoid rebuilding the OpenMP dispatch wrapper for a serial decode call.
int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsSingleThread(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs,
    const uint16_t* bias, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type, float alpha, float beta,
    uint16_t* out, Bf16LinearWorkspace* workspace = nullptr, uint64_t source_version = 0);

// Computes the greedy token directly from a packed transposed RHS. The BF16
// rounding and NaN handling match a Gemm write followed by Qwen's greedy scan,
// but no logits output buffer is materialized.
int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmax(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs, int64_t k,
    int64_t n, int64_t* token, Bf16LinearWorkspace* workspace = nullptr, uint64_t source_version = 0);

// Single-threaded lm-head argmax entry point. The AVX2 path uses a 32-column
// tile here to keep the accumulator set within the available YMM registers.
int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmaxSingleThread(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs, int64_t k,
    int64_t n, int64_t* token, Bf16LinearWorkspace* workspace = nullptr, uint64_t source_version = 0);

// Computes lhs[m, k] * rhs_transposed[n, k]^T. This is the layout used by
// ONNX Gemm when transB is set.
int32_t ComputeLinearRowMajorX86Bf16TransposedRhs(const uint16_t* lhs, const uint16_t* rhs_transposed,
                                                   const uint16_t* bias, int64_t m, int64_t k, int64_t n,
                                                   LinearBiasType bias_type, float alpha, float beta, uint16_t* out,
                                                   Bf16LinearWorkspace* workspace = nullptr);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_H
