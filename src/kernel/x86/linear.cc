#include "src/kernel/x86/linear.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "util/bf16.h"
#include "util/threading.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace x86 {

constexpr int64_t kBf16OutputBlock = 32;
// Packed decode weights use a wider tile than the fallback kernel.  Qwen's
// logits width is divisible by 64, so this removes a second packed-block
// setup per K row without changing the source tensor layout contract.
constexpr int64_t kBf16PackedOutputBlock = 64;
constexpr int64_t kBf16ArgmaxSingleThreadOutputBlock = 32;
constexpr int64_t kBf16MinimumParallelMacs = 1 << 18;
// Decode GEMV is memory-bandwidth bound on the target x86 CPUs. Four workers
// saturate the useful bandwidth without paying the large OpenMP team cost seen
// with eight workers on small projection shapes. The environment variable
// remains an explicit escape hatch for machines with a different balance.
constexpr size_t kBf16DefaultWorkerLimit = 4;

bool Bf16LinearWorkspace::Resize(size_t count) {
    if (values_.size() >= count) {
        return true;
    }
    try {
        values_.resize(count);
    } catch (const std::bad_alloc&) {
        values_.clear();
        return false;
    }
    return true;
}

bool Bf16LinearWorkspace::ResizePanel(size_t count) {
    if (panel_.size() >= count) {
        return true;
    }
    try {
        panel_.resize(count);
    } catch (const std::bad_alloc&) {
        panel_.clear();
        return false;
    }
    return true;
}

size_t Bf16LinearWorkerCount(int64_t m, int64_t k, int64_t n) {
    if (m != 1 || k <= 0 || n < 2 * kBf16OutputBlock || k < kBf16MinimumParallelMacs / n) {
        return 1;
    }

    const int64_t output_blocks =
        n / kBf16PackedOutputBlock + (n % kBf16PackedOutputBlock != 0 ? 1 : 0);
    size_t worker_limit = std::min(DefaultThreadCount(), kBf16DefaultWorkerLimit);
    const char* configured_limit = std::getenv("FEATHER_X86_BF16_THREADS");
    if (configured_limit != nullptr && configured_limit[0] != '\0') {
        char* end = nullptr;
        const unsigned long parsed_limit = std::strtoul(configured_limit, &end, 10);
        if (end != configured_limit && *end == '\0' && parsed_limit > 0) {
            worker_limit = std::min(DefaultThreadCount(), static_cast<size_t>(parsed_limit));
        }
    }
#if defined(FEATHER_WITH_OPENMP)
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) {
        worker_limit = std::min(worker_limit, static_cast<size_t>(openmp_limit));
    }
#endif
    return std::min(worker_limit, static_cast<size_t>(output_blocks));
}

namespace {

template <typename Fn>
void ParallelForBf16OutputColumns(int64_t n, size_t worker_count, Fn&& fn) {
    if (worker_count <= 1) {
        fn(0, n);
        return;
    }

#if defined(FEATHER_WITH_OPENMP)
    // Packed decode kernels consume 64-column tiles. Align work boundaries to
    // that tile so a worker never starts in the middle of a packed block.
    const int64_t output_blocks =
        n / kBf16PackedOutputBlock + (n % kBf16PackedOutputBlock != 0 ? 1 : 0);
    const int64_t blocks_per_chunk =
        std::max<int64_t>(1, output_blocks / static_cast<int64_t>(worker_count) +
                                  (output_blocks % static_cast<int64_t>(worker_count) != 0 ? 1 : 0));
    const int64_t task_count =
        output_blocks / blocks_per_chunk + (output_blocks % blocks_per_chunk != 0 ? 1 : 0);
    // Qwen decode runs its graph serially.  Avoid nested teams for callers
    // that opt into outer OpenMP parallelism elsewhere.
    if (omp_in_parallel()) {
        fn(0, n);
        return;
    }

    const int omp_worker_count = static_cast<int>(worker_count);
#pragma omp parallel for schedule(static) num_threads(omp_worker_count)
    for (int64_t task = 0; task < task_count; ++task) {
        const int64_t block = task * blocks_per_chunk;
        const int64_t begin = block * kBf16PackedOutputBlock;
        const int64_t end = std::min(n, (block + blocks_per_chunk) * kBf16PackedOutputBlock);
        fn(begin, end);
    }
#else
    fn(0, n);
#endif
}

inline __m256 LoadBf16x8(const uint16_t* input) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    const __m256i expanded = _mm256_cvtepu16_epi32(packed);
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
}

struct Bf16x16 {
    __m256 lo;
    __m256 hi;
};

inline Bf16x16 LoadBf16x16(const uint16_t* input) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const __m256i low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(packed));
    const __m256i high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(packed, 1));
    return {_mm256_castsi256_ps(_mm256_slli_epi32(low, 16)),
            _mm256_castsi256_ps(_mm256_slli_epi32(high, 16))};
}

inline void FmaBf16x16(__m256 lhs, const uint16_t* rhs, __m256* acc_lo, __m256* acc_hi) {
    const Bf16x16 values = LoadBf16x16(rhs);
    *acc_lo = _mm256_fmadd_ps(lhs, values.lo, *acc_lo);
    *acc_hi = _mm256_fmadd_ps(lhs, values.hi, *acc_hi);
}

inline __m256i RoundBf16Bits(__m256 value) {
    const __m256i bits = _mm256_castps_si256(value);
    const __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff));
    const __m256i exponent_is_inf_or_nan = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa_is_zero = _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256());
    const __m256i nan_mask = _mm256_andnot_si256(mantissa_is_zero, exponent_is_inf_or_nan);
    const __m256i round_bias =
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    __m256i high_bits = _mm256_srli_epi32(_mm256_add_epi32(bits, round_bias), 16);
    return _mm256_or_si256(high_bits, _mm256_and_si256(nan_mask, _mm256_set1_epi32(0x0040)));
}

inline void StoreBf16x8(__m256 value, uint16_t* output) {
    const __m256i high_bits = RoundBf16Bits(value);
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(high_bits),
                                             _mm256_extracti128_si256(high_bits, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), packed);
}

inline __m256 LoadBiasVectorBf16(const uint16_t* bias, LinearBiasType bias_type, int64_t row, int64_t n,
                                 int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return _mm256_setzero_ps();
    }
    const uint16_t* bias_ptr = bias_type == LinearBiasType::kVector ? bias + col : bias + row * n + col;
    return LoadBf16x8(bias_ptr);
}

inline float LoadBiasScalarBf16(const uint16_t* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return 0.0f;
    }
    const uint16_t value = bias_type == LinearBiasType::kVector ? bias[col] : bias[row * n + col];
    return BFloat16ToFloat(value);
}

inline void ConvertBf16ArrayToFp32(const uint16_t* input, int64_t count, float* output) {
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        _mm256_storeu_ps(output + index, LoadBf16x8(input + index));
    }
    for (; index < count; ++index) {
        output[index] = BFloat16ToFloat(input[index]);
    }
}

inline void ConvertRhsPanelToFp32(const uint16_t* rhs, int64_t k, int64_t n, int64_t col, int64_t panel_width,
                                  float* rhs_panel) {
    for (int64_t t = 0; t < k; ++t) {
        const uint16_t* rhs_ptr = rhs + t * n + col;
        float* panel_ptr = rhs_panel + t * panel_width;
        _mm256_storeu_ps(panel_ptr, LoadBf16x8(rhs_ptr));
        if (panel_width == 16) {
            _mm256_storeu_ps(panel_ptr + 8, LoadBf16x8(rhs_ptr + 8));
        }
    }
}

// Decode-time transformer linear layers have one input row. Keep a wider
// output block in registers so each BF16 RHS value is converted once, without
// materializing a temporary FP32 panel.
inline void ComputeSingleRowWideBf16Columns(const float* lhs, const uint16_t* rhs, const uint16_t* bias,
                                            int64_t k, int64_t n, LinearBiasType bias_type, int64_t begin,
                                            int64_t end, uint16_t* out) {
    int64_t col = begin;
    for (; col + kBf16OutputBlock <= end; col += kBf16OutputBlock) {
        __m256 acc0 = LoadBiasVectorBf16(bias, bias_type, 0, n, col);
        __m256 acc1 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 8);
        __m256 acc2 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 16);
        __m256 acc3 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 24);
        for (int64_t t = 0; t < k; ++t) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[t]);
            const uint16_t* rhs_row = rhs + t * n + col;
            acc0 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row), acc0);
            acc1 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 8), acc1);
            acc2 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 16), acc2);
            acc3 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 24), acc3);
        }
        StoreBf16x8(acc0, out + col);
        StoreBf16x8(acc1, out + col + 8);
        StoreBf16x8(acc2, out + col + 16);
        StoreBf16x8(acc3, out + col + 24);
    }

    for (; col + 8 <= end; col += 8) {
        __m256 acc = LoadBiasVectorBf16(bias, bias_type, 0, n, col);
        for (int64_t t = 0; t < k; ++t) {
            acc = _mm256_fmadd_ps(_mm256_set1_ps(lhs[t]), LoadBf16x8(rhs + t * n + col), acc);
        }
        StoreBf16x8(acc, out + col);
    }

    for (; col < end; ++col) {
        float sum = LoadBiasScalarBf16(bias, bias_type, 0, n, col);
        for (int64_t t = 0; t < k; ++t) {
            sum += lhs[t] * BFloat16ToFloat(rhs[t * n + col]);
        }
        out[col] = FloatToBFloat16(sum);
    }
}

inline void ComputeSingleRowWideBf16PackedColumns(const float* lhs, const uint16_t* packed_rhs,
                                                  const uint16_t* bias, int64_t k, int64_t n,
                                                  LinearBiasType bias_type, int64_t begin, int64_t end,
                                                  uint16_t* out) {
    int64_t col = begin;
    for (; col + kBf16PackedOutputBlock <= end && col % kBf16PackedOutputBlock == 0;
         col += kBf16PackedOutputBlock) {
        const uint16_t* rhs_block = packed_rhs + (col / kBf16PackedOutputBlock) * k * kBf16PackedOutputBlock;
        __m256 acc0 = LoadBiasVectorBf16(bias, bias_type, 0, n, col);
        __m256 acc1 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 8);
        __m256 acc2 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 16);
        __m256 acc3 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 24);
        __m256 acc4 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 32);
        __m256 acc5 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 40);
        __m256 acc6 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 48);
        __m256 acc7 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 56);
        int64_t row = 0;
        for (; row + 4 <= k; row += 4) {
            const uint16_t* rhs_row0 = rhs_block + (row + 0) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row1 = rhs_block + (row + 1) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row2 = rhs_block + (row + 2) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row3 = rhs_block + (row + 3) * kBf16PackedOutputBlock;
            if (row + 8 < k) {
                __builtin_prefetch(rhs_row0 + 8 * kBf16PackedOutputBlock, 0, 1);
            }
            const __m256 lhs0 = _mm256_set1_ps(lhs[row + 0]);
            const __m256 lhs1 = _mm256_set1_ps(lhs[row + 1]);
            const __m256 lhs2 = _mm256_set1_ps(lhs[row + 2]);
            const __m256 lhs3 = _mm256_set1_ps(lhs[row + 3]);
#define FEATHER_BF16_FMA_ROW(lhs_value, rhs_row) \
            acc0 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row), acc0); \
            acc1 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 8), acc1); \
            acc2 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 16), acc2); \
            acc3 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 24), acc3); \
            acc4 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 32), acc4); \
            acc5 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 40), acc5); \
            acc6 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 48), acc6); \
            acc7 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 56), acc7)
            FEATHER_BF16_FMA_ROW(lhs0, rhs_row0);
            FEATHER_BF16_FMA_ROW(lhs1, rhs_row1);
            FEATHER_BF16_FMA_ROW(lhs2, rhs_row2);
            FEATHER_BF16_FMA_ROW(lhs3, rhs_row3);
#undef FEATHER_BF16_FMA_ROW
        }
        for (; row < k; ++row) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[row]);
            const uint16_t* rhs_row = rhs_block + row * kBf16PackedOutputBlock;
            acc0 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row), acc0);
            acc1 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 8), acc1);
            acc2 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 16), acc2);
            acc3 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 24), acc3);
            acc4 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 32), acc4);
            acc5 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 40), acc5);
            acc6 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 48), acc6);
            acc7 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 56), acc7);
        }
        StoreBf16x8(acc0, out + col);
        StoreBf16x8(acc1, out + col + 8);
        StoreBf16x8(acc2, out + col + 16);
        StoreBf16x8(acc3, out + col + 24);
        StoreBf16x8(acc4, out + col + 32);
        StoreBf16x8(acc5, out + col + 40);
        StoreBf16x8(acc6, out + col + 48);
        StoreBf16x8(acc7, out + col + 56);
    }
    if (col < end) {
        const int64_t packed_offset = col % kBf16PackedOutputBlock;
        const int64_t width = std::min({kBf16PackedOutputBlock - packed_offset, end - col,
                                         static_cast<int64_t>(kBf16PackedOutputBlock)});
        const uint16_t* rhs_block = packed_rhs + (col / kBf16PackedOutputBlock) * k * kBf16PackedOutputBlock +
                                    packed_offset;
        if (width != kBf16OutputBlock && width != kBf16PackedOutputBlock) {
            for (int64_t lane = 0; lane < width; ++lane) {
                float sum = LoadBiasScalarBf16(bias, bias_type, 0, n, col + lane);
                for (int64_t t = 0; t < k; ++t) {
                    sum += lhs[t] * BFloat16ToFloat(rhs_block[t * kBf16PackedOutputBlock + lane]);
                }
                out[col + lane] = FloatToBFloat16(sum);
            }
            return;
        }
        __m256 acc0 = LoadBiasVectorBf16(bias, bias_type, 0, n, col);
        __m256 acc1 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 8);
        __m256 acc2 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 16);
        __m256 acc3 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 24);
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        if (width == kBf16PackedOutputBlock) {
            acc4 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 32);
            acc5 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 40);
            acc6 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 48);
            acc7 = LoadBiasVectorBf16(bias, bias_type, 0, n, col + 56);
        }
        for (int64_t t = 0; t < k; ++t) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[t]);
            const uint16_t* rhs_row = rhs_block + t * kBf16PackedOutputBlock;
            acc0 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row), acc0);
            acc1 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 8), acc1);
            acc2 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 16), acc2);
            acc3 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 24), acc3);
            if (width == kBf16PackedOutputBlock) {
                acc4 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 32), acc4);
                acc5 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 40), acc5);
                acc6 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 48), acc6);
                acc7 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 56), acc7);
            }
        }
        StoreBf16x8(acc0, out + col);
        StoreBf16x8(acc1, out + col + 8);
        StoreBf16x8(acc2, out + col + 16);
        StoreBf16x8(acc3, out + col + 24);
        if (width == kBf16PackedOutputBlock) {
            StoreBf16x8(acc4, out + col + 32);
            StoreBf16x8(acc5, out + col + 40);
            StoreBf16x8(acc6, out + col + 48);
            StoreBf16x8(acc7, out + col + 56);
        }
        col += width;
    }
}

inline int32_t ComputeSingleRowWideBf16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t k,
                                        int64_t n, LinearBiasType bias_type, uint16_t* out,
                                        Bf16LinearWorkspace* workspace) {
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    const size_t worker_count = Bf16LinearWorkerCount(1, k, n);
    ParallelForBf16OutputColumns(n, worker_count, [&](int64_t begin, int64_t end) {
        ComputeSingleRowWideBf16Columns(workspace->data(), rhs, bias, k, n, bias_type, begin, end, out);
    });
    return 0;
}

inline int32_t ComputeSingleRowWideBf16Packed(const uint16_t* lhs, const uint16_t* packed_rhs,
                                              const uint16_t* bias, int64_t k, int64_t n, LinearBiasType bias_type,
                                              uint16_t* out, Bf16LinearWorkspace* workspace) {
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    const size_t worker_count = Bf16LinearWorkerCount(1, k, n);
    ParallelForBf16OutputColumns(n, worker_count, [&](int64_t begin, int64_t end) {
        ComputeSingleRowWideBf16PackedColumns(workspace->data(), packed_rhs, bias, k, n, bias_type, begin, end, out);
    });
    return 0;
}

inline void ComputeSingleRowWideBf16PackedTransposedColumns(
    const float* lhs, const uint16_t* packed_rhs, const uint16_t* bias, int64_t k, int64_t n,
    LinearBiasType bias_type, float alpha, float beta, int64_t begin, int64_t end, uint16_t* out) {
    const __m256 alpha_value = _mm256_set1_ps(alpha);
    const __m256 beta_value = _mm256_set1_ps(beta);
    const bool apply_alpha = alpha != 1.0f;
    const bool apply_bias = bias != nullptr && bias_type != LinearBiasType::kNone;
    for (int64_t col = begin; col < end;) {
        const int64_t packed_offset = col % kBf16PackedOutputBlock;
        const int64_t width = std::min({kBf16PackedOutputBlock - packed_offset, end - col,
                                         static_cast<int64_t>(kBf16PackedOutputBlock)});
        const uint16_t* rhs_block = packed_rhs + (col / kBf16PackedOutputBlock) * k * kBf16PackedOutputBlock +
                                    packed_offset;
        if (width != kBf16OutputBlock && width != kBf16PackedOutputBlock) {
            for (int64_t lane = 0; lane < width; ++lane) {
                float sum = beta * LoadBiasScalarBf16(bias, bias_type, 0, n, col + lane);
                for (int64_t t = 0; t < k; ++t) {
                    sum += alpha * lhs[t] * BFloat16ToFloat(rhs_block[t * kBf16PackedOutputBlock + lane]);
                }
                out[col + lane] = FloatToBFloat16(sum);
            }
            col += width;
            continue;
        }
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (int64_t t = 0; t < k; ++t) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[t]);
            const uint16_t* rhs_row = rhs_block + t * kBf16PackedOutputBlock;
            acc0 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row), acc0);
            acc1 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 8), acc1);
            acc2 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 16), acc2);
            acc3 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 24), acc3);
            if (width == kBf16PackedOutputBlock) {
                acc4 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 32), acc4);
                acc5 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 40), acc5);
                acc6 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 48), acc6);
                acc7 = _mm256_fmadd_ps(lhs_value, LoadBf16x8(rhs_row + 56), acc7);
            }
        }
        if (apply_alpha) {
            acc0 = _mm256_mul_ps(acc0, alpha_value);
            acc1 = _mm256_mul_ps(acc1, alpha_value);
            acc2 = _mm256_mul_ps(acc2, alpha_value);
            acc3 = _mm256_mul_ps(acc3, alpha_value);
            if (width == kBf16PackedOutputBlock) {
                acc4 = _mm256_mul_ps(acc4, alpha_value);
                acc5 = _mm256_mul_ps(acc5, alpha_value);
                acc6 = _mm256_mul_ps(acc6, alpha_value);
                acc7 = _mm256_mul_ps(acc7, alpha_value);
            }
        }
        if (apply_bias) {
            acc0 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col), acc0);
            acc1 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 8), acc1);
            acc2 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 16), acc2);
            acc3 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 24), acc3);
            if (width == kBf16PackedOutputBlock) {
                acc4 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 32), acc4);
                acc5 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 40), acc5);
                acc6 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 48), acc6);
                acc7 = _mm256_fmadd_ps(beta_value, LoadBiasVectorBf16(bias, bias_type, 0, n, col + 56), acc7);
            }
        }
        StoreBf16x8(acc0, out + col);
        StoreBf16x8(acc1, out + col + 8);
        StoreBf16x8(acc2, out + col + 16);
        StoreBf16x8(acc3, out + col + 24);
        if (width == kBf16PackedOutputBlock) {
            StoreBf16x8(acc4, out + col + 32);
            StoreBf16x8(acc5, out + col + 40);
            StoreBf16x8(acc6, out + col + 48);
            StoreBf16x8(acc7, out + col + 56);
        }
        col += width;
    }
}

// The lm-head has no bias and uses the default alpha. Keeping that hot path
// separate lets the compiler reserve registers for all eight accumulators in a
// packed 64-column tile instead of carrying the generic Gemm epilogue state.
inline void ComputeSingleRowWideBf16PackedTransposedLogitsColumns(const float* lhs, const uint16_t* packed_rhs,
                                                                   int64_t k, int64_t begin, int64_t end,
                                                                   uint16_t* out) {
    int64_t col = begin;
    for (; col + kBf16PackedOutputBlock <= end && col % kBf16PackedOutputBlock == 0;
         col += kBf16PackedOutputBlock) {
        const uint16_t* rhs_block = packed_rhs + (col / kBf16PackedOutputBlock) * k * kBf16PackedOutputBlock;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        int64_t t = 0;
        for (; t + 4 <= k; t += 4) {
            const uint16_t* rhs_row0 = rhs_block + (t + 0) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row1 = rhs_block + (t + 1) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row2 = rhs_block + (t + 2) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row3 = rhs_block + (t + 3) * kBf16PackedOutputBlock;
            // The logits RHS is a long stream of 128-byte K rows. Keep a
            // short look-ahead in flight while four independent K rows are
            // exposed to the FMA scheduler.
            if (t + 8 < k) {
                __builtin_prefetch(rhs_row0 + 8 * kBf16PackedOutputBlock, 0, 1);
            }
            const __m256 lhs0 = _mm256_set1_ps(lhs[t + 0]);
            const __m256 lhs1 = _mm256_set1_ps(lhs[t + 1]);
            const __m256 lhs2 = _mm256_set1_ps(lhs[t + 2]);
            const __m256 lhs3 = _mm256_set1_ps(lhs[t + 3]);
            FmaBf16x16(lhs0, rhs_row0 + 0, &acc0, &acc1);
            FmaBf16x16(lhs0, rhs_row0 + 16, &acc2, &acc3);
            FmaBf16x16(lhs0, rhs_row0 + 32, &acc4, &acc5);
            FmaBf16x16(lhs0, rhs_row0 + 48, &acc6, &acc7);
            FmaBf16x16(lhs1, rhs_row1 + 0, &acc0, &acc1);
            FmaBf16x16(lhs1, rhs_row1 + 16, &acc2, &acc3);
            FmaBf16x16(lhs1, rhs_row1 + 32, &acc4, &acc5);
            FmaBf16x16(lhs1, rhs_row1 + 48, &acc6, &acc7);
            FmaBf16x16(lhs2, rhs_row2 + 0, &acc0, &acc1);
            FmaBf16x16(lhs2, rhs_row2 + 16, &acc2, &acc3);
            FmaBf16x16(lhs2, rhs_row2 + 32, &acc4, &acc5);
            FmaBf16x16(lhs2, rhs_row2 + 48, &acc6, &acc7);
            FmaBf16x16(lhs3, rhs_row3 + 0, &acc0, &acc1);
            FmaBf16x16(lhs3, rhs_row3 + 16, &acc2, &acc3);
            FmaBf16x16(lhs3, rhs_row3 + 32, &acc4, &acc5);
            FmaBf16x16(lhs3, rhs_row3 + 48, &acc6, &acc7);
        }
        for (; t < k; ++t) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[t]);
            const uint16_t* rhs_row = rhs_block + t * kBf16PackedOutputBlock;
            FmaBf16x16(lhs_value, rhs_row + 0, &acc0, &acc1);
            FmaBf16x16(lhs_value, rhs_row + 16, &acc2, &acc3);
            FmaBf16x16(lhs_value, rhs_row + 32, &acc4, &acc5);
            FmaBf16x16(lhs_value, rhs_row + 48, &acc6, &acc7);
        }
        StoreBf16x8(acc0, out + col);
        StoreBf16x8(acc1, out + col + 8);
        StoreBf16x8(acc2, out + col + 16);
        StoreBf16x8(acc3, out + col + 24);
        StoreBf16x8(acc4, out + col + 32);
        StoreBf16x8(acc5, out + col + 40);
        StoreBf16x8(acc6, out + col + 48);
        StoreBf16x8(acc7, out + col + 56);
    }
    if (col < end) {
        ComputeSingleRowWideBf16PackedTransposedColumns(lhs, packed_rhs, nullptr, k, end, LinearBiasType::kNone,
                                                         1.0f, 0.0f, col, end, out);
    }
}

inline int32_t ComputeSingleRowWideBf16PackedTransposed(const float* lhs, const uint16_t* packed_rhs,
                                                        const uint16_t* bias, int64_t k, int64_t n,
                                                        LinearBiasType bias_type, float alpha, float beta,
                                                        uint16_t* out) {
    const size_t worker_count = Bf16LinearWorkerCount(1, k, n);
    ParallelForBf16OutputColumns(n, worker_count, [&](int64_t begin, int64_t end) {
        if (bias == nullptr && alpha == 1.0f) {
            ComputeSingleRowWideBf16PackedTransposedLogitsColumns(lhs, packed_rhs, k, begin, end, out);
            return;
        }
        ComputeSingleRowWideBf16PackedTransposedColumns(lhs, packed_rhs, bias, k, n, bias_type, alpha, beta, begin,
                                                         end, out);
    });
    return 0;
}

struct Bf16ArgmaxResult {
    int64_t index{-1};
    float value{0.0f};
    uint32_t key{0};
};

inline __m256i Bf16SortableKeys(__m256i rounded_bits) {
    const __m256i sign = _mm256_and_si256(rounded_bits, _mm256_set1_epi32(0x8000));
    const __m256i negative_mask = _mm256_cmpeq_epi32(sign, _mm256_set1_epi32(0x8000));
    const __m256i positive_keys = _mm256_xor_si256(rounded_bits, _mm256_set1_epi32(0x8000));
    const __m256i negative_keys = _mm256_xor_si256(rounded_bits, _mm256_set1_epi32(0xffff));
    __m256i keys = _mm256_blendv_epi8(positive_keys, negative_keys, negative_mask);

    // Match scalar `value > best_value`: +0 and -0 compare equal, so the
    // first zero must win regardless of its sign bit.
    const __m256i magnitude = _mm256_and_si256(rounded_bits, _mm256_set1_epi32(0x7fff));
    const __m256i zero_mask = _mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256());
    keys = _mm256_blendv_epi8(keys, _mm256_set1_epi32(0x8000), zero_mask);
    return keys;
}

inline void ConsiderRoundedBf16Vector(__m256 value, int64_t base_index, int64_t lane_limit,
                                      Bf16ArgmaxResult* result) {
    if (result == nullptr || lane_limit <= 0) {
        return;
    }
    const __m256i rounded_bits = RoundBf16Bits(value);
    const __m256i keys = Bf16SortableKeys(rounded_bits);
    alignas(32) uint32_t bits[8];
    alignas(32) uint32_t sortable_keys[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(bits), rounded_bits);
    _mm256_store_si256(reinterpret_cast<__m256i*>(sortable_keys), keys);
    const int valid_lane_count = std::min<int64_t>(lane_limit, 8);
    for (int lane = 0; lane < valid_lane_count; ++lane) {
        const uint16_t selected_bits = static_cast<uint16_t>(bits[lane]);
        if ((selected_bits & 0x7f80u) == 0x7f80u && (selected_bits & 0x007fu) != 0) {
            continue;
        }
        const uint32_t key = sortable_keys[lane];
        if (result->index < 0 || key > result->key) {
            result->index = base_index + lane;
            result->value = BFloat16ToFloat(selected_bits);
            result->key = key;
        }
    }
}

inline Bf16ArgmaxResult ComputeSingleRowWideBf16PackedTransposedArgmaxColumns(const float* lhs,
                                                                                  const uint16_t* packed_rhs,
                                                                                  int64_t k, int64_t n, int64_t begin,
                                                                                  int64_t end) {
    Bf16ArgmaxResult result;
    for (int64_t col = begin; col < end;) {
        const int64_t block = col / kBf16PackedOutputBlock;
        const int64_t block_begin = block * kBf16PackedOutputBlock;
        const int64_t block_end = std::min(end, std::min(n, block_begin + kBf16PackedOutputBlock));
        const uint16_t* rhs_block = packed_rhs + block * k * kBf16PackedOutputBlock;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        int64_t row = 0;
        for (; row + 4 <= k; row += 4) {
            const uint16_t* rhs_row0 = rhs_block + (row + 0) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row1 = rhs_block + (row + 1) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row2 = rhs_block + (row + 2) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row3 = rhs_block + (row + 3) * kBf16PackedOutputBlock;
            if (row + 8 < k) {
                __builtin_prefetch(rhs_row0 + 8 * kBf16PackedOutputBlock, 0, 1);
            }
            const __m256 lhs0 = _mm256_set1_ps(lhs[row + 0]);
            const __m256 lhs1 = _mm256_set1_ps(lhs[row + 1]);
            const __m256 lhs2 = _mm256_set1_ps(lhs[row + 2]);
            const __m256 lhs3 = _mm256_set1_ps(lhs[row + 3]);
            FmaBf16x16(lhs0, rhs_row0 + 0, &acc0, &acc1);
            FmaBf16x16(lhs0, rhs_row0 + 16, &acc2, &acc3);
            FmaBf16x16(lhs0, rhs_row0 + 32, &acc4, &acc5);
            FmaBf16x16(lhs0, rhs_row0 + 48, &acc6, &acc7);
            FmaBf16x16(lhs1, rhs_row1 + 0, &acc0, &acc1);
            FmaBf16x16(lhs1, rhs_row1 + 16, &acc2, &acc3);
            FmaBf16x16(lhs1, rhs_row1 + 32, &acc4, &acc5);
            FmaBf16x16(lhs1, rhs_row1 + 48, &acc6, &acc7);
            FmaBf16x16(lhs2, rhs_row2 + 0, &acc0, &acc1);
            FmaBf16x16(lhs2, rhs_row2 + 16, &acc2, &acc3);
            FmaBf16x16(lhs2, rhs_row2 + 32, &acc4, &acc5);
            FmaBf16x16(lhs2, rhs_row2 + 48, &acc6, &acc7);
            FmaBf16x16(lhs3, rhs_row3 + 0, &acc0, &acc1);
            FmaBf16x16(lhs3, rhs_row3 + 16, &acc2, &acc3);
            FmaBf16x16(lhs3, rhs_row3 + 32, &acc4, &acc5);
            FmaBf16x16(lhs3, rhs_row3 + 48, &acc6, &acc7);
        }
        for (; row < k; ++row) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[row]);
            const uint16_t* rhs_row = rhs_block + row * kBf16PackedOutputBlock;
            FmaBf16x16(lhs_value, rhs_row + 0, &acc0, &acc1);
            FmaBf16x16(lhs_value, rhs_row + 16, &acc2, &acc3);
            FmaBf16x16(lhs_value, rhs_row + 32, &acc4, &acc5);
            FmaBf16x16(lhs_value, rhs_row + 48, &acc6, &acc7);
        }
        const int64_t lane_limit = block_end - block_begin;
        ConsiderRoundedBf16Vector(acc0, block_begin + 0, lane_limit, &result);
        ConsiderRoundedBf16Vector(acc1, block_begin + 8, lane_limit - 8, &result);
        ConsiderRoundedBf16Vector(acc2, block_begin + 16, lane_limit - 16, &result);
        ConsiderRoundedBf16Vector(acc3, block_begin + 24, lane_limit - 24, &result);
        ConsiderRoundedBf16Vector(acc4, block_begin + 32, lane_limit - 32, &result);
        ConsiderRoundedBf16Vector(acc5, block_begin + 40, lane_limit - 40, &result);
        ConsiderRoundedBf16Vector(acc6, block_begin + 48, lane_limit - 48, &result);
        ConsiderRoundedBf16Vector(acc7, block_begin + 56, lane_limit - 56, &result);
        col = block_end;
    }
    return result;
}

// AVX2 exposes sixteen YMM registers. Keeping only four accumulators for a
// 32-column tile leaves enough registers for widened BF16 loads and the
// four-way K unroll, avoiding spills in the single-threaded lm-head path.
inline Bf16ArgmaxResult ComputeSingleRowWideBf16PackedTransposedArgmaxColumns32(
    const float* lhs, const uint16_t* packed_rhs, int64_t k, int64_t n, int64_t begin, int64_t end) {
    Bf16ArgmaxResult result;
    for (int64_t col = begin; col < end;) {
        const int64_t block_begin = (col / kBf16ArgmaxSingleThreadOutputBlock) *
                                    kBf16ArgmaxSingleThreadOutputBlock;
        const int64_t block_end = std::min(end, std::min(n, block_begin + kBf16ArgmaxSingleThreadOutputBlock));
        const int64_t packed_block = block_begin / kBf16PackedOutputBlock;
        const int64_t packed_offset = block_begin % kBf16PackedOutputBlock;
        const uint16_t* rhs_block =
            packed_rhs + packed_block * k * kBf16PackedOutputBlock + packed_offset;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int64_t row = 0;
        for (; row + 4 <= k; row += 4) {
            const uint16_t* rhs_row0 = rhs_block + (row + 0) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row1 = rhs_block + (row + 1) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row2 = rhs_block + (row + 2) * kBf16PackedOutputBlock;
            const uint16_t* rhs_row3 = rhs_block + (row + 3) * kBf16PackedOutputBlock;
            if (row + 8 < k) {
                __builtin_prefetch(rhs_row0 + 8 * kBf16PackedOutputBlock, 0, 1);
            }
            const __m256 lhs0 = _mm256_set1_ps(lhs[row + 0]);
            const __m256 lhs1 = _mm256_set1_ps(lhs[row + 1]);
            const __m256 lhs2 = _mm256_set1_ps(lhs[row + 2]);
            const __m256 lhs3 = _mm256_set1_ps(lhs[row + 3]);
            FmaBf16x16(lhs0, rhs_row0 + 0, &acc0, &acc1);
            FmaBf16x16(lhs0, rhs_row0 + 16, &acc2, &acc3);
            FmaBf16x16(lhs1, rhs_row1 + 0, &acc0, &acc1);
            FmaBf16x16(lhs1, rhs_row1 + 16, &acc2, &acc3);
            FmaBf16x16(lhs2, rhs_row2 + 0, &acc0, &acc1);
            FmaBf16x16(lhs2, rhs_row2 + 16, &acc2, &acc3);
            FmaBf16x16(lhs3, rhs_row3 + 0, &acc0, &acc1);
            FmaBf16x16(lhs3, rhs_row3 + 16, &acc2, &acc3);
        }
        for (; row < k; ++row) {
            const __m256 lhs_value = _mm256_set1_ps(lhs[row]);
            const uint16_t* rhs_row = rhs_block + row * kBf16PackedOutputBlock;
            FmaBf16x16(lhs_value, rhs_row + 0, &acc0, &acc1);
            FmaBf16x16(lhs_value, rhs_row + 16, &acc2, &acc3);
        }
        const int64_t lane_limit = block_end - block_begin;
        ConsiderRoundedBf16Vector(acc0, block_begin + 0, lane_limit, &result);
        ConsiderRoundedBf16Vector(acc1, block_begin + 8, lane_limit - 8, &result);
        ConsiderRoundedBf16Vector(acc2, block_begin + 16, lane_limit - 16, &result);
        ConsiderRoundedBf16Vector(acc3, block_begin + 24, lane_limit - 24, &result);
        col = block_end;
    }
    return result;
}

inline float HorizontalSum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline void ComputeTransposedRhsRowColumns(const float* lhs_row, const uint16_t* rhs_transposed,
                                           const uint16_t* bias, int64_t row, int64_t k, int64_t n,
                                           LinearBiasType bias_type, float alpha, float beta, int64_t begin,
                                           int64_t end, uint16_t* out_row) {
    for (int64_t col = begin; col < end; ++col) {
        const uint16_t* rhs_row = rhs_transposed + col * k;
        __m256 acc = _mm256_setzero_ps();
        int64_t t = 0;
        for (; t + 8 <= k; t += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(lhs_row + t), LoadBf16x8(rhs_row + t), acc);
        }
        float sum = alpha * HorizontalSum(acc) + beta * LoadBiasScalarBf16(bias, bias_type, row, n, col);
        for (; t < k; ++t) {
            sum += alpha * lhs_row[t] * BFloat16ToFloat(rhs_row[t]);
        }
        out_row[col] = FloatToBFloat16(sum);
    }
}

}  // namespace

bool PackedBf16Rhs::Matches(const uint16_t* rhs, int64_t k, int64_t n, uint64_t source_version) const {
    return rhs != nullptr && source_ == rhs && k_ == k && n_ == n && source_version_ == source_version &&
           !data_.empty();
}

bool PackedBf16Rhs::Pack(const uint16_t* rhs, int64_t k, int64_t n, uint64_t source_version) {
    if (Matches(rhs, k, n, source_version)) {
        return true;
    }
    source_ = nullptr;
    k_ = 0;
    n_ = 0;
    source_version_ = 0;
    data_.clear();
    // The packed kernel performs eight full-width vector loads per output block.
    // Keep the contract strict so a tail cannot read past the source bias/RHS.
    if (rhs == nullptr || k <= 0 || n <= 0 ||
        n > std::numeric_limits<int64_t>::max() - (kBf16PackedOutputBlock - 1)) {
        return false;
    }
    const int64_t block_count = (n + kBf16PackedOutputBlock - 1) / kBf16PackedOutputBlock;
    if (block_count > std::numeric_limits<int64_t>::max() / kBf16PackedOutputBlock ||
        k > std::numeric_limits<int64_t>::max() / (block_count * kBf16PackedOutputBlock)) {
        return false;
    }
    const int64_t element_count = block_count * k * kBf16PackedOutputBlock;
    if (static_cast<uint64_t>(element_count) > std::numeric_limits<size_t>::max()) {
        return false;
    }
    try {
        data_.assign(static_cast<size_t>(element_count), 0);
    } catch (const std::bad_alloc&) {
        data_.clear();
        return false;
    }
    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t col = block * kBf16PackedOutputBlock;
        const int64_t width = std::min(kBf16PackedOutputBlock, n - col);
        for (int64_t row = 0; row < k; ++row) {
            uint16_t* destination = data_.data() + (block * k + row) * kBf16PackedOutputBlock;
            for (int64_t lane = 0; lane < width; ++lane) {
                destination[lane] = rhs[row * n + col + lane];
            }
        }
    }
    source_ = rhs;
    k_ = k;
    n_ = n;
    source_version_ = source_version;
    return true;
}

bool PackedBf16TransposedRhs::Matches(const uint16_t* rhs_transposed, int64_t k, int64_t n,
                                      uint64_t source_version) const {
    return rhs_transposed != nullptr && source_ == rhs_transposed && k_ == k && n_ == n &&
           source_version_ == source_version && !data_.empty();
}

bool PackedBf16TransposedRhs::Pack(const uint16_t* rhs_transposed, int64_t k, int64_t n, uint64_t source_version) {
    if (Matches(rhs_transposed, k, n, source_version)) {
        return true;
    }
    source_ = nullptr;
    k_ = 0;
    n_ = 0;
    source_version_ = 0;
    data_.clear();
    if (rhs_transposed == nullptr || k <= 0 || n <= 0) {
        return false;
    }
    const int64_t block_count = (n + kBf16PackedOutputBlock - 1) / kBf16PackedOutputBlock;
    if (block_count > std::numeric_limits<int64_t>::max() / kBf16PackedOutputBlock ||
        k > std::numeric_limits<int64_t>::max() / (block_count * kBf16PackedOutputBlock)) {
        return false;
    }
    const int64_t element_count = block_count * k * kBf16PackedOutputBlock;
    if (static_cast<uint64_t>(element_count) > std::numeric_limits<size_t>::max()) {
        return false;
    }
    try {
        data_.resize(static_cast<size_t>(element_count));
    } catch (const std::bad_alloc&) {
        data_.clear();
        return false;
    }
    for (int64_t block = 0; block < block_count; ++block) {
        const uint16_t* source_block = rhs_transposed + block * kBf16PackedOutputBlock * k;
        uint16_t* destination_block = data_.data() + block * k * kBf16PackedOutputBlock;
        for (int64_t row = 0; row < k; ++row) {
            uint16_t* destination = destination_block + row * kBf16PackedOutputBlock;
            for (int64_t lane = 0; lane < kBf16PackedOutputBlock; ++lane) {
                if (block * kBf16PackedOutputBlock + lane < n) {
                    destination[lane] = source_block[lane * k + row];
                } else {
                    destination[lane] = 0;
                }
            }
        }
    }
    source_ = rhs_transposed;
    k_ = k;
    n_ = n;
    source_version_ = source_version;
    return true;
}

int32_t ComputeLinearRowMajorX86Bf16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t m,
                                     int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out,
                                     Bf16LinearWorkspace* workspace) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    if (m == 1 && n >= 32) {
        return ComputeSingleRowWideBf16(lhs, rhs, bias, k, n, bias_type, out, workspace);
    }

    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(m * k)) || !workspace->ResizePanel(static_cast<size_t>(k * 16))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, m * k, workspace->data());

    int64_t col = 0;
    for (; col + 16 <= n; col += 16) {
        ConvertRhsPanelToFp32(rhs, k, n, col, 16, workspace->panel_data());
        const bool vector_bias = bias != nullptr && bias_type == LinearBiasType::kVector;
        const __m256 vector_bias0 = vector_bias ? LoadBiasVectorBf16(bias, bias_type, 0, n, col) : _mm256_setzero_ps();
        const __m256 vector_bias1 =
            vector_bias ? LoadBiasVectorBf16(bias, bias_type, 0, n, col + 8) : _mm256_setzero_ps();
        for (int64_t row = 0; row < m; ++row) {
            const float* lhs_row = workspace->data() + row * k;
            uint16_t* out_row = out + row * n;
            __m256 acc0 = vector_bias ? vector_bias0 : LoadBiasVectorBf16(bias, bias_type, row, n, col);
            __m256 acc1 = vector_bias ? vector_bias1 : LoadBiasVectorBf16(bias, bias_type, row, n, col + 8);
            for (int64_t t = 0; t < k; ++t) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[t]);
                const float* rhs_ptr = workspace->panel_data() + t * 16;
                acc0 = _mm256_fmadd_ps(lhs_value, _mm256_loadu_ps(rhs_ptr), acc0);
                acc1 = _mm256_fmadd_ps(lhs_value, _mm256_loadu_ps(rhs_ptr + 8), acc1);
            }
            alignas(32) float values0[8];
            alignas(32) float values1[8];
            _mm256_store_ps(values0, acc0);
            _mm256_store_ps(values1, acc1);
            for (int64_t lane = 0; lane < 8; ++lane) {
                out_row[col + lane] = FloatToBFloat16(values0[lane]);
                out_row[col + 8 + lane] = FloatToBFloat16(values1[lane]);
            }
        }
    }

    for (; col + 8 <= n; col += 8) {
        ConvertRhsPanelToFp32(rhs, k, n, col, 8, workspace->panel_data());
        const bool vector_bias = bias != nullptr && bias_type == LinearBiasType::kVector;
        const __m256 vector_bias_value =
            vector_bias ? LoadBiasVectorBf16(bias, bias_type, 0, n, col) : _mm256_setzero_ps();
        for (int64_t row = 0; row < m; ++row) {
            const float* lhs_row = workspace->data() + row * k;
            __m256 acc = vector_bias ? vector_bias_value : LoadBiasVectorBf16(bias, bias_type, row, n, col);
            for (int64_t t = 0; t < k; ++t) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(lhs_row[t]), _mm256_loadu_ps(workspace->panel_data() + t * 8), acc);
            }
            alignas(32) float values[8];
            _mm256_store_ps(values, acc);
            uint16_t* out_row = out + row * n + col;
            for (int64_t lane = 0; lane < 8; ++lane) {
                out_row[lane] = FloatToBFloat16(values[lane]);
            }
        }
    }

    for (int64_t row = 0; row < m; ++row) {
        const float* lhs_row = workspace->data() + row * k;
        uint16_t* out_row = out + row * n;
        for (int64_t tail_col = col; tail_col < n; ++tail_col) {
            float sum = LoadBiasScalarBf16(bias, bias_type, row, n, tail_col);
            for (int64_t t = 0; t < k; ++t) {
                sum += lhs_row[t] * BFloat16ToFloat(rhs[t * n + tail_col]);
            }
            out_row[tail_col] = FloatToBFloat16(sum);
        }
    }
    return 0;
}

int32_t ComputeLinearRowMajorX86Bf16PackedRhs(const uint16_t* lhs, const uint16_t* rhs,
                                              const PackedBf16Rhs& packed_rhs, const uint16_t* bias, int64_t m,
                                              int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out,
                                              Bf16LinearWorkspace* workspace, uint64_t source_version) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m != 1 || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs, k, n, source_version)) {
        return -1;
    }
    if (Bf16LinearWorkerCount(m, k, n) <= 1) {
        return ComputeLinearRowMajorX86Bf16PackedRhsSingleThread(lhs, rhs, packed_rhs, bias, m, k, n, bias_type,
                                                                  out, workspace, source_version);
    }
    return ComputeSingleRowWideBf16Packed(lhs, packed_rhs.data(), bias, k, n, bias_type, out, workspace);
}

int32_t ComputeLinearRowMajorX86Bf16PackedRhsSingleThread(
    const uint16_t* lhs, const uint16_t* rhs, const PackedBf16Rhs& packed_rhs, const uint16_t* bias, int64_t m,
    int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out, Bf16LinearWorkspace* workspace,
    uint64_t source_version) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m != 1 || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs, k, n, source_version)) {
        return -1;
    }

    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    ComputeSingleRowWideBf16PackedColumns(workspace->data(), packed_rhs.data(), bias, k, n, bias_type, 0, n, out);
    return 0;
}

int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhs(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs,
    const uint16_t* bias, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type, float alpha, float beta,
    uint16_t* out, Bf16LinearWorkspace* workspace, uint64_t source_version) {
    if (lhs == nullptr || rhs_transposed == nullptr || out == nullptr || m != 1 || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs_transposed, k, n, source_version)) {
        return -1;
    }
    if (Bf16LinearWorkerCount(m, k, n) <= 1) {
        return ComputeLinearRowMajorX86Bf16PackedTransposedRhsSingleThread(
            lhs, rhs_transposed, packed_rhs, bias, m, k, n, bias_type, alpha, beta, out, workspace, source_version);
    }
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) workspace = &local_workspace;
    if (!workspace->Resize(static_cast<size_t>(k))) return -1;
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    return ComputeSingleRowWideBf16PackedTransposed(workspace->data(), packed_rhs.data(), bias, k, n, bias_type,
                                                     alpha, beta, out);
}

int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsSingleThread(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs,
    const uint16_t* bias, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type, float alpha, float beta,
    uint16_t* out, Bf16LinearWorkspace* workspace, uint64_t source_version) {
    if (lhs == nullptr || rhs_transposed == nullptr || out == nullptr || m != 1 || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs_transposed, k, n, source_version)) {
        return -1;
    }
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    if (bias == nullptr && alpha == 1.0f) {
        ComputeSingleRowWideBf16PackedTransposedLogitsColumns(workspace->data(), packed_rhs.data(), k, 0, n, out);
    } else {
        ComputeSingleRowWideBf16PackedTransposedColumns(workspace->data(), packed_rhs.data(), bias, k, n, bias_type,
                                                         alpha, beta, 0, n, out);
    }
    return 0;
}

int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmax(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs, int64_t k,
    int64_t n, int64_t* token, Bf16LinearWorkspace* workspace, uint64_t source_version) {
    if (lhs == nullptr || rhs_transposed == nullptr || token == nullptr || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs_transposed, k, n, source_version)) {
        return -1;
    }
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());

    const size_t worker_count = Bf16LinearWorkerCount(1, k, n);
    if (worker_count <= 1) {
        return ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmaxSingleThread(
            lhs, rhs_transposed, packed_rhs, k, n, token, workspace, source_version);
    }

    const int64_t block_count = (n + kBf16PackedOutputBlock - 1) / kBf16PackedOutputBlock;
    std::vector<Bf16ArgmaxResult> partial(static_cast<size_t>(block_count));
    ParallelForBf16OutputColumns(n, worker_count, [&](int64_t begin, int64_t end) {
        for (int64_t block_begin = begin; block_begin < end; block_begin += kBf16PackedOutputBlock) {
            const int64_t block_end = std::min(end, std::min(n, block_begin + kBf16PackedOutputBlock));
            partial[static_cast<size_t>(block_begin / kBf16PackedOutputBlock)] =
                ComputeSingleRowWideBf16PackedTransposedArgmaxColumns(workspace->data(), packed_rhs.data(), k, n,
                                                                       block_begin, block_end);
        }
    });

    Bf16ArgmaxResult best;
    for (const auto& candidate : partial) {
        if (candidate.index >= 0 && (best.index < 0 || candidate.value > best.value)) {
            best = candidate;
        }
    }
    *token = best.index < 0 ? 0 : best.index;
    return 0;
}

int32_t ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmaxSingleThread(
    const uint16_t* lhs, const uint16_t* rhs_transposed, const PackedBf16TransposedRhs& packed_rhs, int64_t k,
    int64_t n, int64_t* token, Bf16LinearWorkspace* workspace, uint64_t source_version) {
    if (lhs == nullptr || rhs_transposed == nullptr || token == nullptr || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(rhs_transposed, k, n, source_version)) {
        return -1;
    }
    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    ConvertBf16ArrayToFp32(lhs, k, workspace->data());
    const Bf16ArgmaxResult result = ComputeSingleRowWideBf16PackedTransposedArgmaxColumns32(
        workspace->data(), packed_rhs.data(), k, n, 0, n);
    *token = result.index < 0 ? 0 : result.index;
    return 0;
}

int32_t ComputeLinearRowMajorX86Bf16TransposedRhs(const uint16_t* lhs, const uint16_t* rhs_transposed,
                                                   const uint16_t* bias, int64_t m, int64_t k, int64_t n,
                                                   LinearBiasType bias_type, float alpha, float beta, uint16_t* out,
                                                   Bf16LinearWorkspace* workspace) {
    if (lhs == nullptr || rhs_transposed == nullptr || out == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    Bf16LinearWorkspace local_workspace;
    if (workspace == nullptr) workspace = &local_workspace;
    if (!workspace->Resize(static_cast<size_t>(m * k))) return -1;
    ConvertBf16ArrayToFp32(lhs, m * k, workspace->data());

    if (m == 1) {
        const size_t worker_count = Bf16LinearWorkerCount(m, k, n);
        ParallelForBf16OutputColumns(n, worker_count, [&](int64_t begin, int64_t end) {
            ComputeTransposedRhsRowColumns(workspace->data(), rhs_transposed, bias, 0, k, n, bias_type, alpha, beta,
                                           begin, end, out);
        });
        return 0;
    }

    for (int64_t row = 0; row < m; ++row) {
        const float* lhs_row = workspace->data() + row * k;
        uint16_t* out_row = out + row * n;
        ComputeTransposedRhsRowColumns(lhs_row, rhs_transposed, bias, row, k, n, bias_type, alpha, beta, 0, n,
                                       out_row);
    }
    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
