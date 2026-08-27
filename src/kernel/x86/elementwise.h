#ifndef FEATHER_KERNEL_X86_ELEMENTWISE_H
#define FEATHER_KERNEL_X86_ELEMENTWISE_H

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "src/operator/params.h"
#include "util/bf16.h"
#include "util/threading.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace x86 {
namespace elementwise_detail {

enum class BinaryOperation {
    kAdd,
    kSub,
    kMul,
    kDiv,
};

inline __m256 LoadBf16x8(const uint16_t* input) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    const __m256i expanded = _mm256_cvtepu16_epi32(packed);
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
}

inline void StoreBf16x8(__m256 value, uint16_t* output) {
    const __m256i bits = _mm256_castps_si256(value);
    const __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff));
    const __m256i exponent_is_inf_or_nan = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7f800000));
    const __m256i mantissa_is_zero = _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256());
    const __m256i nan_mask = _mm256_andnot_si256(mantissa_is_zero, exponent_is_inf_or_nan);
    const __m256i round_bias =
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    __m256i high_bits = _mm256_srli_epi32(_mm256_add_epi32(bits, round_bias), 16);
    high_bits = _mm256_or_si256(high_bits, _mm256_and_si256(nan_mask, _mm256_set1_epi32(0x0040)));
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(high_bits),
                                             _mm256_extracti128_si256(high_bits, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), packed);
}

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<size_t>(axis)] =
            strides[static_cast<size_t>(axis + 1)] * dims[static_cast<size_t>(axis + 1)];
    }
    return strides;
}

inline bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                                std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t lhs_dim =
            axis < out_rank - lhs_dims.size() ? 1 : lhs_dims[axis - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim =
            axis < out_rank - rhs_dims.size() ? 1 : rhs_dims[axis - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[axis] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

template <BinaryOperation operation>
inline __m256 Apply(__m256 lhs, __m256 rhs) {
    if constexpr (operation == BinaryOperation::kAdd) {
        return _mm256_add_ps(lhs, rhs);
    }
    if constexpr (operation == BinaryOperation::kSub) {
        return _mm256_sub_ps(lhs, rhs);
    }
    if constexpr (operation == BinaryOperation::kDiv) {
        return _mm256_div_ps(lhs, rhs);
    }
    return _mm256_mul_ps(lhs, rhs);
}

template <BinaryOperation operation>
inline float Apply(float lhs, float rhs) {
    if constexpr (operation == BinaryOperation::kAdd) {
        return lhs + rhs;
    }
    if constexpr (operation == BinaryOperation::kSub) {
        return lhs - rhs;
    }
    if constexpr (operation == BinaryOperation::kDiv) {
        return lhs / rhs;
    }
    return lhs * rhs;
}

inline size_t Fp32ElementwiseWorkerCount(int64_t rows, int64_t inner) {
    constexpr int64_t kParallelWorkItems = 1 << 18;
    if (rows < 2 || inner <= 0 || rows > std::numeric_limits<int64_t>::max() / inner ||
        rows * inner < kParallelWorkItems) {
        return 1;
    }

#if defined(FEATHER_WITH_OPENMP)
    if (omp_in_parallel()) {
        return 1;
    }
    size_t worker_count = std::min(feather::DefaultThreadCount(), static_cast<size_t>(rows));
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) {
        worker_count = std::min(worker_count, static_cast<size_t>(openmp_limit));
    }
    return std::max<size_t>(1, worker_count);
#else
    return 1;
#endif
}

template <typename Fn>
inline void ParallelForFp32Rows(int64_t rows, int64_t inner, Fn&& fn) {
    const size_t worker_count = Fp32ElementwiseWorkerCount(rows, inner);
#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int64_t row = 0; row < rows; ++row) {
            fn(row, row + 1);
        }
        return;
    }
#endif
    fn(0, rows);
}

template <BinaryOperation operation>
bool TryComputeScalarBroadcastFp32(operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        !param->lhs->IsInitialized() || !param->rhs->IsInitialized() || !param->out->IsInitialized() ||
        param->lhs->data_type() != DataType::FP32 || param->rhs->data_type() != DataType::FP32 ||
        param->out->data_type() != DataType::FP32) {
        return false;
    }

    const int64_t output_numel = param->out->numel();
    if (output_numel <= 0) {
        return false;
    }
    const bool lhs_scalar = param->lhs->numel() == 1 && param->rhs->numel() == output_numel;
    const bool rhs_scalar = param->rhs->numel() == 1 && param->lhs->numel() == output_numel;
    if ((!lhs_scalar && !rhs_scalar) ||
        (lhs_scalar ? param->rhs->dims().data() : param->lhs->dims().data()) != param->out->dims().data()) {
        return false;
    }

    const float* lhs = param->lhs->data<float>();
    const float* rhs = param->rhs->data<float>();
    float* out = param->out->mutable_data<float>();
    const float lhs_scalar_value = lhs[0];
    const float rhs_scalar_value = rhs[0];
    int64_t index = 0;
    for (; index + 8 <= output_numel; index += 8) {
        const __m256 lhs_values = lhs_scalar ? _mm256_set1_ps(lhs_scalar_value) : _mm256_loadu_ps(lhs + index);
        const __m256 rhs_values = rhs_scalar ? _mm256_set1_ps(rhs_scalar_value) : _mm256_loadu_ps(rhs + index);
        _mm256_storeu_ps(out + index, Apply<operation>(lhs_values, rhs_values));
    }
    for (; index < output_numel; ++index) {
        out[index] = Apply<operation>(lhs_scalar ? lhs_scalar_value : lhs[index],
                                      rhs_scalar ? rhs_scalar_value : rhs[index]);
    }
    return true;
}

template <BinaryOperation operation>
bool TryComputeLastDimensionBroadcastFp32(operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        !param->lhs->IsInitialized() || !param->rhs->IsInitialized() || !param->out->IsInitialized() ||
        param->lhs->data_type() != DataType::FP32 || param->rhs->data_type() != DataType::FP32) {
        return false;
    }

    std::vector<int64_t> out_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &out_dims) || out_dims.empty() ||
        param->out->dims().data() != out_dims) {
        return false;
    }

    const int64_t inner = out_dims.back();
    if (inner <= 0 || param->out->numel() <= 0 || param->out->numel() % inner != 0 ||
        param->lhs->dims().empty() || param->rhs->dims().empty()) {
        return false;
    }
    const int64_t lhs_inner = param->lhs->dims()[param->lhs->dims().size() - 1];
    const int64_t rhs_inner = param->rhs->dims()[param->rhs->dims().size() - 1];
    if ((lhs_inner != 1 && lhs_inner != inner) || (rhs_inner != 1 && rhs_inner != inner)) {
        return false;
    }

    const size_t out_rank = out_dims.size();
    const size_t lhs_gap = out_rank - param->lhs->dims().size();
    const size_t rhs_gap = out_rank - param->rhs->dims().size();
    const std::vector<int64_t> outer_dims(out_dims.begin(), out_dims.end() - 1);
    const auto outer_strides = ComputeStrides(outer_dims);
    const auto lhs_strides = ComputeStrides(param->lhs->dims().data());
    const auto rhs_strides = ComputeStrides(param->rhs->dims().data());
    const int64_t rows = param->out->numel() / inner;

    param->out->set_data_type(DataType::FP32);
    const float* lhs = param->lhs->data<float>();
    const float* rhs = param->rhs->data<float>();
    float* out = param->out->mutable_data<float>();
    ParallelForFp32Rows(rows, inner, [&](int64_t begin_row, int64_t end_row) {
        for (int64_t row = begin_row; row < end_row; ++row) {
            int64_t remaining = row;
            int64_t lhs_offset = 0;
            int64_t rhs_offset = 0;
            for (size_t axis = 0; axis < out_rank - 1; ++axis) {
                const int64_t coordinate = remaining / outer_strides[axis];
                remaining %= outer_strides[axis];
                if (axis >= lhs_gap && param->lhs->dims()[axis - lhs_gap] != 1) {
                    lhs_offset += coordinate * lhs_strides[axis - lhs_gap];
                }
                if (axis >= rhs_gap && param->rhs->dims()[axis - rhs_gap] != 1) {
                    rhs_offset += coordinate * rhs_strides[axis - rhs_gap];
                }
            }

            const float* lhs_row = lhs + lhs_offset;
            const float* rhs_row = rhs + rhs_offset;
            float* out_row = out + row * inner;
            int64_t index = 0;
            if (lhs_inner == inner && rhs_inner == inner) {
                for (; index + 8 <= inner; index += 8) {
                    _mm256_storeu_ps(
                        out_row + index,
                        Apply<operation>(_mm256_loadu_ps(lhs_row + index), _mm256_loadu_ps(rhs_row + index)));
                }
            } else if (lhs_inner == 1 && rhs_inner == inner) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[0]);
                for (; index + 8 <= inner; index += 8) {
                    _mm256_storeu_ps(out_row + index, Apply<operation>(lhs_value, _mm256_loadu_ps(rhs_row + index)));
                }
            } else if (lhs_inner == inner && rhs_inner == 1) {
                const __m256 rhs_value = _mm256_set1_ps(rhs_row[0]);
                for (; index + 8 <= inner; index += 8) {
                    _mm256_storeu_ps(out_row + index, Apply<operation>(_mm256_loadu_ps(lhs_row + index), rhs_value));
                }
            } else {
                const __m256 value = _mm256_set1_ps(Apply<operation>(lhs_row[0], rhs_row[0]));
                for (; index + 8 <= inner; index += 8) {
                    _mm256_storeu_ps(out_row + index, value);
                }
            }
            for (; index < inner; ++index) {
                out_row[index] = Apply<operation>(lhs_row[lhs_inner == 1 ? 0 : index],
                                                  rhs_row[rhs_inner == 1 ? 0 : index]);
            }
        }
    });
    return true;
}

template <BinaryOperation operation>
bool TryComputeLastDimensionBroadcast(operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        !param->lhs->IsInitialized() || !param->rhs->IsInitialized() || !param->out->IsInitialized() ||
        param->lhs->data_type() != DataType::BF16 || param->rhs->data_type() != DataType::BF16) {
        return false;
    }

    std::vector<int64_t> out_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &out_dims) || out_dims.empty() ||
        param->out->dims().data() != out_dims) {
        return false;
    }
    const int64_t inner = out_dims.back();
    if (inner <= 0 || param->out->numel() <= 0 || param->out->numel() % inner != 0 ||
        param->lhs->dims().empty() || param->rhs->dims().empty()) {
        return false;
    }
    const int64_t lhs_inner = param->lhs->dims()[param->lhs->dims().size() - 1];
    const int64_t rhs_inner = param->rhs->dims()[param->rhs->dims().size() - 1];
    if ((lhs_inner != 1 && lhs_inner != inner) || (rhs_inner != 1 && rhs_inner != inner)) {
        return false;
    }

    const size_t out_rank = out_dims.size();
    const size_t lhs_rank = param->lhs->dims().size();
    const size_t rhs_rank = param->rhs->dims().size();
    const size_t lhs_gap = out_rank - lhs_rank;
    const size_t rhs_gap = out_rank - rhs_rank;
    const std::vector<int64_t> outer_dims(out_dims.begin(), out_dims.end() - 1);
    const auto outer_strides = ComputeStrides(outer_dims);
    const auto lhs_strides = ComputeStrides(param->lhs->dims().data());
    const auto rhs_strides = ComputeStrides(param->rhs->dims().data());
    const int64_t rows = param->out->numel() / inner;

    param->out->set_data_type(DataType::BF16);
    const auto* lhs = static_cast<const uint16_t*>(param->lhs->raw_data());
    const auto* rhs = static_cast<const uint16_t*>(param->rhs->raw_data());
    auto* out = static_cast<uint16_t*>(param->out->raw_data());
    for (int64_t row = 0; row < rows; ++row) {
        int64_t remaining = row;
        int64_t lhs_offset = 0;
        int64_t rhs_offset = 0;
        for (size_t axis = 0; axis + 1 < out_rank; ++axis) {
            const int64_t coordinate = remaining / outer_strides[axis];
            remaining %= outer_strides[axis];
            if (axis >= lhs_gap && param->lhs->dims()[axis - lhs_gap] != 1) {
                lhs_offset += coordinate * lhs_strides[axis - lhs_gap];
            }
            if (axis >= rhs_gap && param->rhs->dims()[axis - rhs_gap] != 1) {
                rhs_offset += coordinate * rhs_strides[axis - rhs_gap];
            }
        }

        const uint16_t* lhs_row = lhs + lhs_offset;
        const uint16_t* rhs_row = rhs + rhs_offset;
        uint16_t* out_row = out + row * inner;
        int64_t index = 0;
        if (lhs_inner == inner && rhs_inner == inner) {
            for (; index + 8 <= inner; index += 8) {
                StoreBf16x8(Apply<operation>(LoadBf16x8(lhs_row + index), LoadBf16x8(rhs_row + index)),
                             out_row + index);
            }
        } else if (lhs_inner == 1 && rhs_inner == inner) {
            const __m256 lhs_value = _mm256_set1_ps(BFloat16ToFloat(lhs_row[0]));
            for (; index + 8 <= inner; index += 8) {
                StoreBf16x8(Apply<operation>(lhs_value, LoadBf16x8(rhs_row + index)), out_row + index);
            }
        } else if (lhs_inner == inner && rhs_inner == 1) {
            const __m256 rhs_value = _mm256_set1_ps(BFloat16ToFloat(rhs_row[0]));
            for (; index + 8 <= inner; index += 8) {
                StoreBf16x8(Apply<operation>(LoadBf16x8(lhs_row + index), rhs_value), out_row + index);
            }
        } else {
            const __m256 value =
                _mm256_set1_ps(Apply<operation>(BFloat16ToFloat(lhs_row[0]), BFloat16ToFloat(rhs_row[0])));
            for (; index + 8 <= inner; index += 8) {
                StoreBf16x8(value, out_row + index);
            }
        }
        for (; index < inner; ++index) {
            const float lhs_value = BFloat16ToFloat(lhs_row[lhs_inner == 1 ? 0 : index]);
            const float rhs_value = BFloat16ToFloat(rhs_row[rhs_inner == 1 ? 0 : index]);
            out_row[index] = FloatToBFloat16(Apply<operation>(lhs_value, rhs_value));
        }
    }
    return true;
}

}  // namespace elementwise_detail
}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_ELEMENTWISE_H
