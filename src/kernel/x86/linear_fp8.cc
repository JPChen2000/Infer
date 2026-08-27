#include "src/kernel/x86/linear_fp8.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#include "util/fp8.h"
#include "util/bf16.h"
#include "util/threading.h"
#include "src/kernel/x86/fp8_utils.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace x86 {

namespace {

constexpr int64_t kFp8OutputBlock = 64;
constexpr int64_t kFp8MinimumParallelMacs = 1 << 18;
constexpr size_t kFp8DefaultWorkerLimit = 4;

size_t Fp8WorkerCount(int64_t m, int64_t k, int64_t n) {
    if (k <= 0 || n < kFp8OutputBlock || m <= 0 || m > std::numeric_limits<int64_t>::max() / k ||
        m * k > std::numeric_limits<int64_t>::max() / n || m * k * n < kFp8MinimumParallelMacs) {
        return 1;
    }
#if defined(FEATHER_WITH_OPENMP)
    size_t limit = std::min(DefaultThreadCount(), kFp8DefaultWorkerLimit);
    const char* configured = std::getenv("FEATHER_X86_FP8_THREADS");
    if (configured != nullptr && configured[0] != '\0') {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed > 0) {
            limit = std::min(limit, static_cast<size_t>(parsed));
        }
    }
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) limit = std::min(limit, static_cast<size_t>(openmp_limit));
    return std::max<size_t>(1, limit);
#else
    return 1;
#endif
}

inline bool IsValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

inline bool IsNaNBf16(uint16_t bits) {
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0;
}

template <DataType dtype>
const float* DecodeTable() {
    static const std::array<float, 256> table = []() {
        std::array<float, 256> values{};
        for (size_t index = 0; index < values.size(); ++index) {
            if constexpr (dtype == DataType::FP8E4M3) {
                values[index] = Fp8E4M3ToFloat(static_cast<uint8_t>(index));
            } else {
                values[index] = Fp8E5M2ToFloat(static_cast<uint8_t>(index));
            }
        }
        return values;
    }();
    return table.data();
}

inline __m256 LoadFp8x8(const uint8_t* input, const float* decode_table) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    const __m256i indices = _mm256_cvtepu8_epi32(bytes);
    return _mm256_i32gather_ps(decode_table, indices, 4);
}

inline __m256 LoadBf16x8(const uint16_t* input) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    const __m256i expanded = _mm256_cvtepu16_epi32(packed);
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
}

// FP8 has only a five-bit exponent and a three/two-bit mantissa. Reconstruct
// normal values directly in IEEE-754 bit fields instead of issuing an AVX2
// gather for every eight values. Subnormals are the only cases that need a
// floating-point multiply; the special E4M3 NaN and E5M2 Inf/NaN encodings are
// patched after the common normal path.
template <DataType dtype>
inline __m256 LoadFp8x8Typed(const uint8_t* input) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    const __m256i codes = _mm256_cvtepu8_epi32(bytes);
    const __m256i sign_codes = _mm256_and_si256(codes, _mm256_set1_epi32(0x80));
    const __m256i sign_bits = _mm256_slli_epi32(sign_codes, 24);
    const __m256 sign = _mm256_castsi256_ps(sign_bits);
    const __m256i mantissa_mask = _mm256_set1_epi32(dtype == DataType::FP8E4M3 ? 0x07 : 0x03);
    const __m256i mantissa = _mm256_and_si256(codes, mantissa_mask);
    const __m256i exponent = _mm256_and_si256(
        _mm256_srli_epi32(codes, dtype == DataType::FP8E4M3 ? 3 : 2),
        _mm256_set1_epi32(dtype == DataType::FP8E4M3 ? 0x0f : 0x1f));
    const __m256i normal_exponent = _mm256_slli_epi32(
        _mm256_add_epi32(exponent, _mm256_set1_epi32(dtype == DataType::FP8E4M3 ? 120 : 112)), 23);
    const __m256i normal_mantissa = _mm256_slli_epi32(mantissa, dtype == DataType::FP8E4M3 ? 20 : 21);
    const __m256 normal = _mm256_castsi256_ps(_mm256_or_si256(sign_bits, _mm256_or_si256(normal_exponent,
                                                                                           normal_mantissa)));

    const __m256 subnormal = _mm256_xor_ps(
        _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa),
                      _mm256_set1_ps(dtype == DataType::FP8E4M3 ? 0.001953125f : 0.0000152587890625f)),
        sign);
    const __m256i exponent_zero = _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256());
    __m256 result = _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(exponent_zero));

    if constexpr (dtype == DataType::FP8E4M3) {
        const __m256i nan_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x0f)),
            _mm256_cmpeq_epi32(mantissa, _mm256_set1_epi32(0x07)));
        result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                                  _mm256_castsi256_ps(nan_mask));
    } else {
        const __m256i infinity_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x1f)),
            _mm256_cmpeq_epi32(mantissa, _mm256_setzero_si256()));
        const __m256i special_nan_mask = _mm256_and_si256(
            _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x1f)),
            _mm256_cmpgt_epi32(mantissa, _mm256_setzero_si256()));
        result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_or_si256(
                                      sign_bits, _mm256_set1_epi32(0x7f800000))),
                                  _mm256_castsi256_ps(infinity_mask));
        result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                                  _mm256_castsi256_ps(special_nan_mask));
    }
    return result;
}

inline float HorizontalSum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    const __m128 pair = _mm_add_ps(low, high);
    const __m128 shuffled = _mm_movehdup_ps(pair);
    const __m128 sums = _mm_add_ps(pair, shuffled);
    const __m128 high_sum = _mm_movehl_ps(shuffled, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, high_sum));
}

inline bool ValidateCommonArguments(DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs,
                                    float rhs_scale, const uint8_t* bias, float bias_scale, int64_t m, int64_t k,
                                    int64_t n, LinearBiasType bias_type, uint8_t* out, float out_scale) {
    if ((dtype != DataType::FP8E4M3 && dtype != DataType::FP8E5M2) || lhs == nullptr || rhs == nullptr ||
        out == nullptr || m <= 0 || k <= 0 || n <= 0 || !IsValidScale(lhs_scale) || !IsValidScale(rhs_scale) ||
        !IsValidScale(out_scale)) {
        return false;
    }
    if (bias_type == LinearBiasType::kNone) {
        return bias == nullptr || IsValidScale(bias_scale);
    }
    return bias != nullptr && IsValidScale(bias_scale);
}

template <DataType dtype>
inline float DecodeScalar(const uint8_t* input, const float* decode_table) {
    return decode_table[*input];
}

inline uint16_t DecodeFp8ToBf16(DataType dtype, uint8_t code) {
    const float value = dtype == DataType::FP8E4M3 ? Fp8E4M3ToFloat(code) : Fp8E5M2ToFloat(code);
    return FloatToBFloat16(value);
}

template <DataType dtype>
bool DecodeInputRow(const uint8_t* input, int64_t k, float scale, std::vector<float>* decoded) {
    if (input == nullptr || decoded == nullptr || k <= 0 || !IsValidScale(scale)) {
        return false;
    }
    try {
        decoded->resize(static_cast<size_t>(k));
    } catch (...) {
        return false;
    }
    const float* table = DecodeTable<dtype>();
    const __m256 scale_vector = _mm256_set1_ps(scale);
    int64_t index = 0;
    for (; index + 8 <= k; index += 8) {
        _mm256_storeu_ps(decoded->data() + index,
                         _mm256_mul_ps(LoadFp8x8Typed<dtype>(input + index), scale_vector));
    }
    for (; index < k; ++index) {
        (*decoded)[static_cast<size_t>(index)] = DecodeScalar<dtype>(input + index, table) * scale;
    }
    return true;
}

template <DataType dtype>
void ComputeRowMajorFullBlock(const float* lhs, const uint8_t* rhs, const uint8_t* bias, float rhs_scale,
                              float bias_scale, float out_scale, float alpha, float beta, int64_t row, int64_t k,
                              int64_t n, LinearBiasType bias_type, int64_t col, uint8_t* out) {
    __m256 accumulators[kFp8OutputBlock / 8];
    for (auto& accumulator : accumulators) {
        accumulator = _mm256_setzero_ps();
    }
    if (bias != nullptr && bias_type != LinearBiasType::kNone) {
        const uint8_t* bias_row = bias_type == LinearBiasType::kVector ? bias : bias + row * n;
        const __m256 bias_scale_vector = _mm256_set1_ps(bias_scale);
        for (int64_t block = 0; block < kFp8OutputBlock / 8; ++block) {
            accumulators[block] = _mm256_mul_ps(
                _mm256_set1_ps(beta),
                _mm256_mul_ps(LoadFp8x8Typed<dtype>(bias_row + col + block * 8), bias_scale_vector));
        }
    }

    for (int64_t inner = 0; inner < k; ++inner) {
        const __m256 lhs_value = _mm256_set1_ps(lhs[inner]);
        const uint8_t* rhs_row = rhs + inner * n + col;
        for (int64_t block = 0; block < kFp8OutputBlock / 8; ++block) {
            accumulators[block] = _mm256_fmadd_ps(
                lhs_value, LoadFp8x8Typed<dtype>(rhs_row + block * 8), accumulators[block]);
        }
    }

    alignas(32) float values[8];
    const __m256 rhs_alpha_scale = _mm256_set1_ps(rhs_scale * alpha);
    const __m256 output_scale = _mm256_set1_ps(out_scale);
    for (int64_t block = 0; block < kFp8OutputBlock / 8; ++block) {
        _mm256_store_ps(values, _mm256_div_ps(_mm256_mul_ps(accumulators[block], rhs_alpha_scale), output_scale));
        EncodeFp8x8ForX86(dtype, values, out + row * n + col + block * 8);
    }
}

template <DataType dtype>
void ComputeRowMajorTail(const float* lhs, const uint8_t* rhs, const uint8_t* bias, float rhs_scale,
                         float bias_scale, float out_scale, float alpha, float beta, int64_t row, int64_t k,
                         int64_t n, LinearBiasType bias_type, int64_t begin_col, uint8_t* out) {
    const uint8_t* bias_row = nullptr;
    if (bias != nullptr && bias_type != LinearBiasType::kNone) {
        bias_row = bias_type == LinearBiasType::kVector ? bias : bias + row * n;
    }
    for (int64_t col = begin_col; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t inner = 0; inner < k; ++inner) {
            sum += lhs[inner] * DecodeScalar<dtype>(rhs + inner * n + col, DecodeTable<dtype>());
        }
        sum *= rhs_scale * alpha;
        if (bias_row != nullptr) {
            sum += beta * DecodeScalar<dtype>(bias_row + col, DecodeTable<dtype>()) * bias_scale;
        }
        out[row * n + col] = EncodeFp8ForX86<dtype>(sum / out_scale);
    }
}

template <DataType dtype>
int32_t ComputeRowMajorTyped(const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
                             const uint8_t* bias, float bias_scale, int64_t m, int64_t k, int64_t n,
                             LinearBiasType bias_type, uint8_t* out, float out_scale, float alpha, float beta) {
    std::vector<float> decoded_lhs;
    for (int64_t row = 0; row < m; ++row) {
        if (!DecodeInputRow<dtype>(lhs + row * k, k, lhs_scale, &decoded_lhs)) {
            return -1;
        }
        const int64_t full_blocks = n / kFp8OutputBlock;
        const size_t workers = Fp8WorkerCount(m, k, n);
#if defined(FEATHER_WITH_OPENMP)
        if (workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t block = 0; block < full_blocks; ++block) {
                ComputeRowMajorFullBlock<dtype>(decoded_lhs.data(), rhs, bias, rhs_scale, bias_scale, out_scale,
                                                alpha, beta, row, k, n, bias_type, block * kFp8OutputBlock, out);
            }
        } else
#endif
        {
            for (int64_t block = 0; block < full_blocks; ++block) {
                ComputeRowMajorFullBlock<dtype>(decoded_lhs.data(), rhs, bias, rhs_scale, bias_scale, out_scale,
                                                alpha, beta, row, k, n, bias_type, block * kFp8OutputBlock, out);
            }
        }
        const int64_t tail_begin = full_blocks * kFp8OutputBlock;
        if (tail_begin < n) {
            ComputeRowMajorTail<dtype>(decoded_lhs.data(), rhs, bias, rhs_scale, bias_scale, out_scale, alpha, beta,
                                       row, k, n, bias_type, tail_begin, out);
        }
    }
    return 0;
}

template <DataType dtype>
int32_t ComputeTransposedTyped(const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
                               const uint8_t* bias, float bias_scale, float alpha, float beta, int64_t m,
                               int64_t k, int64_t n, LinearBiasType bias_type, uint8_t* out, float out_scale) {
    if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        return -1;
    }
    std::vector<float> decoded_lhs;
    for (int64_t row = 0; row < m; ++row) {
        if (!DecodeInputRow<dtype>(lhs + row * k, k, lhs_scale, &decoded_lhs)) {
            return -1;
        }
#if defined(FEATHER_WITH_OPENMP)
        if (Fp8WorkerCount(m, k, n) > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(Fp8WorkerCount(m, k, n))
            for (int64_t col = 0; col < n; ++col) {
#else
        {
            for (int64_t col = 0; col < n; ++col) {
#endif
            const uint8_t* rhs_row = rhs + col * k;
            __m256 accumulator = _mm256_setzero_ps();
            int64_t inner = 0;
            for (; inner + 8 <= k; inner += 8) {
            accumulator = _mm256_fmadd_ps(
                    _mm256_loadu_ps(decoded_lhs.data() + inner), LoadFp8x8Typed<dtype>(rhs_row + inner), accumulator);
            }
            float sum = HorizontalSum(accumulator);
            for (; inner < k; ++inner) {
                sum += decoded_lhs[static_cast<size_t>(inner)] * DecodeScalar<dtype>(rhs_row + inner, DecodeTable<dtype>());
            }
            sum *= rhs_scale * alpha;
            if (bias != nullptr && bias_type != LinearBiasType::kNone) {
                const int64_t bias_offset = bias_type == LinearBiasType::kVector ? col : row * n + col;
                sum += beta * bias_scale * DecodeScalar<dtype>(bias + bias_offset, DecodeTable<dtype>());
            }
            out[row * n + col] = EncodeFp8ForX86<dtype>(sum / out_scale);
        }
#if defined(FEATHER_WITH_OPENMP)
        } else {
            for (int64_t col = 0; col < n; ++col) {
                const uint8_t* rhs_row = rhs + col * k;
                float sum = 0.0f;
                int64_t inner = 0;
                for (; inner + 8 <= k; inner += 8) {
                    __m256 accumulator = _mm256_mul_ps(_mm256_loadu_ps(decoded_lhs.data() + inner),
                                                       LoadFp8x8Typed<dtype>(rhs_row + inner));
                    sum += HorizontalSum(accumulator);
                }
                for (; inner < k; ++inner) {
                    sum += decoded_lhs[static_cast<size_t>(inner)] *
                           DecodeScalar<dtype>(rhs_row + inner, DecodeTable<dtype>());
                }
                sum *= rhs_scale * alpha;
                if (bias != nullptr && bias_type != LinearBiasType::kNone) {
                    const int64_t bias_offset = bias_type == LinearBiasType::kVector ? col : row * n + col;
                    sum += beta * bias_scale * DecodeScalar<dtype>(bias + bias_offset, DecodeTable<dtype>());
                }
                out[row * n + col] = EncodeFp8ForX86<dtype>(sum / out_scale);
            }
        }
#endif
    }
    return 0;
}

}  // namespace

bool Fp8LinearWorkspace::Resize(size_t count) {
    try {
        values_.resize(count);
    } catch (const std::bad_alloc&) {
        values_.clear();
        return false;
    }
    return true;
}

bool PackedFp8Rhs::Matches(DataType dtype, const uint8_t* rhs, int64_t k, int64_t n,
                           uint64_t source_version) const {
    return (dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2) && rhs != nullptr && dtype_ == dtype &&
           source_ == rhs && k_ == k && n_ == n && source_version_ == source_version && !data_.empty();
}

bool PackedFp8Rhs::Pack(DataType dtype, const uint8_t* rhs, int64_t k, int64_t n, uint64_t source_version) {
    if (Matches(dtype, rhs, k, n, source_version)) {
        return true;
    }
    dtype_ = DataType::UNKNOWN;
    source_ = nullptr;
    k_ = 0;
    n_ = 0;
    source_version_ = 0;
    data_.clear();
    if ((dtype != DataType::FP8E4M3 && dtype != DataType::FP8E5M2) || rhs == nullptr || k <= 0 || n <= 0 ||
        n > std::numeric_limits<int64_t>::max() - (kFp8OutputBlock - 1)) {
        return false;
    }
    const int64_t block_count = (n + kFp8OutputBlock - 1) / kFp8OutputBlock;
    if (block_count > std::numeric_limits<int64_t>::max() / kFp8OutputBlock ||
        k > std::numeric_limits<int64_t>::max() / (block_count * kFp8OutputBlock)) {
        return false;
    }
    const int64_t element_count = block_count * k * kFp8OutputBlock;
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
        const int64_t block_begin = block * kFp8OutputBlock;
        const int64_t width = std::min(kFp8OutputBlock, n - block_begin);
        for (int64_t row = 0; row < k; ++row) {
            uint16_t* destination = data_.data() + (block * k + row) * kFp8OutputBlock;
            for (int64_t lane = 0; lane < width; ++lane) {
                destination[lane] = DecodeFp8ToBf16(dtype, rhs[row * n + block_begin + lane]);
            }
        }
    }
    dtype_ = dtype;
    source_ = rhs;
    k_ = k;
    n_ = n;
    source_version_ = source_version;
    return true;
}

bool PackedFp8TransposedRhs::Matches(DataType dtype, const uint8_t* rhs_transposed, int64_t k, int64_t n,
                                     uint64_t source_version) const {
    return (dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2) && rhs_transposed != nullptr &&
           dtype_ == dtype && source_ == rhs_transposed && k_ == k && n_ == n &&
           source_version_ == source_version && !data_.empty();
}

bool PackedFp8TransposedRhs::Pack(DataType dtype, const uint8_t* rhs_transposed, int64_t k, int64_t n,
                                  uint64_t source_version) {
    if (Matches(dtype, rhs_transposed, k, n, source_version)) {
        return true;
    }
    dtype_ = DataType::UNKNOWN;
    source_ = nullptr;
    k_ = 0;
    n_ = 0;
    source_version_ = 0;
    data_.clear();
    if ((dtype != DataType::FP8E4M3 && dtype != DataType::FP8E5M2) || rhs_transposed == nullptr || k <= 0 || n <= 0 ||
        n > std::numeric_limits<int64_t>::max() - (kFp8OutputBlock - 1)) {
        return false;
    }
    const int64_t block_count = (n + kFp8OutputBlock - 1) / kFp8OutputBlock;
    if (block_count > std::numeric_limits<int64_t>::max() / kFp8OutputBlock ||
        k > std::numeric_limits<int64_t>::max() / (block_count * kFp8OutputBlock)) {
        return false;
    }
    const int64_t element_count = block_count * k * kFp8OutputBlock;
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
        const int64_t block_begin = block * kFp8OutputBlock;
        const int64_t width = std::min(kFp8OutputBlock, n - block_begin);
        for (int64_t row = 0; row < k; ++row) {
            uint16_t* destination = data_.data() + (block * k + row) * kFp8OutputBlock;
            for (int64_t lane = 0; lane < width; ++lane) {
                destination[lane] = DecodeFp8ToBf16(dtype, rhs_transposed[(block_begin + lane) * k + row]);
            }
        }
    }
    dtype_ = dtype;
    source_ = rhs_transposed;
    k_ = k;
    n_ = n;
    source_version_ = source_version;
    return true;
}

int32_t ComputeLinearRowMajorX86Fp8(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
    const uint8_t* bias, float bias_scale, int64_t m, int64_t k, int64_t n, LinearBiasType bias_type,
    uint8_t* out, float out_scale, float alpha, float beta) {
    if (!ValidateCommonArguments(dtype, lhs, lhs_scale, rhs, rhs_scale, bias, bias_scale, m, k, n, bias_type, out,
                                 out_scale) ||
        !std::isfinite(alpha) || !std::isfinite(beta)) {
        return -1;
    }
    if (dtype == DataType::FP8E4M3) {
        return ComputeRowMajorTyped<DataType::FP8E4M3>(lhs, lhs_scale, rhs, rhs_scale, bias, bias_scale, m, k, n,
                                                       bias_type, out, out_scale, alpha, beta);
    }
    return ComputeRowMajorTyped<DataType::FP8E5M2>(lhs, lhs_scale, rhs, rhs_scale, bias, bias_scale, m, k, n,
                                                   bias_type, out, out_scale, alpha, beta);
}

template <DataType dtype, typename PackedRhs>
int32_t ComputePackedFp8RowMajorTyped(const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
                                      const PackedRhs& packed_rhs, const uint8_t* bias, float bias_scale,
                                      int64_t m, int64_t k, int64_t n, LinearBiasType bias_type, uint8_t* out,
                                      float out_scale, float alpha, float beta, uint64_t source_version,
                                      Fp8LinearWorkspace* workspace) {
    if (m != 1 || lhs == nullptr || rhs == nullptr || out == nullptr || k <= 0 || n <= 0 ||
        !packed_rhs.Matches(dtype, rhs, k, n, source_version) || !IsValidScale(lhs_scale) ||
        !IsValidScale(rhs_scale) || !IsValidScale(out_scale) || !std::isfinite(alpha) || !std::isfinite(beta)) {
        return -1;
    }
    Fp8LinearWorkspace local_workspace;
    if (workspace == nullptr) workspace = &local_workspace;
    if (!workspace->Resize(static_cast<size_t>(k))) return -1;
    const float* table = DecodeTable<dtype>();
    const __m256 lhs_scale_vector = _mm256_set1_ps(lhs_scale);
    int64_t inner = 0;
    for (; inner + 8 <= k; inner += 8) {
        _mm256_storeu_ps(workspace->data() + inner,
                         _mm256_mul_ps(LoadFp8x8Typed<dtype>(lhs + inner), lhs_scale_vector));
    }
    for (; inner < k; ++inner) {
        workspace->data()[inner] = table[lhs[inner]] * lhs_scale;
    }

    const int64_t block_count = (n + kFp8OutputBlock - 1) / kFp8OutputBlock;
    const size_t workers = Fp8WorkerCount(m, k, n);
    auto compute_block = [&](int64_t block) {
        const int64_t block_begin = block * kFp8OutputBlock;
        const int64_t width = std::min(kFp8OutputBlock, n - block_begin);
        const uint16_t* rhs_block = packed_rhs.data() + block * k * kFp8OutputBlock;
        __m256 accumulators[kFp8OutputBlock / 8];
        for (auto& accumulator : accumulators) accumulator = _mm256_setzero_ps();
        if (bias != nullptr && bias_type != LinearBiasType::kNone) {
            const uint8_t* bias_row = bias_type == LinearBiasType::kVector ? bias : bias;
            const __m256 bias_scale_vector = _mm256_set1_ps(bias_scale * beta);
            for (int64_t lane_block = 0; lane_block < kFp8OutputBlock / 8; ++lane_block) {
                const int64_t offset = lane_block * 8;
                const int64_t lane_limit = std::min<int64_t>(8, width - offset);
                if (lane_limit == 8) {
                    accumulators[lane_block] = _mm256_mul_ps(
                        LoadFp8x8Typed<dtype>(bias_row + block_begin + offset), bias_scale_vector);
                } else {
                    // The final output block may be narrower than one SIMD
                    // load. Copy only the logical bias lanes so the decoder
                    // never reads beyond the tensor allocation.
                    alignas(32) uint8_t tail_bias[8] = {};
                    for (int64_t lane = 0; lane < lane_limit; ++lane) {
                        tail_bias[lane] = bias_row[block_begin + offset + lane];
                    }
                    accumulators[lane_block] = _mm256_mul_ps(
                        LoadFp8x8Typed<dtype>(tail_bias), bias_scale_vector);
                }
            }
        }
        int64_t row = 0;
        for (; row + 4 <= k; row += 4) {
            const __m256 lhs0 = _mm256_set1_ps(workspace->data()[row + 0]);
            const __m256 lhs1 = _mm256_set1_ps(workspace->data()[row + 1]);
            const __m256 lhs2 = _mm256_set1_ps(workspace->data()[row + 2]);
            const __m256 lhs3 = _mm256_set1_ps(workspace->data()[row + 3]);
            const uint16_t* rhs_row0 = rhs_block + (row + 0) * kFp8OutputBlock;
            const uint16_t* rhs_row1 = rhs_block + (row + 1) * kFp8OutputBlock;
            const uint16_t* rhs_row2 = rhs_block + (row + 2) * kFp8OutputBlock;
            const uint16_t* rhs_row3 = rhs_block + (row + 3) * kFp8OutputBlock;
            if (row + 8 < k) {
                __builtin_prefetch(rhs_row0 + 8 * kFp8OutputBlock, 0, 1);
            }
            for (int64_t lane_block = 0; lane_block < kFp8OutputBlock / 8; ++lane_block) {
                const int64_t offset = lane_block * 8;
                accumulators[lane_block] = _mm256_fmadd_ps(lhs0, LoadBf16x8(rhs_row0 + offset), accumulators[lane_block]);
                accumulators[lane_block] = _mm256_fmadd_ps(lhs1, LoadBf16x8(rhs_row1 + offset), accumulators[lane_block]);
                accumulators[lane_block] = _mm256_fmadd_ps(lhs2, LoadBf16x8(rhs_row2 + offset), accumulators[lane_block]);
                accumulators[lane_block] = _mm256_fmadd_ps(lhs3, LoadBf16x8(rhs_row3 + offset), accumulators[lane_block]);
            }
        }
        for (; row < k; ++row) {
            const __m256 lhs_value = _mm256_set1_ps(workspace->data()[row]);
            const uint16_t* rhs_row = rhs_block + row * kFp8OutputBlock;
            for (int64_t lane_block = 0; lane_block < kFp8OutputBlock / 8; ++lane_block) {
                accumulators[lane_block] = _mm256_fmadd_ps(
                    lhs_value, LoadBf16x8(rhs_row + lane_block * 8), accumulators[lane_block]);
            }
        }
        alignas(32) float values[8];
        const __m256 rhs_alpha_scale = _mm256_set1_ps(rhs_scale * alpha);
        const __m256 output_scale = _mm256_set1_ps(out_scale);
        for (int64_t lane_block = 0; lane_block < kFp8OutputBlock / 8; ++lane_block) {
            const int64_t lane_limit = std::min<int64_t>(8, width - lane_block * 8);
            if (lane_limit == 8) {
                _mm256_store_ps(values, _mm256_div_ps(
                                             _mm256_mul_ps(accumulators[lane_block], rhs_alpha_scale), output_scale));
                EncodeFp8x8ForX86(dtype, values, out + block_begin + lane_block * 8);
            } else if (lane_limit > 0) {
                _mm256_store_ps(values, _mm256_div_ps(
                                             _mm256_mul_ps(accumulators[lane_block], rhs_alpha_scale), output_scale));
                for (int64_t lane = 0; lane < lane_limit; ++lane) {
                    out[block_begin + lane_block * 8 + lane] = EncodeFp8ForX86<dtype>(values[lane]);
                }
            }
        }
    };

#if defined(FEATHER_WITH_OPENMP)
    if (workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t block = 0; block < block_count; ++block) compute_block(block);
    } else
#endif
    {
        for (int64_t block = 0; block < block_count; ++block) compute_block(block);
    }
    return 0;
}

int32_t ComputeLinearRowMajorX86Fp8PackedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs, float rhs_scale,
    const PackedFp8Rhs& packed_rhs, const uint8_t* bias, float bias_scale, int64_t m, int64_t k, int64_t n,
    LinearBiasType bias_type, uint8_t* out, float out_scale, float alpha, float beta, uint64_t source_version,
    Fp8LinearWorkspace* workspace) {
    if (!ValidateCommonArguments(dtype, lhs, lhs_scale, rhs, rhs_scale, bias, bias_scale, m, k, n, bias_type, out,
                                 out_scale)) {
        return -1;
    }
    if (dtype == DataType::FP8E4M3) {
        return ComputePackedFp8RowMajorTyped<DataType::FP8E4M3>(
            lhs, lhs_scale, rhs, rhs_scale, packed_rhs, bias, bias_scale, m, k, n, bias_type, out, out_scale,
            alpha, beta, source_version, workspace);
    }
    if (dtype == DataType::FP8E5M2) {
        return ComputePackedFp8RowMajorTyped<DataType::FP8E5M2>(
            lhs, lhs_scale, rhs, rhs_scale, packed_rhs, bias, bias_scale, m, k, n, bias_type, out, out_scale,
            alpha, beta, source_version, workspace);
    }
    return -1;
}

int32_t ComputeLinearRowMajorX86Fp8PackedTransposedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const PackedFp8TransposedRhs& packed_rhs, const uint8_t* bias, float bias_scale, int64_t m, int64_t k,
    int64_t n, LinearBiasType bias_type, uint8_t* out, float out_scale, float alpha, float beta,
    uint64_t source_version, Fp8LinearWorkspace* workspace) {
    if (!ValidateCommonArguments(dtype, lhs, lhs_scale, rhs_transposed, rhs_scale, bias, bias_scale, m, k, n,
                                 bias_type, out, out_scale)) {
        return -1;
    }
    if (dtype == DataType::FP8E4M3) {
        return ComputePackedFp8RowMajorTyped<DataType::FP8E4M3>(
            lhs, lhs_scale, rhs_transposed, rhs_scale, packed_rhs, bias, bias_scale, m, k, n, bias_type, out,
            out_scale, alpha, beta, source_version, workspace);
    }
    if (dtype == DataType::FP8E5M2) {
        return ComputePackedFp8RowMajorTyped<DataType::FP8E5M2>(
            lhs, lhs_scale, rhs_transposed, rhs_scale, packed_rhs, bias, bias_scale, m, k, n, bias_type, out,
            out_scale, alpha, beta, source_version, workspace);
    }
    return -1;
}

int32_t ComputeLinearRowMajorX86Fp8TransposedRhs(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const uint8_t* bias, float bias_scale, float alpha, float beta, int64_t m, int64_t k, int64_t n,
    LinearBiasType bias_type, uint8_t* out, float out_scale) {
    if (!ValidateCommonArguments(dtype, lhs, lhs_scale, rhs_transposed, rhs_scale, bias, bias_scale, m, k, n,
                                 bias_type, out, out_scale)) {
        return -1;
    }
    if (dtype == DataType::FP8E4M3) {
        return ComputeTransposedTyped<DataType::FP8E4M3>(lhs, lhs_scale, rhs_transposed, rhs_scale, bias,
                                                         bias_scale, alpha, beta, m, k, n, bias_type, out, out_scale);
    }
    return ComputeTransposedTyped<DataType::FP8E5M2>(lhs, lhs_scale, rhs_transposed, rhs_scale, bias, bias_scale,
                                                     alpha, beta, m, k, n, bias_type, out, out_scale);
}

struct Fp8ArgmaxResult {
    int64_t index{-1};
    float value{0.0f};
};

struct Fp8Bf16x16 {
    __m256 lo;
    __m256 hi;
};

inline Fp8Bf16x16 LoadBf16x16ForArgmax(const uint16_t* input) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const __m256i low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(packed));
    const __m256i high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(packed, 1));
    return {_mm256_castsi256_ps(_mm256_slli_epi32(low, 16)),
            _mm256_castsi256_ps(_mm256_slli_epi32(high, 16))};
}

inline void FmaBf16x16ForArgmax(__m256 lhs, const uint16_t* rhs, __m256* acc_lo, __m256* acc_hi) {
    const Fp8Bf16x16 values = LoadBf16x16ForArgmax(rhs);
    *acc_lo = _mm256_fmadd_ps(lhs, values.lo, *acc_lo);
    *acc_hi = _mm256_fmadd_ps(lhs, values.hi, *acc_hi);
}

inline uint16_t RoundedBf16Scalar(float value) { return FloatToBFloat16(value); }

inline void ConsiderFp8ArgmaxValue(int64_t index, uint16_t rounded_bits, Fp8ArgmaxResult* result) {
    if (result == nullptr || IsNaNBf16(rounded_bits)) {
        return;
    }
    const float rounded = BFloat16ToFloat(rounded_bits);
    // Keep the first index for equal BF16 values, including signed zero.
    if (result->index < 0 || rounded > result->value) {
        result->index = index;
        result->value = rounded;
    }
}

template <DataType dtype>
inline void ConsiderFp8ArgmaxVector(__m256 value, int64_t base_index, int64_t lane_limit, float rhs_scale,
                                    float out_scale, const std::array<uint16_t, 256>& rounded_codes,
                                    Fp8ArgmaxResult* result) {
    if (result == nullptr || lane_limit <= 0) {
        return;
    }
    alignas(32) float values[8];
    alignas(32) uint8_t encoded[8];
    _mm256_store_ps(values, value);
    const int valid_lane_count = std::min<int64_t>(lane_limit, 8);
    _mm256_store_ps(values, _mm256_div_ps(_mm256_mul_ps(value, _mm256_set1_ps(rhs_scale)),
                                          _mm256_set1_ps(out_scale)));
    EncodeFp8x8ForX86(dtype, values, encoded);
    for (int lane = 0; lane < valid_lane_count; ++lane) {
        ConsiderFp8ArgmaxValue(base_index + lane, rounded_codes[encoded[lane]], result);
    }
}

template <DataType dtype>
int32_t ComputeFp8ArgmaxTyped(const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed,
                              float rhs_scale, const PackedFp8TransposedRhs* packed_rhs, int64_t k, int64_t n,
                              float out_scale, int64_t* token, Fp8LinearWorkspace* workspace,
                              uint64_t source_version) {
    if (lhs == nullptr || rhs_transposed == nullptr || token == nullptr || k <= 0 || n <= 0 ||
        !IsValidScale(lhs_scale) || !IsValidScale(rhs_scale) || !IsValidScale(out_scale) ||
        (packed_rhs != nullptr && !packed_rhs->Matches(dtype, rhs_transposed, k, n, source_version))) {
        return -1;
    }
    Fp8LinearWorkspace local_workspace;
    if (workspace == nullptr) {
        workspace = &local_workspace;
    }
    if (!workspace->Resize(static_cast<size_t>(k))) {
        return -1;
    }
    const float* lhs_table = DecodeTable<dtype>();
    for (int64_t row = 0; row < k; ++row) {
        workspace->data()[row] = lhs_table[lhs[row]] * lhs_scale;
    }

    constexpr int64_t kTile = kFp8OutputBlock;
    const int64_t block_count = (n + kTile - 1) / kTile;
    std::vector<Fp8ArgmaxResult> partial(static_cast<size_t>(block_count));
    const uint16_t* packed_data = packed_rhs == nullptr ? nullptr : packed_rhs->data();
    std::array<uint16_t, 256> rounded_codes{};
    for (int code = 0; code < 256; ++code) {
        const float decoded = dtype == DataType::FP8E4M3 ? Fp8E4M3ToFloat(static_cast<uint8_t>(code))
                                                          : Fp8E5M2ToFloat(static_cast<uint8_t>(code));
        rounded_codes[static_cast<size_t>(code)] = FloatToBFloat16(decoded * out_scale);
    }
    const size_t worker_count = Fp8WorkerCount(1, k, n);
    auto compute_block = [&](int64_t block) {
        const int64_t block_begin = block * kTile;
        const int64_t width = std::min(kTile, n - block_begin);
        auto& result = partial[static_cast<size_t>(block)];
        // Match the tuned BF16 lm-head layout when the packed RHS is split
        // across workers. Eight accumulators cover the full 64-column block,
        // so each K row is loaded once for the block instead of twice through
        // two 32-column tiles. Keep the 32-column form for serial execution,
        // where the smaller register footprint is faster on AVX2.
        if (packed_data != nullptr && worker_count > 1) {
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
                const uint16_t* rhs_row0 = packed_data + (block * k + row + 0) * kTile;
                const uint16_t* rhs_row1 = packed_data + (block * k + row + 1) * kTile;
                const uint16_t* rhs_row2 = packed_data + (block * k + row + 2) * kTile;
                const uint16_t* rhs_row3 = packed_data + (block * k + row + 3) * kTile;
                if (row + 8 < k) {
                    __builtin_prefetch(rhs_row0 + 8 * kTile, 0, 1);
                }
                const __m256 lhs0 = _mm256_set1_ps(workspace->data()[row + 0]);
                const __m256 lhs1 = _mm256_set1_ps(workspace->data()[row + 1]);
                const __m256 lhs2 = _mm256_set1_ps(workspace->data()[row + 2]);
                const __m256 lhs3 = _mm256_set1_ps(workspace->data()[row + 3]);
                FmaBf16x16ForArgmax(lhs0, rhs_row0 + 0, &acc0, &acc1);
                FmaBf16x16ForArgmax(lhs0, rhs_row0 + 16, &acc2, &acc3);
                FmaBf16x16ForArgmax(lhs0, rhs_row0 + 32, &acc4, &acc5);
                FmaBf16x16ForArgmax(lhs0, rhs_row0 + 48, &acc6, &acc7);
                FmaBf16x16ForArgmax(lhs1, rhs_row1 + 0, &acc0, &acc1);
                FmaBf16x16ForArgmax(lhs1, rhs_row1 + 16, &acc2, &acc3);
                FmaBf16x16ForArgmax(lhs1, rhs_row1 + 32, &acc4, &acc5);
                FmaBf16x16ForArgmax(lhs1, rhs_row1 + 48, &acc6, &acc7);
                FmaBf16x16ForArgmax(lhs2, rhs_row2 + 0, &acc0, &acc1);
                FmaBf16x16ForArgmax(lhs2, rhs_row2 + 16, &acc2, &acc3);
                FmaBf16x16ForArgmax(lhs2, rhs_row2 + 32, &acc4, &acc5);
                FmaBf16x16ForArgmax(lhs2, rhs_row2 + 48, &acc6, &acc7);
                FmaBf16x16ForArgmax(lhs3, rhs_row3 + 0, &acc0, &acc1);
                FmaBf16x16ForArgmax(lhs3, rhs_row3 + 16, &acc2, &acc3);
                FmaBf16x16ForArgmax(lhs3, rhs_row3 + 32, &acc4, &acc5);
                FmaBf16x16ForArgmax(lhs3, rhs_row3 + 48, &acc6, &acc7);
            }
            for (; row < k; ++row) {
                const __m256 lhs_value = _mm256_set1_ps(workspace->data()[row]);
                const uint16_t* rhs_row = packed_data + (block * k + row) * kTile;
                FmaBf16x16ForArgmax(lhs_value, rhs_row + 0, &acc0, &acc1);
                FmaBf16x16ForArgmax(lhs_value, rhs_row + 16, &acc2, &acc3);
                FmaBf16x16ForArgmax(lhs_value, rhs_row + 32, &acc4, &acc5);
                FmaBf16x16ForArgmax(lhs_value, rhs_row + 48, &acc6, &acc7);
            }
            ConsiderFp8ArgmaxVector<dtype>(acc0, block_begin + 0, width, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc1, block_begin + 8, width - 8, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc2, block_begin + 16, width - 16, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc3, block_begin + 24, width - 24, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc4, block_begin + 32, width - 32, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc5, block_begin + 40, width - 40, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc6, block_begin + 48, width - 48, rhs_scale, out_scale, rounded_codes,
                                            &result);
            ConsiderFp8ArgmaxVector<dtype>(acc7, block_begin + 56, width - 56, rhs_scale, out_scale, rounded_codes,
                                            &result);
            return;
        }

        // Packed weights are laid out in 64-column blocks. Process each block
        // as two 32-column tiles so AVX2 keeps four accumulators live, matching
        // the tuned BF16 lm-head kernel and avoiding register spills.
        for (int64_t tile_begin = block_begin; tile_begin < block_begin + width; tile_begin += 32) {
            const int64_t tile_width = std::min<int64_t>(32, block_begin + width - tile_begin);
            const int64_t packed_offset = tile_begin - block_begin;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            if (packed_data != nullptr) {
                for (int64_t row = 0; row + 4 <= k; row += 4) {
                    const uint16_t* rhs_row0 = packed_data + (block * k + row + 0) * kTile + packed_offset;
                    const uint16_t* rhs_row1 = packed_data + (block * k + row + 1) * kTile + packed_offset;
                    const uint16_t* rhs_row2 = packed_data + (block * k + row + 2) * kTile + packed_offset;
                    const uint16_t* rhs_row3 = packed_data + (block * k + row + 3) * kTile + packed_offset;
                    if (row + 8 < k) {
                        __builtin_prefetch(rhs_row0 + 8 * kTile, 0, 1);
                    }
                    const __m256 lhs0 = _mm256_set1_ps(workspace->data()[row + 0]);
                    const __m256 lhs1 = _mm256_set1_ps(workspace->data()[row + 1]);
                    const __m256 lhs2 = _mm256_set1_ps(workspace->data()[row + 2]);
                    const __m256 lhs3 = _mm256_set1_ps(workspace->data()[row + 3]);
                    FmaBf16x16ForArgmax(lhs0, rhs_row0 + 0, &acc0, &acc1);
                    FmaBf16x16ForArgmax(lhs0, rhs_row0 + 16, &acc2, &acc3);
                    FmaBf16x16ForArgmax(lhs1, rhs_row1 + 0, &acc0, &acc1);
                    FmaBf16x16ForArgmax(lhs1, rhs_row1 + 16, &acc2, &acc3);
                    FmaBf16x16ForArgmax(lhs2, rhs_row2 + 0, &acc0, &acc1);
                    FmaBf16x16ForArgmax(lhs2, rhs_row2 + 16, &acc2, &acc3);
                    FmaBf16x16ForArgmax(lhs3, rhs_row3 + 0, &acc0, &acc1);
                    FmaBf16x16ForArgmax(lhs3, rhs_row3 + 16, &acc2, &acc3);
                }
                for (int64_t row = (k / 4) * 4; row < k; ++row) {
                    const __m256 lhs_value = _mm256_set1_ps(workspace->data()[row]);
                    const uint16_t* rhs_row = packed_data + (block * k + row) * kTile + packed_offset;
                    FmaBf16x16ForArgmax(lhs_value, rhs_row + 0, &acc0, &acc1);
                    FmaBf16x16ForArgmax(lhs_value, rhs_row + 16, &acc2, &acc3);
                }
            } else {
                // The direct path is retained for callers without an immutable
                // packed RHS. It is uncommon in Qwen inference and keeps the
                // existing bounds-safe behavior for the final partial tile.
                for (int64_t row = 0; row < k; ++row) {
                    const __m256 lhs_value = _mm256_set1_ps(workspace->data()[row]);
                    alignas(32) uint8_t bytes[32]{};
                    for (int lane = 0; lane < 32; ++lane) {
                        const int64_t col = tile_begin + lane;
                        bytes[lane] = col < n ? rhs_transposed[col * k + row] : 0;
                    }
                    const __m256 decoded0 = LoadFp8x8Typed<dtype>(bytes + 0);
                    const __m256 decoded1 = LoadFp8x8Typed<dtype>(bytes + 8);
                    const __m256 decoded2 = LoadFp8x8Typed<dtype>(bytes + 16);
                    const __m256 decoded3 = LoadFp8x8Typed<dtype>(bytes + 24);
                    acc0 = _mm256_fmadd_ps(lhs_value, decoded0, acc0);
                    acc1 = _mm256_fmadd_ps(lhs_value, decoded1, acc1);
                    acc2 = _mm256_fmadd_ps(lhs_value, decoded2, acc2);
                    acc3 = _mm256_fmadd_ps(lhs_value, decoded3, acc3);
                }
            }

            const __m256 accumulators[] = {acc0, acc1, acc2, acc3};
            for (int64_t lane_block = 0; lane_block < 4; ++lane_block) {
                const int64_t lane_limit = std::min<int64_t>(8, tile_width - lane_block * 8);
                ConsiderFp8ArgmaxVector<dtype>(accumulators[lane_block], tile_begin + lane_block * 8, lane_limit,
                                                rhs_scale, out_scale, rounded_codes, &result);
            }
        }
    };

#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int64_t block = 0; block < block_count; ++block) {
            compute_block(block);
        }
    } else
#endif
    {
        for (int64_t block = 0; block < block_count; ++block) {
            compute_block(block);
        }
    }

    Fp8ArgmaxResult best;
    for (const auto& candidate : partial) {
        if (candidate.index >= 0 && (best.index < 0 || candidate.value > best.value)) {
            best = candidate;
        }
    }
    *token = best.index < 0 ? 0 : best.index;
    return 0;
}

int32_t ComputeLinearRowMajorX86Fp8TransposedRhsArgmax(
    DataType dtype, const uint8_t* lhs, float lhs_scale, const uint8_t* rhs_transposed, float rhs_scale,
    const PackedFp8TransposedRhs* packed_rhs, int64_t k, int64_t n, float out_scale, int64_t* token,
    Fp8LinearWorkspace* workspace, uint64_t source_version) {
    if (dtype == DataType::FP8E4M3) {
        return ComputeFp8ArgmaxTyped<DataType::FP8E4M3>(lhs, lhs_scale, rhs_transposed, rhs_scale, packed_rhs, k, n,
                                                        out_scale, token, workspace, source_version);
    }
    if (dtype == DataType::FP8E5M2) {
        return ComputeFp8ArgmaxTyped<DataType::FP8E5M2>(lhs, lhs_scale, rhs_transposed, rhs_scale, packed_rhs, k, n,
                                                        out_scale, token, workspace, source_version);
    }
    return -1;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
