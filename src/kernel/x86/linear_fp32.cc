#include "src/kernel/x86/linear_fp32.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <limits>

#include "util/threading.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace x86 {

namespace {

constexpr int64_t kFp32MinimumParallelMacs = 1 << 20;
constexpr int64_t kFp32BlockedMinimumRows = 4;
constexpr int64_t kFp32BlockedRows = 8;

size_t ConfiguredFp32WorkerLimit() {
    size_t worker_limit = DefaultThreadCount();
    const char* configured_limit = std::getenv("FEATHER_X86_FP32_THREADS");
    if (configured_limit == nullptr || configured_limit[0] == '\0') {
        return worker_limit;
    }

    char* end = nullptr;
    const unsigned long parsed_limit = std::strtoul(configured_limit, &end, 10);
    if (end != configured_limit && *end == '\0' && parsed_limit > 0) {
        worker_limit = std::min(worker_limit, static_cast<size_t>(parsed_limit));
    }
    return worker_limit;
}

inline __m256 LoadBiasVector(const float* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return _mm256_setzero_ps();
    }
    if (bias_type == LinearBiasType::kVector) {
        return _mm256_loadu_ps(bias + col);
    }
    return _mm256_loadu_ps(bias + row * n + col);
}

inline float LoadBiasScalar(const float* bias, LinearBiasType bias_type, int64_t row, int64_t n, int64_t col) {
    if (bias == nullptr || bias_type == LinearBiasType::kNone) {
        return 0.0f;
    }
    if (bias_type == LinearBiasType::kVector) {
        return bias[col];
    }
    return bias[row * n + col];
}

void ComputeBlockedRowsFp32(const float* lhs, const float* rhs, const float* bias, int64_t k, int64_t n,
                            LinearBiasType bias_type, float* out, int64_t begin_row, int64_t end_row) {
    for (int64_t row_block = begin_row; row_block < end_row; row_block += kFp32BlockedRows) {
        const int64_t rows = std::min<int64_t>(kFp32BlockedRows, end_row - row_block);
        int64_t col = 0;
        for (; col + 8 <= n; col += 8) {
            __m256 accumulators[kFp32BlockedRows];
            for (int64_t row = 0; row < rows; ++row) {
                accumulators[row] = LoadBiasVector(bias, bias_type, row_block + row, n, col);
            }

            for (int64_t t = 0; t < k; ++t) {
                const __m256 rhs_value = _mm256_loadu_ps(rhs + t * n + col);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[row] = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs[(row_block + row) * k + t]), rhs_value, accumulators[row]);
                }
            }

            for (int64_t row = 0; row < rows; ++row) {
                _mm256_storeu_ps(out + (row_block + row) * n + col, accumulators[row]);
            }
        }

        for (; col < n; ++col) {
            for (int64_t row = 0; row < rows; ++row) {
                float sum = LoadBiasScalar(bias, bias_type, row_block + row, n, col);
                for (int64_t t = 0; t < k; ++t) {
                    sum += lhs[(row_block + row) * k + t] * rhs[t * n + col];
                }
                out[(row_block + row) * n + col] = sum;
            }
        }
    }
}

}  // namespace

bool Fp32LinearUsesBlockedKernel(int64_t m, int64_t k, int64_t n) {
    return m >= kFp32BlockedMinimumRows && k > 0 && n >= 8 &&
           m <= std::numeric_limits<int64_t>::max() / k &&
           m * k <= std::numeric_limits<int64_t>::max() / n &&
           m * k * n >= kFp32MinimumParallelMacs;
}

size_t Fp32LinearWorkerCount(int64_t m, int64_t k, int64_t n) {
    if (m < 2 || k <= 0 || n <= 0 || m > std::numeric_limits<int64_t>::max() / k ||
        m * k > std::numeric_limits<int64_t>::max() / n || m * k * n < kFp32MinimumParallelMacs) {
        return 1;
    }

#if defined(FEATHER_WITH_OPENMP)
    if (omp_in_parallel()) {
        return 1;
    }
    size_t worker_limit = std::min(ConfiguredFp32WorkerLimit(), static_cast<size_t>(m));
    const int openmp_limit = omp_get_max_threads();
    if (openmp_limit > 0) {
        worker_limit = std::min(worker_limit, static_cast<size_t>(openmp_limit));
    }
    return std::max<size_t>(1, worker_limit);
#else
    return 1;
#endif
}

int32_t ComputeLinearRowMajorX86Fp32(const float* lhs, const float* rhs, const float* bias, int64_t m, int64_t k,
                                     int64_t n, LinearBiasType bias_type, float* out) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }

    const auto compute_row = [&](int64_t i) {
        const float* lhs_row = lhs + i * k;
        float* out_row = out + i * n;

        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 acc = LoadBiasVector(bias, bias_type, i, n, j);
            for (int64_t t = 0; t < k; ++t) {
                const __m256 lhs_value = _mm256_set1_ps(lhs_row[t]);
                const __m256 rhs_value = _mm256_loadu_ps(rhs + t * n + j);
                acc = _mm256_fmadd_ps(lhs_value, rhs_value, acc);
            }
            _mm256_storeu_ps(out_row + j, acc);
        }

        for (; j < n; ++j) {
            float sum = LoadBiasScalar(bias, bias_type, i, n, j);
            for (int64_t t = 0; t < k; ++t) {
                sum += lhs_row[t] * rhs[t * n + j];
            }
            out_row[j] = sum;
        }
    };

    if (Fp32LinearUsesBlockedKernel(m, k, n)) {
        const size_t worker_count = Fp32LinearWorkerCount(m, k, n);
#if defined(FEATHER_WITH_OPENMP)
        if (worker_count > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
            for (int64_t row_block = 0; row_block < m; row_block += kFp32BlockedRows) {
                const int64_t block_end = std::min(m, row_block + kFp32BlockedRows);
                ComputeBlockedRowsFp32(lhs, rhs, bias, k, n, bias_type, out, row_block, block_end);
            }
            return 0;
        }
#endif
        ComputeBlockedRowsFp32(lhs, rhs, bias, k, n, bias_type, out, 0, m);
        return 0;
    }

    const size_t worker_count = Fp32LinearWorkerCount(m, k, n);
#if defined(FEATHER_WITH_OPENMP)
    if (worker_count > 1) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int64_t i = 0; i < m; ++i) {
            compute_row(i);
        }
        return 0;
    }
#endif
    for (int64_t i = 0; i < m; ++i) {
        compute_row(i);
    }

    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
