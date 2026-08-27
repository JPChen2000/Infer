#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "util/bf16.h"
#include "util/fp16.h"

namespace {

std::vector<float> MakeMatrix(int64_t rows, int64_t cols, float seed) {
    std::vector<float> values(static_cast<size_t>(rows * cols));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            values[static_cast<size_t>(i * cols + j)] = seed + static_cast<float>(((i * 13 + j * 7) % 17) - 8) * 0.125f;
        }
    }
    return values;
}

std::vector<uint16_t> MakeMatrixFp16(int64_t rows, int64_t cols, float seed) {
    const std::vector<float> fp32 = MakeMatrix(rows, cols, seed);
    std::vector<uint16_t> fp16(fp32.size());
    for (size_t i = 0; i < fp32.size(); ++i) {
        fp16[i] = feather::FloatToHalf(fp32[i]);
    }
    return fp16;
}

std::vector<uint16_t> MakeMatrixBf16(int64_t rows, int64_t cols, float seed) {
    const std::vector<float> fp32 = MakeMatrix(rows, cols, seed);
    std::vector<uint16_t> bf16(fp32.size());
    for (size_t i = 0; i < fp32.size(); ++i) {
        bf16[i] = feather::FloatToBFloat16(fp32[i]);
    }
    return bf16;
}

void BenchmarkMatMul(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<float> lhs = MakeMatrix(m, k, 1.0f);
    const std::vector<float> rhs = MakeMatrix(k, n, 2.0f);
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Fp32(
            lhs.data(), rhs.data(), nullptr, m, k, n, feather::kernel::x86::LinearBiasType::kNone, out.data());
        if (status != 0) {
            state.SkipWithError("matmul kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkGemmVectorBias(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<float> lhs = MakeMatrix(m, k, 1.5f);
    const std::vector<float> rhs = MakeMatrix(k, n, 0.5f);
    const std::vector<float> bias = MakeMatrix(1, n, 0.25f);
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Fp32(
            lhs.data(), rhs.data(), bias.data(), m, k, n, feather::kernel::x86::LinearBiasType::kVector, out.data());
        if (status != 0) {
            state.SkipWithError("gemm kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkMatMulFp16(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<uint16_t> lhs = MakeMatrixFp16(m, k, 1.0f);
    const std::vector<uint16_t> rhs = MakeMatrixFp16(k, n, 2.0f);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Fp16(
            lhs.data(), rhs.data(), nullptr, m, k, n, feather::kernel::x86::LinearBiasType::kNone, out.data());
        if (status != 0) {
            state.SkipWithError("fp16 matmul kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkMatMulBf16(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<uint16_t> lhs = MakeMatrixBf16(m, k, 1.0f);
    const std::vector<uint16_t> rhs = MakeMatrixBf16(k, n, 2.0f);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Bf16(
            lhs.data(), rhs.data(), nullptr, m, k, n, feather::kernel::x86::LinearBiasType::kNone, out.data());
        if (status != 0) {
            state.SkipWithError("bf16 matmul kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkMatMulBf16Packed(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<uint16_t> lhs = MakeMatrixBf16(m, k, 1.0f);
    const std::vector<uint16_t> rhs = MakeMatrixBf16(k, n, 2.0f);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
    feather::kernel::x86::PackedBf16Rhs packed_rhs;
    if (!packed_rhs.Pack(rhs.data(), k, n)) {
        state.SkipWithError("bf16 rhs packing failed");
        return;
    }

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedRhs(
            lhs.data(), rhs.data(), packed_rhs, nullptr, m, k, n, feather::kernel::x86::LinearBiasType::kNone,
            out.data());
        if (status != 0) {
            state.SkipWithError("bf16 packed matmul kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkGemmBf16PackedTransposed(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const std::vector<uint16_t> lhs = MakeMatrixBf16(m, k, 1.0f);
    std::vector<uint16_t> rhs_transposed(static_cast<size_t>(n * k));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            rhs_transposed[static_cast<size_t>(row * k + col)] =
                feather::FloatToBFloat16(2.0f + static_cast<float>((row * 13 + col * 7) % 17) * 0.03125f);
        }
    }
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
    feather::kernel::x86::PackedBf16TransposedRhs packed_rhs;
    if (!packed_rhs.Pack(rhs_transposed.data(), k, n)) {
        state.SkipWithError("bf16 transposed rhs packing failed");
        return;
    }

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhs(
            lhs.data(), rhs_transposed.data(), packed_rhs, nullptr, m, k, n,
            feather::kernel::x86::LinearBiasType::kNone, 1.0f, 0.0f, out.data());
        if (status != 0) {
            state.SkipWithError("bf16 transposed packed gemm failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkGemmBf16DirectTransposed(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const std::vector<uint16_t> lhs = MakeMatrixBf16(m, k, 1.0f);
    std::vector<uint16_t> rhs_transposed(static_cast<size_t>(n * k));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            rhs_transposed[static_cast<size_t>(row * k + col)] =
                feather::FloatToBFloat16(2.0f + static_cast<float>((row * 13 + col * 7) % 17) * 0.03125f);
        }
    }
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Bf16TransposedRhs(
            lhs.data(), rhs_transposed.data(), nullptr, m, k, n, feather::kernel::x86::LinearBiasType::kNone,
            1.0f, 0.0f, out.data());
        if (status != 0) {
            state.SkipWithError("bf16 direct transposed gemm failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkGemmBf16PackedTransposedArgmax(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const std::vector<uint16_t> lhs = MakeMatrixBf16(m, k, 1.0f);
    std::vector<uint16_t> rhs_transposed(static_cast<size_t>(n * k));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            rhs_transposed[static_cast<size_t>(row * k + col)] =
                feather::FloatToBFloat16(2.0f + static_cast<float>((row * 13 + col * 7) % 17) * 0.03125f);
        }
    }
    feather::kernel::x86::PackedBf16TransposedRhs packed_rhs;
    if (!packed_rhs.Pack(rhs_transposed.data(), k, n)) {
        state.SkipWithError("bf16 transposed rhs packing failed");
        return;
    }
    int64_t token = 0;
    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmax(
            lhs.data(), rhs_transposed.data(), packed_rhs, k, n, &token);
        if (status != 0) {
            state.SkipWithError("bf16 transposed argmax kernel failed");
            return;
        }
        benchmark::DoNotOptimize(token);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * m * n * k);
}

void BenchmarkGemmVectorBiasFp16(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);

    const std::vector<uint16_t> lhs = MakeMatrixFp16(m, k, 1.5f);
    const std::vector<uint16_t> rhs = MakeMatrixFp16(k, n, 0.5f);
    const std::vector<uint16_t> bias = MakeMatrixFp16(1, n, 0.25f);
    std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeLinearRowMajorX86Fp16(
            lhs.data(), rhs.data(), bias.data(), m, k, n, feather::kernel::x86::LinearBiasType::kVector, out.data());
        if (status != 0) {
            state.SkipWithError("fp16 gemm kernel failed");
            return;
        }
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * m * n * k);
}

BENCHMARK(BenchmarkMatMul)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({197, 768, 3072})
    ->Args({197, 3072, 768})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmVectorBias)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkMatMulFp16)->Args({64, 64, 64})->Args({128, 128, 128})->Args({256, 256, 256})->Unit(
    benchmark::kMicrosecond);
BENCHMARK(BenchmarkMatMulBf16)
    ->Args({1, 1024, 512})
    ->Args({1, 1024, 2048})
    ->Args({1, 1024, 3584})
    ->Args({1, 1024, 6144})
    ->Args({1, 2048, 1024})
    ->Args({1, 3584, 1024})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkMatMulBf16Packed)
    ->Args({1, 1024, 512})
    ->Args({1, 1024, 2048})
    ->Args({1, 1024, 3584})
    ->Args({1, 1024, 6144})
    ->Args({1, 2048, 1024})
    ->Args({1, 3584, 1024})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmBf16PackedTransposed)->Args({1, 1024, 65536})->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmBf16DirectTransposed)->Args({1, 1024, 65536})->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmBf16PackedTransposedArgmax)->Args({1, 1024, 65536})->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmVectorBiasFp16)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
