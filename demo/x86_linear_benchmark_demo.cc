#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
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

BENCHMARK(BenchmarkMatMul)->Args({64, 64, 64})->Args({128, 128, 128})->Args({256, 256, 256})->Unit(
    benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmVectorBias)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BenchmarkMatMulFp16)->Args({64, 64, 64})->Args({128, 128, 128})->Args({256, 256, 256})->Unit(
    benchmark::kMicrosecond);
BENCHMARK(BenchmarkGemmVectorBiasFp16)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
