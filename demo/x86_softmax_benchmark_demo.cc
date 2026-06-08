#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/operator/softmax_op.h"
#include "util/fp16.h"

namespace {

std::vector<float> MakeTensor(int64_t rows, int64_t cols, float seed) {
    std::vector<float> values(static_cast<size_t>(rows * cols));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            values[static_cast<size_t>(i * cols + j)] =
                seed + static_cast<float>(((i * 13 + j * 7) % 23) - 11) * 0.125f;
        }
    }
    return values;
}

std::vector<uint16_t> MakeTensorFp16(int64_t rows, int64_t cols, float seed) {
    const std::vector<float> fp32 = MakeTensor(rows, cols, seed);
    std::vector<uint16_t> fp16(fp32.size());
    for (size_t i = 0; i < fp32.size(); ++i) {
        fp16[i] = feather::FloatToHalf(fp32[i]);
    }
    return fp16;
}

void BenchmarkSoftmaxAxisLast(benchmark::State& state) {
    const int64_t rows = state.range(0);
    const int64_t cols = state.range(1);

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>(MakeTensor(rows, cols, 0.5f), {rows, cols});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{rows, cols});

    feather::operators::SoftmaxParam param{};
    param.input = input;
    param.out = output;
    param.axis = 1;

    auto kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP32, "Softmax");
    if (kernel == nullptr) {
        state.SkipWithError("failed to create x86 fp32 softmax kernel");
        return;
    }
    kernel->SetParam(&param);

    for (auto _ : state) {
        if (kernel->compute() != 0) {
            state.SkipWithError("x86 fp32 softmax failed");
            return;
        }
        benchmark::DoNotOptimize(output->raw_data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * rows * cols);
}

void BenchmarkSoftmaxAxisLastFp16(benchmark::State& state) {
    const int64_t rows = state.range(0);
    const int64_t cols = state.range(1);

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(MakeTensorFp16(rows, cols, 0.5f), {rows, cols});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{rows, cols});
    output->mutable_data<uint16_t>();

    feather::operators::SoftmaxParam param{};
    param.input = input;
    param.out = output;
    param.axis = -1;

    auto kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Softmax");
    if (kernel == nullptr) {
        state.SkipWithError("failed to create x86 fp16 softmax kernel");
        return;
    }
    kernel->SetParam(&param);

    if (kernel->compute() != 0) {
        state.SkipWithError("x86 fp16 softmax warmup failed");
        return;
    }

    for (auto _ : state) {
        if (kernel->compute() != 0) {
            state.SkipWithError("x86 fp16 softmax failed");
            return;
        }
        benchmark::DoNotOptimize(output->raw_data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * rows * cols);
}

void BenchmarkSoftmaxAxis0Fp16(benchmark::State& state) {
    const int64_t rows = state.range(0);
    const int64_t cols = state.range(1);

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(MakeTensorFp16(rows, cols, -0.25f), {rows, cols});
    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{rows, cols});
    output->mutable_data<uint16_t>();

    feather::operators::SoftmaxParam param{};
    param.input = input;
    param.out = output;
    param.axis = 0;

    auto kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Softmax");
    if (kernel == nullptr) {
        state.SkipWithError("failed to create x86 fp16 softmax kernel");
        return;
    }
    kernel->SetParam(&param);

    if (kernel->compute() != 0) {
        state.SkipWithError("x86 fp16 axis0 softmax warmup failed");
        return;
    }

    for (auto _ : state) {
        if (kernel->compute() != 0) {
            state.SkipWithError("x86 fp16 axis0 softmax failed");
            return;
        }
        benchmark::DoNotOptimize(output->raw_data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(BenchmarkSoftmaxAxisLast)
    ->Args({256, 256})
    ->Args({256, 1024})
    ->Args({1024, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BenchmarkSoftmaxAxisLastFp16)
    ->Args({256, 256})
    ->Args({256, 1024})
    ->Args({1024, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BenchmarkSoftmaxAxis0Fp16)
    ->Args({256, 256})
    ->Args({1024, 256})
    ->Args({2048, 256})
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
