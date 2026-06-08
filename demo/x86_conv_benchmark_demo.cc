#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/operator/conv2d_op.h"
#include "src/kernel/x86/direct_conv_fp32.h"
#include "src/kernel/x86/pointwise_conv_fp32.h"
#include "util/fp16.h"

namespace {

std::vector<float> MakeTensor(int64_t size, float seed) {
    std::vector<float> values(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        values[static_cast<size_t>(i)] = seed + static_cast<float>(((i * 17 + 11) % 29) - 14) * 0.0625f;
    }
    return values;
}

std::vector<uint16_t> MakeHalfTensor(int64_t size, float seed) {
    std::vector<uint16_t> values(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        const float value = seed + static_cast<float>(((i * 19 + 7) % 31) - 15) * 0.0625f;
        values[static_cast<size_t>(i)] = feather::FloatToHalf(value);
    }
    return values;
}

void BenchmarkPointwiseConv1x1(benchmark::State& state) {
    const int64_t batch = state.range(0);
    const int64_t in_c = state.range(1);
    const int64_t in_h = state.range(2);
    const int64_t in_w = state.range(3);
    const int64_t out_c = state.range(4);
    const int64_t stride = state.range(5);
    const int64_t out_h = (in_h - 1) / stride + 1;
    const int64_t out_w = (in_w - 1) / stride + 1;

    const std::vector<float> input = MakeTensor(batch * in_c * in_h * in_w, 1.0f);
    const std::vector<float> weight = MakeTensor(out_c * in_c, 0.25f);
    const std::vector<float> bias = MakeTensor(out_c, -0.5f);
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputePointwiseConv2DX86Fp32(
            input.data(), weight.data(), bias.data(), batch, in_c, in_h, in_w, out_c, stride, stride, output.data());
        if (status != 0) {
            state.SkipWithError("pointwise conv kernel failed");
            return;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * batch * out_c * out_h * out_w * in_c);
}

void BenchmarkPointwiseConv1x1Fp16(benchmark::State& state) {
    const int64_t batch = state.range(0);
    const int64_t in_c = state.range(1);
    const int64_t in_h = state.range(2);
    const int64_t in_w = state.range(3);
    const int64_t out_c = state.range(4);
    const int64_t stride = state.range(5);
    const int64_t out_h = (in_h - 1) / stride + 1;
    const int64_t out_w = (in_w - 1) / stride + 1;

    auto input = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> input_data = MakeHalfTensor(batch * in_c * in_h * in_w, 1.0f);
    input->Assign<uint16_t>(input_data, {batch, in_c, in_h, in_w});

    auto weight = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> weight_data = MakeHalfTensor(out_c * in_c, -0.25f);
    weight->Assign<uint16_t>(weight_data, {out_c, in_c, 1, 1});

    auto bias = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> bias_data = MakeHalfTensor(out_c, 0.125f);
    bias->Assign<uint16_t>(bias_data, {out_c});

    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{batch, out_c, out_h, out_w});
    output->mutable_data<uint16_t>();

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    param.stride_h = static_cast<int32_t>(stride);
    param.stride_w = static_cast<int32_t>(stride);
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;

    auto kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    if (kernel == nullptr) {
        state.SkipWithError("failed to create x86 fp16 conv kernel");
        return;
    }
    kernel->SetParam(&param);

    if (kernel->compute() != 0) {
        state.SkipWithError("fp16 pointwise conv warmup failed");
        return;
    }

    for (auto _ : state) {
        if (kernel->compute() != 0) {
            state.SkipWithError("fp16 pointwise conv failed");
            return;
        }
        benchmark::DoNotOptimize(output->raw_data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * batch * out_c * out_h * out_w * in_c);
}

void BenchmarkDirectConv3x3(benchmark::State& state) {
    const int64_t batch = state.range(0);
    const int64_t in_c = state.range(1);
    const int64_t in_h = state.range(2);
    const int64_t in_w = state.range(3);
    const int64_t out_c = state.range(4);
    const int64_t stride = state.range(5);
    const int64_t pad = state.range(6);
    const int64_t kernel_h = 3;
    const int64_t kernel_w = 3;
    const int64_t out_h = (in_h + 2 * pad - kernel_h) / stride + 1;
    const int64_t out_w = (in_w + 2 * pad - kernel_w) / stride + 1;

    const std::vector<float> input = MakeTensor(batch * in_c * in_h * in_w, 0.75f);
    const std::vector<float> weight = MakeTensor(out_c * in_c * kernel_h * kernel_w, -0.25f);
    const std::vector<float> bias = MakeTensor(out_c, 0.125f);
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    for (auto _ : state) {
        const int32_t status = feather::kernel::x86::ComputeDirectConv2DX86Fp32(
            input.data(), weight.data(), bias.data(), batch, in_c, in_h, in_w, out_c, kernel_h, kernel_w, stride,
            stride, pad, pad, output.data());
        if (status != 0) {
            state.SkipWithError("direct conv kernel failed");
            return;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * batch * out_c * out_h * out_w * in_c * kernel_h * kernel_w);
}

void BenchmarkDirectConv3x3Fp16(benchmark::State& state) {
    const int64_t batch = state.range(0);
    const int64_t in_c = state.range(1);
    const int64_t in_h = state.range(2);
    const int64_t in_w = state.range(3);
    const int64_t out_c = state.range(4);
    const int64_t stride = state.range(5);
    const int64_t pad = state.range(6);
    const int64_t kernel_h = 3;
    const int64_t kernel_w = 3;
    const int64_t out_h = (in_h + 2 * pad - kernel_h) / stride + 1;
    const int64_t out_w = (in_w + 2 * pad - kernel_w) / stride + 1;

    auto input = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> input_data = MakeHalfTensor(batch * in_c * in_h * in_w, 0.75f);
    input->Assign<uint16_t>(input_data, {batch, in_c, in_h, in_w});

    auto weight = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> weight_data = MakeHalfTensor(out_c * in_c * kernel_h * kernel_w, -0.25f);
    weight->Assign<uint16_t>(weight_data, {out_c, in_c, kernel_h, kernel_w});

    auto bias = std::make_shared<feather::Tensor>();
    const std::vector<uint16_t> bias_data = MakeHalfTensor(out_c, 0.125f);
    bias->Assign<uint16_t>(bias_data, {out_c});

    auto output = std::make_shared<feather::Tensor>(std::vector<int64_t>{batch, out_c, out_h, out_w});
    output->mutable_data<uint16_t>();

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = output;
    param.stride_h = static_cast<int32_t>(stride);
    param.stride_w = static_cast<int32_t>(stride);
    param.pad_h = static_cast<int32_t>(pad);
    param.pad_w = static_cast<int32_t>(pad);
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;

    auto kernel = feather::KernelDispatcher::instance().create(
        feather::DeviceType::X86, feather::DataType::FP16, "Conv2D");
    if (kernel == nullptr) {
        state.SkipWithError("failed to create x86 fp16 conv kernel");
        return;
    }
    kernel->SetParam(&param);

    if (kernel->compute() != 0) {
        state.SkipWithError("fp16 direct conv warmup failed");
        return;
    }

    for (auto _ : state) {
        if (kernel->compute() != 0) {
            state.SkipWithError("fp16 direct conv failed");
            return;
        }
        benchmark::DoNotOptimize(output->raw_data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * batch * out_c * out_h * out_w * in_c * kernel_h * kernel_w);
}

BENCHMARK(BenchmarkPointwiseConv1x1)
    ->Args({1, 32, 80, 80, 32, 1})
    ->Args({1, 64, 80, 80, 64, 1})
    ->Args({1, 128, 40, 40, 128, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BenchmarkPointwiseConv1x1Fp16)
    ->Args({1, 32, 80, 80, 32, 1})
    ->Args({1, 64, 80, 80, 64, 1})
    ->Args({1, 128, 40, 40, 128, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BenchmarkDirectConv3x3)
    ->Args({1, 32, 80, 80, 32, 1, 1})
    ->Args({1, 64, 40, 40, 64, 1, 1})
    ->Args({1, 128, 20, 20, 128, 1, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BenchmarkDirectConv3x3Fp16)
    ->Args({1, 32, 80, 80, 32, 1, 1})
    ->Args({1, 64, 40, 40, 64, 1, 1})
    ->Args({1, 128, 20, 20, 128, 1, 1})
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
