#include <benchmark/benchmark.h>

#include <iostream>
#include <string>
#include <vector>

#include "demo/image_io.h"
#include "demo/yolov5_benchmark_cli.h"
#include "demo/yolov5_runner.h"

namespace {

struct BenchmarkContext {
    feather::demo::Yolov5Runner runner;
    feather::demo::ImageData image;
    std::vector<feather::demo::Detection> detections;
    float conf_thresh{0.25f};
    float iou_thresh{0.45f};
};

BenchmarkContext* g_context = nullptr;

void Yolov5Benchmark(benchmark::State& state) {
    if (g_context == nullptr) {
        state.SkipWithError("benchmark context not initialized");
        return;
    }

    for (auto _ : state) {
        const int32_t status = g_context->runner.RunPreparedImage(
            g_context->image, g_context->conf_thresh, g_context->iou_thresh, &g_context->detections);
        if (status != 0) {
            state.SkipWithError("inference failed");
            return;
        }
        benchmark::DoNotOptimize(g_context->detections);
    }
}

void PrintUsage() {
    std::cerr << "usage: yolov5_benchmark_demo --model <model.fth> --image <image> "
                 "[--conf-thresh 0.25] [--iou-thresh 0.45] <google-benchmark-args>\n";
}

}  // namespace

int main(int argc, char** argv) {
    feather::demo::Yolov5BenchmarkCommandLine options;
    if (!feather::demo::ParseYolov5BenchmarkCommandLine(&argc, &argv, &options)) {
        PrintUsage();
        return 1;
    }

    benchmark::Initialize(&argc, argv);

    static BenchmarkContext context;
    context.conf_thresh = options.conf_thresh;
    context.iou_thresh = options.iou_thresh;

    if (context.runner.Load(options.model_path) != 0) {
        std::cerr << "failed to load model: " << options.model_path << '\n';
        return 1;
    }
    if (feather::demo::LoadImage(options.image_path, &context.image) != 0) {
        std::cerr << "failed to load image: " << options.image_path << '\n';
        return 1;
    }

    g_context = &context;
    benchmark::RegisterBenchmark("yolov5/infer", Yolov5Benchmark)
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
