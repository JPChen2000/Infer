#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "demo/image_io.h"
#include "demo/yolov5_runner.h"

namespace {

void PrintUsage() {
    std::cerr << "usage: yolov5_demo --model <model.fth> --image <image> "
                 "[--backend host|common|x86|cuda] [--conf-thresh 0.25] [--iou-thresh 0.45] [--profile]\n";
}

void PrintRuntimeProfile(const feather::demo::Yolov5Runner& runner, size_t limit) {
    auto summaries = runner.RuntimeProfileSummaries();
    std::sort(summaries.begin(), summaries.end(),
              [](const feather::RuntimeProfileSummary& lhs, const feather::RuntimeProfileSummary& rhs) {
                  return lhs.total_ms > rhs.total_ms;
              });
    const size_t count = std::min(limit, summaries.size());
    std::cout << "[profile] top_runtime_nodes=" << count << '\n';
    for (size_t i = 0; i < count; ++i) {
        const auto& item = summaries[i];
        std::cout << "[profile] #" << (i + 1)
                  << " node=" << item.node_name
                  << " op=" << item.op_type
                  << " calls=" << item.call_count
                  << " total_ms=" << item.total_ms
                  << " avg_ms=" << item.avg_ms << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string image_path;
    std::string output_path = "output.jpg";
    feather::demo::Yolov5Backend backend = feather::demo::Yolov5Backend::kHost;
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;
    bool profile = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            if (!feather::demo::ParseYolov5Backend(argv[++i], &backend)) {
                std::cerr << "invalid backend: " << argv[i] << '\n';
                PrintUsage();
                return 1;
            }
        } else if (arg == "--conf-thresh" && i + 1 < argc) {
            conf_thresh = std::strtof(argv[++i], nullptr);
        } else if (arg == "--iou-thresh" && i + 1 < argc) {
            iou_thresh = std::strtof(argv[++i], nullptr);
        } else if (arg == "--profile") {
            profile = true;
        } else {
            PrintUsage();
            return 1;
        }
    }

    if (model_path.empty() || image_path.empty()) {
        PrintUsage();
        return 1;
    }

    feather::demo::Yolov5Runner runner;
    if (runner.Load(model_path, backend) != 0) {
        std::cerr << "failed to load model: " << model_path << '\n';
        return 1;
    }
    runner.SetRuntimeProfilingEnabled(profile);
    std::cout << "[framework] " << runner.DescribeLastBuild() << '\n';

    std::vector<feather::demo::Detection> detections;
    if (runner.Run(image_path, conf_thresh, iou_thresh, &detections) != 0) {
        std::cerr << "failed to run inference for image: " << image_path << '\n';
        return 1;
    }
    std::cout << "[framework] " << runner.DescribeLastRun() << '\n';
    if (profile) {
        PrintRuntimeProfile(runner, 20);
    }

    std::cout << "detections: " << detections.size() << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& det : detections) {
        std::cout << "class=" << det.class_id
                  << " score=" << det.score
                  << " box=[" << det.x1 << ", " << det.y1
                  << ", " << det.x2 << ", " << det.y2 << "]\n";
    }

    feather::demo::ImageData image;
    const auto save_begin = std::chrono::steady_clock::now();
    if (feather::demo::LoadImage(image_path, &image) != 0) {
        std::cerr << "failed to reload image for annotation: " << image_path << '\n';
        return 1;
    }
    if (feather::demo::SaveDetectionsImage(image, detections, output_path) != 0) {
        std::cerr << "failed to save annotated image: " << output_path << '\n';
        return 1;
    }
    const auto save_end = std::chrono::steady_clock::now();
    const double save_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(save_end - save_begin).count();
    std::cout << "[framework] save_ms=" << save_ms << '\n';
    std::cout << "saved annotated image: " << output_path << '\n';
    return 0;
}
