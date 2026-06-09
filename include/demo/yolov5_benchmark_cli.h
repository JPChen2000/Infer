#ifndef FEATHER_DEMO_YOLOV5_BENCHMARK_CLI_H
#define FEATHER_DEMO_YOLOV5_BENCHMARK_CLI_H

#include <cstdlib>
#include <string>

#include "demo/yolov5_runner.h"

namespace feather {
namespace demo {

struct Yolov5BenchmarkCommandLine {
    std::string model_path;
    std::string image_path;
    Yolov5Backend backend{Yolov5Backend::kHost};
    Yolov5LayoutOverride layout{Yolov5LayoutOverride::kAuto};
    float conf_thresh{0.25f};
    float iou_thresh{0.45f};
};

inline bool ParseYolov5BenchmarkCommandLine(int* argc, char*** argv, Yolov5BenchmarkCommandLine* result) {
    if (result == nullptr) {
        return false;
    }
    if (argc == nullptr || argv == nullptr || *argv == nullptr) {
        return false;
    }

    *result = Yolov5BenchmarkCommandLine();
    int write_index = 1;
    for (int read_index = 1; read_index < *argc; ++read_index) {
        const std::string arg = (*argv)[read_index];
        if (arg == "--model" && read_index + 1 < *argc) {
            result->model_path = (*argv)[++read_index];
            continue;
        }
        if (arg == "--image" && read_index + 1 < *argc) {
            result->image_path = (*argv)[++read_index];
            continue;
        }
        if (arg == "--backend" && read_index + 1 < *argc) {
            if (!ParseYolov5Backend((*argv)[++read_index], &result->backend)) {
                return false;
            }
            continue;
        }
        if (arg == "--layout" && read_index + 1 < *argc) {
            if (!ParseYolov5LayoutOverride((*argv)[++read_index], &result->layout)) {
                return false;
            }
            continue;
        }
        if (arg == "--conf-thresh" && read_index + 1 < *argc) {
            result->conf_thresh = std::strtof((*argv)[++read_index], nullptr);
            continue;
        }
        if (arg == "--iou-thresh" && read_index + 1 < *argc) {
            result->iou_thresh = std::strtof((*argv)[++read_index], nullptr);
            continue;
        }
        (*argv)[write_index++] = (*argv)[read_index];
    }
    (*argv)[write_index] = nullptr;
    *argc = write_index;
    return !result->model_path.empty() && !result->image_path.empty();
}

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_YOLOV5_BENCHMARK_CLI_H
