// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_YOLO_RUNNER_H
#define FEATHER_DEMO_YOLO_RUNNER_H
#include "demo/yolov5_runner.h"
namespace feather {
namespace demo {
using YoloRunner = Yolov5Runner;
using YoloBackend = Yolov5Backend;
using YoloLayoutOverride = Yolov5LayoutOverride;
inline bool ParseYoloBackend(const std::string& value, YoloBackend* backend) { return ParseYolov5Backend(value, backend); }
inline const char* YoloBackendName(YoloBackend backend) { return Yolov5BackendName(backend); }
inline bool ParseYoloLayoutOverride(const std::string& value, YoloLayoutOverride* layout) { return ParseYolov5LayoutOverride(value, layout); }
inline const char* YoloLayoutOverrideName(YoloLayoutOverride layout) { return Yolov5LayoutOverrideName(layout); }
}  // namespace demo
}  // namespace feather
#endif
