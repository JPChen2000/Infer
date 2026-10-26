// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_DEMO_YOLO_POSTPROCESS_H
#define FEATHER_DEMO_YOLO_POSTPROCESS_H

#include <cstddef>
#include <string>
#include <vector>

#include "demo/yolov5_postprocess.h"
#include "model/model_format.h"

namespace feather {
namespace demo {

struct YoloMetadata {
    bool legacy_yolov5{true};
    std::string family{"yolov5"};
    int64_t num_classes{0};
    std::string output_format{"cxcywh_objectness"};
    std::string output_layout{"bnc"};
    bool nms_required{true};
};

bool ParseYoloMetadata(const model::ModelDesc& model, YoloMetadata* metadata, std::string* error);
bool ValidateYoloOutput(const std::vector<int64_t>& dims, DataType dtype,
                        const YoloMetadata& metadata, std::string* error);
bool DecodeYoloDetections(const Tensor& output, const YoloMetadata& metadata,
                          const LetterboxInfo& letterbox, int image_width, int image_height,
                          float conf_thresh, float iou_thresh, size_t max_detections,
                          std::vector<Detection>* detections, std::string* error);
#ifdef FEATHER_WITH_CUDA
bool DecodeYoloDetectionsCuda(Tensor& output, const YoloMetadata& metadata,
                              const LetterboxInfo& letterbox, int image_width, int image_height,
                              float conf_thresh, float iou_thresh, size_t max_detections,
                              std::vector<Detection>* detections, std::string* error);
#endif

}  // namespace demo
}  // namespace feather
#endif
