// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "demo/yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <variant>

#include "util/fp16.h"

namespace feather {
namespace demo {
namespace {
bool Fail(const std::string& message, std::string* error) {
    if (error) *error = message;
    return false;
}

template <typename T>
const T* MetadataValue(const model::ModelDesc& model, const std::string& key) {
    const auto it = model.metadata.find(key);
    return it == model.metadata.end() ? nullptr : std::get_if<T>(&it->second);
}

float Read(const Tensor& tensor, int64_t index) {
    return tensor.data_type() == DataType::FP16 ? HalfToFloat(tensor.data<uint16_t>()[index])
                                               : tensor.data<float>()[index];
}

double IoU(const Detection& a, const Detection& b) {
    const double w = std::max(0.0, static_cast<double>(std::min(a.x2, b.x2)) - std::max(a.x1, b.x1));
    const double h = std::max(0.0, static_cast<double>(std::min(a.y2, b.y2)) - std::max(a.y1, b.y1));
    const double intersection = w * h;
    const double area_a = (static_cast<double>(a.x2) - a.x1) * (static_cast<double>(a.y2) - a.y1);
    const double area_b = (static_cast<double>(b.x2) - b.x1) * (static_cast<double>(b.y2) - b.y1);
    const double union_area = area_a + area_b - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}
}  // namespace

bool ParseYoloMetadata(const model::ModelDesc& model, YoloMetadata* metadata, std::string* error) {
    if (!metadata) return Fail("YOLO metadata destination is null", error);
    *metadata = YoloMetadata{};
    if (error) error->clear();
    bool has_yolo_metadata = false;
    for (const auto& entry : model.metadata) {
        if (entry.first.compare(0, 5, "yolo.") == 0) has_yolo_metadata = true;
    }
    if (!has_yolo_metadata) return true;
    const auto* version = MetadataValue<int64_t>(model, "yolo.version");
    const auto* family = MetadataValue<std::string>(model, "yolo.family");
    const auto* classes = MetadataValue<int64_t>(model, "yolo.num_classes");
    const auto* format = MetadataValue<std::string>(model, "yolo.output_format");
    const auto* layout = MetadataValue<std::string>(model, "yolo.output_layout");
    const auto* nms = MetadataValue<int64_t>(model, "yolo.nms_required");
    if (!version || *version != 1 || !family || !classes || !format || !layout || !nms) {
        return Fail("YOLO metadata requires version 1 and typed family, num_classes, output_format, output_layout, nms_required", error);
    }
    if (*family != "yolov6" && *family != "yolov8" && *family != "yolov10" &&
        *family != "yolo11" && *family != "yolo26") {
        return Fail("unsupported YOLO family: " + *family, error);
    }
    if (*classes <= 0 || *classes > std::numeric_limits<int>::max() - 5) {
        return Fail("YOLO num_classes must be positive and fit an integer class ID", error);
    }
    if (*format != "cxcywh_objectness" && *format != "cxcywh_scores" && *format != "xyxy_score_class") {
        return Fail("unsupported YOLO output_format: " + *format, error);
    }
    if (*layout != "bnc" && *layout != "bcn") return Fail("YOLO output_layout must be bnc or bcn", error);
    if (*nms != 0 && *nms != 1) return Fail("YOLO nms_required must be 0 or 1", error);
    if (*format == "xyxy_score_class" && *nms != 0) {
        return Fail("YOLO end-to-end output must disable redundant NMS", error);
    }
    metadata->legacy_yolov5 = false;
    metadata->family = *family;
    metadata->num_classes = *classes;
    metadata->output_format = *format;
    metadata->output_layout = *layout;
    metadata->nms_required = *nms != 0;
    return true;
}

bool ValidateYoloOutput(const std::vector<int64_t>& dims, DataType dtype,
                        const YoloMetadata& metadata, std::string* error) {
    if (dtype != DataType::FP32 && dtype != DataType::FP16) return Fail("YOLO output must be FP32 or FP16", error);
    if (dims.size() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 ||
        dims[1] > std::numeric_limits<int64_t>::max() / dims[2]) {
        return Fail("YOLO output must have a static batch-one rank-three shape", error);
    }
    const int64_t channels = metadata.output_layout == "bcn" ? dims[1] : dims[2];
    if (metadata.legacy_yolov5) {
        return channels >= 6 || Fail("legacy YOLOv5 output must have shape [1,N,5+classes]", error);
    }
    const int64_t expected = metadata.output_format == "xyxy_score_class" ? 6 :
                            metadata.num_classes + (metadata.output_format == "cxcywh_objectness" ? 5 : 4);
    return channels == expected || Fail("YOLO output channel count disagrees with metadata: expected " +
                                        std::to_string(expected) + ", received " + std::to_string(channels), error);
}

bool DecodeYoloDetections(const Tensor& output, const YoloMetadata& metadata,
                          const LetterboxInfo& letterbox, int image_width, int image_height,
                          float conf_thresh, float iou_thresh, size_t max_detections,
                          std::vector<Detection>* detections, std::string* error) {
    if (error) error->clear();
    if (!detections) return Fail("YOLO detection destination is null", error);
    if (!std::isfinite(conf_thresh) || conf_thresh < 0 || conf_thresh > 1 ||
        !std::isfinite(iou_thresh) || iou_thresh < 0 || iou_thresh > 1 || max_detections == 0) {
        return Fail("YOLO thresholds must be in [0,1] and max_detections must be positive", error);
    }
    if (image_width <= 0 || image_height <= 0 || !std::isfinite(letterbox.scale) || letterbox.scale <= 0) {
        return Fail("YOLO image dimensions and letterbox scale must be positive", error);
    }
    if (!output.IsInitialized()) return Fail("YOLO output tensor is not initialized", error);
    if (!ValidateYoloOutput(output.dims().data(), output.data_type(), metadata, error)) return false;
    if (metadata.legacy_yolov5) {
        *detections = DecodeYolov5Detections(output, letterbox, image_width, image_height, conf_thresh, iou_thresh);
        if (detections->size() > max_detections) detections->resize(max_detections);
        return true;
    }
    const bool transposed = metadata.output_layout == "bcn";
    const auto& dims = output.dims().data();
    const int64_t boxes = dims[transposed ? 2 : 1];
    const int64_t channels = dims[transposed ? 1 : 2];
    const auto read = [&](int64_t box, int64_t channel) {
        return Read(output, transposed ? channel * boxes + box : box * channels + channel);
    };
    std::vector<Detection> candidates;
    for (int64_t box = 0; box < boxes; ++box) {
        Detection det{};
        float x = read(box, 0), y = read(box, 1), w = read(box, 2), h = read(box, 3);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h)) continue;
        if (metadata.output_format == "xyxy_score_class") {
            det.score = read(box, 4);
            const float cls = read(box, 5);
            if (!std::isfinite(cls) || cls < 0 || static_cast<double>(cls) >= metadata.num_classes || std::floor(cls) != cls) continue;
            det.class_id = static_cast<int>(cls);
            det.x1 = x; det.y1 = y; det.x2 = w; det.y2 = h;
        } else {
            if (w <= 0 || h <= 0) continue;
            const bool objectness = metadata.output_format == "cxcywh_objectness";
            const float obj = objectness ? read(box, 4) : 1.0f;
            if (!std::isfinite(obj) || obj < 0) continue;
            float best_score = -std::numeric_limits<float>::infinity();
            int best_class = -1;
            bool finite_scores = true;
            for (int64_t cls = 0; cls < metadata.num_classes; ++cls) {
                const float score = read(box, cls + (objectness ? 5 : 4));
                if (!std::isfinite(score)) { finite_scores = false; break; }
                if (score > best_score) { best_score = score; best_class = static_cast<int>(cls); }
            }
            if (!finite_scores || best_class < 0 || best_score < 0) continue;
            det.class_id = best_class;
            det.score = obj * best_score;
            det.x1 = x - w * 0.5f; det.y1 = y - h * 0.5f;
            det.x2 = x + w * 0.5f; det.y2 = y + h * 0.5f;
        }
        if (!std::isfinite(det.score) || det.score < conf_thresh ||
            !std::isfinite(det.x1) || !std::isfinite(det.y1) || !std::isfinite(det.x2) || !std::isfinite(det.y2) ||
            det.x2 <= det.x1 || det.y2 <= det.y1) continue;
        candidates.push_back(det);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<Detection> result;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        if (metadata.nms_required) {
            for (const auto& previous : kept) {
                if (candidate.class_id == previous.class_id && IoU(candidate, previous) > iou_thresh) { suppressed = true; break; }
            }
        }
        if (suppressed) continue;
        kept.push_back(candidate);
        Detection det = candidate;
        det.x1 = static_cast<float>(std::clamp((static_cast<double>(det.x1) - letterbox.pad_x) / letterbox.scale, 0.0, static_cast<double>(image_width)));
        det.y1 = static_cast<float>(std::clamp((static_cast<double>(det.y1) - letterbox.pad_y) / letterbox.scale, 0.0, static_cast<double>(image_height)));
        det.x2 = static_cast<float>(std::clamp((static_cast<double>(det.x2) - letterbox.pad_x) / letterbox.scale, 0.0, static_cast<double>(image_width)));
        det.y2 = static_cast<float>(std::clamp((static_cast<double>(det.y2) - letterbox.pad_y) / letterbox.scale, 0.0, static_cast<double>(image_height)));
        if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;
        result.push_back(det);
        if (result.size() == max_detections) break;
    }
    *detections = std::move(result);
    return true;
}
}  // namespace demo
}  // namespace feather
