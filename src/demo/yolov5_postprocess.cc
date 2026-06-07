#include "demo/yolov5_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "util/fp16.h"

namespace feather {
namespace demo {

namespace {

float ReadTensorValue(const Tensor& tensor, int64_t index) {
    if (tensor.data_type() == DataType::FP16) {
        return HalfToFloat(tensor.data<uint16_t>()[index]);
    }
    return tensor.data<float>()[index];
}

float ComputeIoU(const Detection& lhs, const Detection& rhs) {
    const float inter_x1 = std::max(lhs.x1, rhs.x1);
    const float inter_y1 = std::max(lhs.y1, rhs.y1);
    const float inter_x2 = std::min(lhs.x2, rhs.x2);
    const float inter_y2 = std::min(lhs.y2, rhs.y2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float lhs_area = std::max(0.0f, lhs.x2 - lhs.x1) * std::max(0.0f, lhs.y2 - lhs.y1);
    const float rhs_area = std::max(0.0f, rhs.x2 - rhs.x1) * std::max(0.0f, rhs.y2 - rhs.y1);
    const float union_area = lhs_area + rhs_area - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

}  // namespace

std::vector<Detection> DecodeYolov5Detections(const Tensor& output, const LetterboxInfo& letterbox,
                                              int image_width, int image_height,
                                              float conf_thresh, float iou_thresh) {
    std::vector<Detection> candidates;
    const auto& dims = output.dims().data();
    if (dims.size() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] < 6) {
        return candidates;
    }

    const int64_t num_boxes = dims[1];
    const int64_t stride = dims[2];
    const int64_t num_classes = stride - 5;
    for (int64_t i = 0; i < num_boxes; ++i) {
        const int64_t base = i * stride;
        const float obj = ReadTensorValue(output, base + 4);
        if (obj <= 0.0f) {
            continue;
        }

        int best_class = -1;
        float best_class_score = 0.0f;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float cls = ReadTensorValue(output, base + 5 + c);
            if (cls > best_class_score) {
                best_class_score = cls;
                best_class = static_cast<int>(c);
            }
        }
        const float score = obj * best_class_score;
        if (best_class < 0 || score < conf_thresh) {
            continue;
        }

        const float cx = ReadTensorValue(output, base + 0);
        const float cy = ReadTensorValue(output, base + 1);
        const float w = ReadTensorValue(output, base + 2);
        const float h = ReadTensorValue(output, base + 3);
        const float x1 = (cx - w * 0.5f - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
        const float y1 = (cy - h * 0.5f - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
        const float x2 = (cx + w * 0.5f - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
        const float y2 = (cy + h * 0.5f - static_cast<float>(letterbox.pad_y)) / letterbox.scale;

        Detection det;
        det.class_id = best_class;
        det.score = score;
        det.x1 = std::clamp(x1, 0.0f, static_cast<float>(image_width));
        det.y1 = std::clamp(y1, 0.0f, static_cast<float>(image_height));
        det.x2 = std::clamp(x2, 0.0f, static_cast<float>(image_width));
        det.y2 = std::clamp(y2, 0.0f, static_cast<float>(image_height));
        candidates.push_back(det);
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& lhs, const Detection& rhs) { return lhs.score > rhs.score; });

    std::vector<Detection> detections;
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        detections.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[i].class_id != candidates[j].class_id) {
                continue;
            }
            if (ComputeIoU(candidates[i], candidates[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return detections;
}

}  // namespace demo
}  // namespace feather
