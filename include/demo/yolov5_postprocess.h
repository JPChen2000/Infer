#ifndef FEATHER_DEMO_YOLOV5_POSTPROCESS_H
#define FEATHER_DEMO_YOLOV5_POSTPROCESS_H

#include <vector>

#include "core/tensor.h"
#include "demo/detection.h"
#include "demo/image_io.h"

namespace feather {
namespace demo {

std::vector<Detection> DecodeYolov5Detections(const Tensor& output, const LetterboxInfo& letterbox,
                                              int image_width, int image_height,
                                              float conf_thresh, float iou_thresh);

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_YOLOV5_POSTPROCESS_H
