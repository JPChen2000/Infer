// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "demo/yolo_postprocess.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <math_constants.h>

#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/cuda/nms.h"

namespace feather {
namespace demo {
namespace {
namespace cd = kernel::cuda_detail;
class Workspace {
 public:
    ~Workspace() { cd::ReleaseTemporaryDeviceBuffer(data_, bytes_); }
    bool Allocate(int64_t count, size_t item_size) {
        return cd::SafeByteCount(count, item_size, &bytes_) &&
               cd::AcquireTemporaryDeviceBuffer(bytes_, &data_) == 0;
    }
    template <typename T> T* As() const { return static_cast<T*>(data_); }
 private:
    void* data_{nullptr};
    size_t bytes_{0};
};

template <typename T>
__device__ float Read(const T* data, int64_t box, int64_t channel, int64_t boxes, int64_t channels, bool transposed) {
    return cd::ReadDevice(data, transposed ? channel * boxes + box : box * channels + channel);
}

template <typename T>
__global__ void DecodeCuda(const T* raw, float* coordinates, float* scores, int64_t* labels,
                            int64_t boxes, int64_t channels, int64_t classes, bool transposed,
                            bool objectness, bool legacy, float confidence, LetterboxInfo letterbox,
                            int image_width, int image_height) {
    for (int64_t box = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         box < boxes; box += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        scores[box] = -CUDART_INF_F;
        labels[box] = -1;
        for (int j = 0; j < 4; ++j) coordinates[box * 4 + j] = 0;
        const float x = Read(raw, box, 0, boxes, channels, transposed);
        const float y = Read(raw, box, 1, boxes, channels, transposed);
        const float w = Read(raw, box, 2, boxes, channels, transposed);
        const float h = Read(raw, box, 3, boxes, channels, transposed);
        if (!isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h) || (!legacy && (w <= 0 || h <= 0))) continue;
        const float obj = objectness ? Read(raw, box, 4, boxes, channels, transposed) : 1.0f;
        if (isnan(obj) || (!legacy && !isfinite(obj)) || obj < 0 || (legacy && obj <= 0)) continue;
        float best_score = legacy ? 0.0f : -CUDART_INF_F;
        int64_t best_class = -1;
        bool finite = true;
        for (int64_t cls = 0; cls < classes; ++cls) {
            const float value = Read(raw, box, cls + (objectness ? 5 : 4), boxes, channels, transposed);
            // Legacy best-class selection ignores NaN in comparisons.
            if (!legacy && !isfinite(value)) { finite = false; break; }
            if (value > best_score) { best_class = cls; best_score = value; }
        }
        const float score = obj * best_score;
        if (!finite || best_class < 0 || best_score < 0 ||
            (legacy ? isnan(score) : !isfinite(score)) || score < confidence) continue;
        float x1 = __fsub_rn(x, __fmul_rn(w, 0.5f)), y1 = __fsub_rn(y, __fmul_rn(h, 0.5f));
        float x2 = __fadd_rn(x, __fmul_rn(w, 0.5f)), y2 = __fadd_rn(y, __fmul_rn(h, 0.5f));
        if (!legacy && (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2) ||
                        x2 <= x1 || y2 <= y1)) continue;
        if (legacy) {
            // YOLOv5's established contract applies NMS after coordinate
            // restoration/clipping. New YOLO heads apply it in model space.
            x1 = fminf(fmaxf(__fsub_rn(x1, static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0f), static_cast<float>(image_width));
            y1 = fminf(fmaxf(__fsub_rn(y1, static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0f), static_cast<float>(image_height));
            x2 = fminf(fmaxf(__fsub_rn(x2, static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0f), static_cast<float>(image_width));
            y2 = fminf(fmaxf(__fsub_rn(y2, static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0f), static_cast<float>(image_height));
        }
        coordinates[box * 4] = x1; coordinates[box * 4 + 1] = y1;
        coordinates[box * 4 + 2] = x2; coordinates[box * 4 + 3] = y2;
        scores[box] = score;
        labels[box] = best_class;
    }
}

__global__ void GatherDetections(const float* coordinates, const float* scores, const int64_t* labels,
                                 const int64_t* triples, Detection* detections, int64_t count) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t index = triples[i * 3 + 2];
        Detection value;
        value.class_id = static_cast<int>(labels[index]); value.score = scores[index];
        value.x1 = coordinates[index * 4]; value.y1 = coordinates[index * 4 + 1];
        value.x2 = coordinates[index * 4 + 2]; value.y2 = coordinates[index * 4 + 3];
        detections[i] = value;
    }
}

bool Fail(const char* message, std::string* error) {
    if (error) *error = message;
    return false;
}
}  // namespace

bool DecodeYoloDetectionsCuda(Tensor& output, const YoloMetadata& metadata,
                              const LetterboxInfo& letterbox, int image_width, int image_height,
                              float confidence, float iou, size_t maximum,
                              std::vector<Detection>* detections, std::string* error) {
    // The graph intentionally leaves its output on the device. Preserve it
    // when acquiring inputs outside RuntimeGraph's deferred-sync scope.
    cd::DeferredHostSyncScope deferred_sync;
    if (!detections || !metadata.nms_required || !output.IsInitialized() ||
        !ValidateYoloOutput(output.dims().data(), output.data_type(), metadata, error)) return false;
    if (image_width <= 0 || image_height <= 0 || !std::isfinite(letterbox.scale) || letterbox.scale <= 0 ||
        !std::isfinite(confidence) || confidence < 0 || confidence > 1 || !std::isfinite(iou) || iou < 0 || iou > 1 || !maximum)
        return Fail("invalid CUDA YOLO postprocessing parameters", error);
    const bool transposed = metadata.output_layout == "bcn";
    const int64_t boxes = output.dims()[transposed ? 2 : 1], channels = output.dims()[transposed ? 1 : 2];
    if (boxes > INT_MAX) return Fail("CUDA YOLO NMS requires at most INT_MAX boxes", error);
    const int64_t classes = metadata.legacy_yolov5 ? channels - 5 : metadata.num_classes;
    const bool objectness = metadata.output_format == "cxcywh_objectness";
    Workspace coordinates, scores, labels;
    if (!coordinates.Allocate(boxes, 4 * sizeof(float)) || !scores.Allocate(boxes, sizeof(float)) ||
        !labels.Allocate(boxes, sizeof(int64_t))) return Fail("CUDA YOLO candidate allocation failed", error);
    const int blocks = static_cast<int>(std::min<int64_t>(65535, cd::DivUp(boxes, cd::kCudaThreads)));
    if (output.data_type() == DataType::FP32) {
        cd::DeviceBuffer<float> raw;
        if (cd::CopyTensorToDevice(&output, &raw) != 0) return Fail("CUDA YOLO output access failed", error);
        DecodeCuda<<<blocks, cd::kCudaThreads, 0, cd::InferenceStream()>>>(raw.get(), coordinates.As<float>(),
            scores.As<float>(), labels.As<int64_t>(), boxes, channels, classes, transposed, objectness,
            metadata.legacy_yolov5, confidence, letterbox, image_width, image_height);
    } else {
        cd::DeviceBuffer<uint16_t> raw;
        if (cd::CopyTensorToDevice(&output, &raw) != 0) return Fail("CUDA YOLO output access failed", error);
        DecodeCuda<<<blocks, cd::kCudaThreads, 0, cd::InferenceStream()>>>(raw.get(), coordinates.As<float>(),
            scores.As<float>(), labels.As<int64_t>(), boxes, channels, classes, transposed, objectness,
            metadata.legacy_yolov5, confidence, letterbox, image_width, image_height);
    }
    if (cd::CudaCheck(cudaGetLastError()) != 0) return Fail("CUDA YOLO decode launch failed", error);
    cd::CudaNmsOptions options;
    options.boxes = boxes; options.max_per_class = boxes;
    options.score_threshold = confidence; options.iou_threshold = iou;
    options.score_inclusive = true; options.double_iou = !metadata.legacy_yolov5;
    options.preserve_corner_order = true;
    std::vector<int64_t> selected;
    if (cd::SelectNmsOnCuda(coordinates.As<float>(), scores.As<float>(), labels.As<int64_t>(), options, &selected) != 0)
        return Fail("CUDA YOLO NMS failed", error);
    if (selected.empty()) { detections->clear(); return true; }
    const int64_t count = static_cast<int64_t>(selected.size() / 3);
    Workspace indices, gathered;
    if (!indices.Allocate(selected.size(), sizeof(int64_t)) || !gathered.Allocate(count, sizeof(Detection)) ||
        cd::CudaCheck(cudaMemcpyAsync(indices.As<int64_t>(), selected.data(), selected.size() * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, cd::InferenceStream())) != 0)
        return Fail("CUDA YOLO result allocation failed", error);
    GatherDetections<<<static_cast<int>(std::min<int64_t>(65535, cd::DivUp(count, cd::kCudaThreads))),
                       cd::kCudaThreads, 0, cd::InferenceStream()>>>(coordinates.As<float>(), scores.As<float>(),
                         labels.As<int64_t>(), indices.As<int64_t>(), gathered.As<Detection>(), count);
    std::vector<Detection> kept(static_cast<size_t>(count));
    if (cd::CudaCheck(cudaGetLastError()) != 0 ||
        cd::CudaCheck(cudaMemcpyAsync(kept.data(), gathered.As<Detection>(), kept.size() * sizeof(Detection),
                                       cudaMemcpyDeviceToHost, cd::InferenceStream())) != 0 || cd::SynchronizeInferenceStream() != 0)
        return Fail("CUDA YOLO result download failed", error);
    detections->clear();
    for (auto detection : kept) {
        if (!metadata.legacy_yolov5) {
            detection.x1 = static_cast<float>(std::clamp((static_cast<double>(detection.x1) - letterbox.pad_x) / letterbox.scale, 0.0, static_cast<double>(image_width)));
            detection.y1 = static_cast<float>(std::clamp((static_cast<double>(detection.y1) - letterbox.pad_y) / letterbox.scale, 0.0, static_cast<double>(image_height)));
            detection.x2 = static_cast<float>(std::clamp((static_cast<double>(detection.x2) - letterbox.pad_x) / letterbox.scale, 0.0, static_cast<double>(image_width)));
            detection.y2 = static_cast<float>(std::clamp((static_cast<double>(detection.y2) - letterbox.pad_y) / letterbox.scale, 0.0, static_cast<double>(image_height)));
            if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1) continue;
        }
        detections->push_back(detection);
        if (detections->size() == maximum) break;
    }
    return true;
}
}  // namespace demo
}  // namespace feather
