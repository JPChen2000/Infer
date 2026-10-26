// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_DEMO_YOLOV5_RUNNER_H
#define FEATHER_DEMO_YOLOV5_RUNNER_H

#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/execution_plan.h"
#include "core/static_graph.h"
#include "demo/yolo_postprocess.h"
#include "model/model_io.h"

namespace feather {
namespace demo {

enum class Yolov5Backend {
    kHost,
    kCommon,
    kX86,
    kCuda,
};

enum class Yolov5LayoutOverride {
    kAuto,
    kNchw,
    kNhwc,
};

bool ParseYolov5Backend(const std::string& value, Yolov5Backend* backend);
const char* Yolov5BackendName(Yolov5Backend backend);
bool ParseYolov5LayoutOverride(const std::string& value, Yolov5LayoutOverride* layout);
const char* Yolov5LayoutOverrideName(Yolov5LayoutOverride layout);

class Yolov5Runner {
   public:
    int32_t Load(const std::string& model_path, Yolov5Backend backend = Yolov5Backend::kHost,
                 Yolov5LayoutOverride layout_override = Yolov5LayoutOverride::kAuto);
    int32_t Run(const std::string& image_path, float conf_thresh, float iou_thresh,
                std::vector<Detection>* detections);
    int32_t RunPreparedImage(const ImageData& image, float conf_thresh, float iou_thresh,
                             std::vector<Detection>* detections);
    const std::string& DescribeLastBuild() const { return last_build_summary_; }
    const std::string& DescribeLastRun() const { return last_run_summary_; }
    const std::string& LastError() const { return last_error_; }
    const YoloMetadata& DetectionMetadata() const { return detection_metadata_; }
    const Tensor* LastRawOutput() const;
    const Tensor* LastInput() const { return Runtime().GetTensor(input_name_).get(); }
    const LetterboxInfo& LastLetterbox() const { return last_letterbox_; }
    void SetMaxDetections(size_t maximum) { max_detections_ = maximum; }
    void SetRuntimeProfilingEnabled(bool enabled) { Runtime().SetProfilingEnabled(enabled); }
    const std::vector<RuntimeProfileSummary>& RuntimeProfileSummaries() const { return Runtime().ProfileSummaries(); }

   private:
    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t PrepareExecutableGraph();
    int32_t RunOnImage(const std::string& image_path, const ImageData& image, float conf_thresh,
                       float iou_thresh, std::vector<Detection>* detections, bool record_summary);

    model::ModelLoader loader_;
    StaticGraph static_graph_;
    RuntimeGraph& Runtime() { return session_ ? session_->runtime() : bootstrap_graph_; }
    const RuntimeGraph& Runtime() const { return session_ ? session_->runtime() : bootstrap_graph_; }
    RuntimeGraph bootstrap_graph_;
    std::shared_ptr<const CompiledGraph> compiled_graph_;
    std::unique_ptr<ExecutionSession> session_;
    std::string input_name_;
    std::string output_name_;
    int input_size_{};
    DataType input_dtype_{DataType::UNKNOWN};
    Yolov5Backend backend_{Yolov5Backend::kHost};
    Yolov5LayoutOverride layout_override_{Yolov5LayoutOverride::kAuto};
    DataLayout input_layout_{DataLayout::NCHW};
    DeviceType backend_device_{GetHostRuntimeDevice()};
    std::string model_name_;
    YoloMetadata detection_metadata_;
    LetterboxInfo last_letterbox_;
    size_t max_detections_{std::numeric_limits<size_t>::max()};
    bool loaded_{false};
    std::string last_error_;
    std::string last_build_summary_;
    std::string last_run_summary_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_YOLOV5_RUNNER_H
