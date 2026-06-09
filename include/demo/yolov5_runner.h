#ifndef FEATHER_DEMO_YOLOV5_RUNNER_H
#define FEATHER_DEMO_YOLOV5_RUNNER_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "demo/yolov5_postprocess.h"
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
    void SetRuntimeProfilingEnabled(bool enabled) { runtime_graph_.SetProfilingEnabled(enabled); }
    const std::vector<RuntimeProfileSummary>& RuntimeProfileSummaries() const { return runtime_graph_.ProfileSummaries(); }

   private:
    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t PrepareExecutableGraph();
    int32_t RunOnImage(const std::string& image_path, const ImageData& image, float conf_thresh,
                       float iou_thresh, std::vector<Detection>* detections, bool record_summary);

    model::ModelLoader loader_;
    StaticGraph static_graph_;
    RuntimeGraph runtime_graph_;
    GraphLowering lowering_;
    std::string input_name_;
    std::string output_name_;
    int input_size_{};
    DataType input_dtype_{DataType::UNKNOWN};
    Yolov5Backend backend_{Yolov5Backend::kHost};
    Yolov5LayoutOverride layout_override_{Yolov5LayoutOverride::kAuto};
    DataLayout input_layout_{DataLayout::NCHW};
    DeviceType backend_device_{GetHostRuntimeDevice()};
    std::string model_name_;
    std::string last_build_summary_;
    std::string last_run_summary_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_YOLOV5_RUNNER_H
