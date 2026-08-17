#ifndef FEATHER_DEMO_CLASSIFICATION_RUNNER_H
#define FEATHER_DEMO_CLASSIFICATION_RUNNER_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "demo/image_io.h"
#include "model/model_io.h"

namespace feather {
namespace demo {

enum class ClassificationBackend {
    kHost,
    kCommon,
    kX86,
    kCuda,
};

struct ImageNetPreprocessConfig {
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
    int input_size{224};
    int resize_shorter_side{256};
};

struct ClassificationResult {
    int class_id{};
    float logit{};
    float probability{};
};

bool ParseClassificationBackend(const std::string& value, ClassificationBackend* backend);
const char* ClassificationBackendName(ClassificationBackend backend);

int32_t PreprocessImageNetToTensor(const ImageData& image, int input_size, int resize_shorter_side,
                                   const ImageNetPreprocessConfig& config, DataType dtype, Tensor* tensor);

std::vector<std::string> LoadClassificationLabels(const std::string& path);

class ClassificationRunner {
   public:
    int32_t Load(const std::string& model_path,
                 ClassificationBackend backend = ClassificationBackend::kHost);
    int32_t Run(const std::string& image_path, const ImageNetPreprocessConfig& config, int top_k,
                std::vector<ClassificationResult>* results);
    int32_t RunPreparedImage(const ImageData& image, const ImageNetPreprocessConfig& config, int top_k,
                             std::vector<ClassificationResult>* results);

    const std::string& DescribeLastBuild() const { return last_build_summary_; }
    const std::string& DescribeLastRun() const { return last_run_summary_; }
    const std::string& LastError() const { return last_error_; }
    void SetRuntimeProfilingEnabled(bool enabled) { runtime_graph_.SetProfilingEnabled(enabled); }
    const std::vector<RuntimeProfileSummary>& RuntimeProfileSummaries() const {
        return runtime_graph_.ProfileSummaries();
    }

   private:
    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t PrepareExecutableGraph();
    int32_t RunOnImage(const std::string& image_path, const ImageData& image,
                       const ImageNetPreprocessConfig& config, int top_k,
                       std::vector<ClassificationResult>* results, bool record_summary);

    model::ModelLoader loader_;
    StaticGraph static_graph_;
    RuntimeGraph runtime_graph_;
    GraphLowering lowering_;
    std::string input_name_;
    std::string output_name_;
    int input_size_{};
    DataType input_dtype_{DataType::UNKNOWN};
    DataLayout input_layout_{DataLayout::NCHW};
    ClassificationBackend backend_{ClassificationBackend::kHost};
    DeviceType backend_device_{GetHostRuntimeDevice()};
    std::string model_name_;
    std::string last_build_summary_;
    std::string last_run_summary_;
    std::string last_error_;
};

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_CLASSIFICATION_RUNNER_H
