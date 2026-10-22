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

#ifndef FEATHER_DEMO_CLASSIFICATION_RUNNER_H
#define FEATHER_DEMO_CLASSIFICATION_RUNNER_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/execution_plan.h"
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
    void SetRuntimeProfilingEnabled(bool enabled) { Runtime().SetProfilingEnabled(enabled); }
    const std::vector<RuntimeProfileSummary>& RuntimeProfileSummaries() const {
        return Runtime().ProfileSummaries();
    }

   private:
    const model::ValueDesc* FindValueDesc(const std::string& name) const;
    int32_t PrepareExecutableGraph();
    int32_t RunOnImage(const std::string& image_path, const ImageData& image,
                       const ImageNetPreprocessConfig& config, int top_k,
                       std::vector<ClassificationResult>* results, bool record_summary);

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
