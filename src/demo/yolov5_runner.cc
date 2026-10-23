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

#include "demo/yolov5_runner.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

#include "demo/image_io.h"
#include "pass/graph_pass.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
namespace demo {

namespace {

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point& begin,
                           const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
}

void AppendNodeDevices(std::ostringstream& summary, const StaticGraph& graph, const RuntimeGraph& runtime) {
    size_t cuda_nodes = 0, common_nodes = 0;
    std::string common_ops;
    for (const auto& node : graph.nodes()) {
        if (node.removed) continue;
        const auto* executable = runtime.GetNode(node.name);
        if (!executable) continue;
        cuda_nodes += executable->kernel_device == DeviceType::CUDA;
        common_nodes += executable->kernel_device == DeviceType::COMMON;
        if (executable->kernel_device == DeviceType::COMMON) {
            if (!common_ops.empty()) common_ops += ',';
            common_ops += executable->op_type;
        }
    }
    summary << " cuda_nodes=" << cuda_nodes << " common_nodes=" << common_nodes;
    if (!common_ops.empty()) summary << " common_ops=" << common_ops;
}

const model::ValueDesc* FindValueDescByName(const model::ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) {
            return &value;
        }
    }
    return nullptr;
}

DeviceType ResolveBackendDevice(Yolov5Backend backend) {
    switch (backend) {
        case Yolov5Backend::kCommon:
            return DeviceType::COMMON;
        case Yolov5Backend::kX86:
            return DeviceType::X86;
        case Yolov5Backend::kCuda:
            return DeviceType::CUDA;
        case Yolov5Backend::kHost:
        default:
            return GetHostRuntimeDevice();
    }
}

const char* DeviceBackendName(DeviceType device) {
    switch (device) {
        case DeviceType::COMMON:
            return "common";
        case DeviceType::X86:
            return "x86";
        case DeviceType::CUDA:
            return "cuda";
        default:
            return "host";
    }
}

size_t TensorTransferBytes(const Tensor& tensor) {
    const auto dtype_bytes = DataTypeBytes(tensor.data_type());
    if (dtype_bytes == 0) {
        return tensor.memory_size();
    }
    return static_cast<size_t>(std::max<int64_t>(0, tensor.numel())) * dtype_bytes;
}

#ifdef FEATHER_WITH_CUDA
int PrimeTensorDevice(Tensor* tensor) {
    if (tensor == nullptr || !tensor->IsInitialized()) {
        return 0;
    }
    void* device_ptr = nullptr;
    return kernel::cuda_detail::AcquireTensorDevice(tensor, TensorTransferBytes(*tensor), tensor->raw_data(), &device_ptr);
}

int PrimeConstantTensorDevices(const model::ModelDesc& model, RuntimeGraph* runtime_graph) {
    if (runtime_graph == nullptr) {
        return -1;
    }
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = runtime_graph->GetTensor(value.tensor.name);
        if (tensor == nullptr || PrimeTensorDevice(tensor.get()) != 0) {
            return -1;
        }
        kernel::cuda_detail::MarkTensorDevicePersistent(tensor.get(), true);
    }
    return kernel::cuda_detail::SynchronizeInferenceStream();
}

int WarmupRuntimeGraph(RuntimeGraph* runtime_graph, const std::string& output_name, std::string* error) {
    if (runtime_graph == nullptr) {
        return -1;
    }
    {
        kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
        const auto status = runtime_graph->RunStatus();
        if (!status.ok()) {
            *error = "YOLO warmup: " + status.message();
            return -1;
        }
    }
    if (kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
        *error = "YOLO CUDA warmup synchronization failed: " + kernel::cuda_detail::CudaLastErrorMessage();
        return -1;
    }
    auto output_tensor = runtime_graph->GetTensor(output_name);
    if (output_tensor != nullptr) {
        kernel::cuda_detail::ReleaseTensorDevice(output_tensor.get());
    }
    return 0;
}
#endif

}  // namespace

const Tensor* Yolov5Runner::LastRawOutput() const {
    auto tensor = Runtime().GetTensor(output_name_);
    if (!tensor || !Runtime().Backend().ReadHost(*tensor).ok()) return nullptr;
    return tensor.get();
}

bool ParseYolov5Backend(const std::string& value, Yolov5Backend* backend) {
    if (backend == nullptr) {
        return false;
    }

    std::string normalized;
    normalized.reserve(value.size());
    for (const auto ch : value) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (normalized == "host") {
        *backend = Yolov5Backend::kHost;
        return true;
    }
    if (normalized == "common") {
        *backend = Yolov5Backend::kCommon;
        return true;
    }
    if (normalized == "x86") {
        *backend = Yolov5Backend::kX86;
        return true;
    }
    if (normalized == "cuda") {
        *backend = Yolov5Backend::kCuda;
        return true;
    }
    return false;
}

bool ParseYolov5LayoutOverride(const std::string& value, Yolov5LayoutOverride* layout) {
    if (layout == nullptr) {
        return false;
    }

    std::string normalized;
    normalized.reserve(value.size());
    for (const auto ch : value) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (normalized == "auto") {
        *layout = Yolov5LayoutOverride::kAuto;
        return true;
    }
    if (normalized == "nchw") {
        *layout = Yolov5LayoutOverride::kNchw;
        return true;
    }
    if (normalized == "nhwc") {
        *layout = Yolov5LayoutOverride::kNhwc;
        return true;
    }
    return false;
}

const char* Yolov5BackendName(Yolov5Backend backend) {
    switch (backend) {
        case Yolov5Backend::kCommon:
            return "common";
        case Yolov5Backend::kX86:
            return "x86";
        case Yolov5Backend::kCuda:
            return "cuda";
        case Yolov5Backend::kHost:
        default:
            return "host";
    }
}

const char* Yolov5LayoutOverrideName(Yolov5LayoutOverride layout) {
    switch (layout) {
        case Yolov5LayoutOverride::kNchw:
            return "nchw";
        case Yolov5LayoutOverride::kNhwc:
            return "nhwc";
        case Yolov5LayoutOverride::kAuto:
        default:
            return "auto";
    }
}

const char* DataLayoutName(DataLayout layout) {
    switch (NormalizeDataLayout(layout)) {
        case DataLayout::NHWC:
            return "nhwc";
        case DataLayout::NCHW:
        default:
            return "nchw";
    }
}

DataLayout ResolveInputLayout(DataLayout model_layout, Yolov5LayoutOverride override_layout) {
    switch (override_layout) {
        case Yolov5LayoutOverride::kNchw:
            return DataLayout::NCHW;
        case Yolov5LayoutOverride::kNhwc:
            return DataLayout::NHWC;
        case Yolov5LayoutOverride::kAuto:
        default:
            return NormalizeDataLayout(model_layout);
    }
}

int32_t Yolov5Runner::RunOnImage(const std::string& image_path, const ImageData& image, float conf_thresh,
                                 float iou_thresh, std::vector<Detection>* detections, bool record_summary) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    if (detections == nullptr) {
        last_error_ = "YOLO detection destination is null";
        return -1;
    }
    if (!std::isfinite(conf_thresh) || conf_thresh < 0 || conf_thresh > 1 ||
        !std::isfinite(iou_thresh) || iou_thresh < 0 || iou_thresh > 1 || max_detections_ == 0) {
        last_error_ = "YOLO thresholds must be in [0,1] and max_detections must be positive";
        return -1;
    }

    const auto run_begin = std::chrono::steady_clock::now();

    LetterboxInfo letterbox;
    const auto preprocess_begin = std::chrono::steady_clock::now();
    auto runtime_input = Runtime().GetTensor(input_name_);
    if (runtime_input == nullptr) {
        last_error_ = "YOLO runtime input is missing: " + input_name_;
        return -1;
    }
    if (PreprocessImageToTensor(image, input_size_, input_dtype_, runtime_input.get(), &letterbox) != 0) {
        last_error_ = "failed to preprocess YOLO input image";
        return -1;
    }
    last_letterbox_ = letterbox;
    double input_copy_ms = 0.0;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(runtime_input.get());
        const auto input_copy_begin = std::chrono::steady_clock::now();
        if (PrimeTensorDevice(runtime_input.get()) != 0 || kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
            last_error_ = "YOLO CUDA input transfer failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        const auto input_copy_end = std::chrono::steady_clock::now();
        input_copy_ms = ElapsedMilliseconds(input_copy_begin, input_copy_end);
    }
#endif
    const auto preprocess_end = std::chrono::steady_clock::now();

    const auto rungraph_begin = std::chrono::steady_clock::now();
    Status run_status;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
        run_status = Runtime().RunStatus();
    } else
#endif
    {
        run_status = Runtime().RunStatus();
    }
    if (!run_status.ok()) {
        last_error_ = run_status.WithContext("YOLO graph execution").message();
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    // Graph time includes execution on the device, including when every node
    // remains on CUDA and no Common fallback forces an earlier synchronization.
    if (backend_device_ == DeviceType::CUDA && kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
        last_error_ = "YOLO CUDA graph synchronization failed: " + kernel::cuda_detail::CudaLastErrorMessage();
        return -1;
    }
#endif
    const auto rungraph_end = std::chrono::steady_clock::now();

    auto output_tensor = Runtime().GetTensor(output_name_);
    if (output_tensor == nullptr) {
        last_error_ = "YOLO runtime output is missing: " + output_name_;
        return -1;
    }
    bool cuda_nms = false;
#ifdef FEATHER_WITH_CUDA
    cuda_nms = backend_device_ == DeviceType::CUDA && detection_metadata_.nms_required;
    if (backend_device_ == DeviceType::CUDA && !cuda_nms) {
        if (kernel::cuda_detail::SyncTensorToHost(output_tensor.get(),
                                                 static_cast<size_t>(output_tensor->numel()) *
                                                     DataTypeBytes(output_tensor->data_type()),
                                                 output_tensor->raw_data()) != 0) {
            last_error_ = "YOLO CUDA output transfer failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
    }
#endif
    const auto postprocess_begin = std::chrono::steady_clock::now();
    bool decoded = false;
#ifdef FEATHER_WITH_CUDA
    if (cuda_nms) {
        decoded = DecodeYoloDetectionsCuda(*output_tensor, detection_metadata_, letterbox,
                                          image.width, image.height, conf_thresh, iou_thresh,
                                          max_detections_, detections, &last_error_);
    } else
#endif
    {
        decoded = DecodeYoloDetections(*output_tensor, detection_metadata_, letterbox,
                                       image.width, image.height, conf_thresh, iou_thresh,
                                       max_detections_, detections, &last_error_);
    }
    const auto postprocess_end = std::chrono::steady_clock::now();
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::TensorCacheStats cuda_cache_stats;
    if (backend_device_ == DeviceType::CUDA) {
        // Keep raw output on device for an optional LastRawOutput()/JSON read.
        // Ordinary detection only downloads the selected candidate records.
        if (!cuda_nms) kernel::cuda_detail::ReleaseTensorDevice(output_tensor.get());
        cuda_cache_stats = kernel::cuda_detail::GetTensorCacheStats();
    }
#endif
    if (!decoded) return -1;

    if (record_summary) {
        std::ostringstream build_ss;
        build_ss << "model=" << model_name_
                 << " family=" << detection_metadata_.family
                 << " backend=" << DeviceBackendName(backend_device_)
                 << " input=" << input_name_
                 << " output=" << output_name_
                 << " input_shape=[";
        const auto* input_value = FindValueDesc(input_name_);
        if (input_value != nullptr) {
            for (size_t i = 0; i < input_value->tensor.dims.size(); ++i) {
                if (i != 0) {
                    build_ss << ",";
                }
                build_ss << input_value->tensor.dims[i];
            }
        }
        build_ss << "]"
                 << " input_dtype=" << static_cast<int>(input_dtype_)
                 << " static_nodes=" << static_graph_.NodeSize()
                 << " runtime_nodes=" << Runtime().NodeSize();
        AppendNodeDevices(build_ss, static_graph_, Runtime());
        last_build_summary_ = build_ss.str();

        std::ostringstream run_ss;
        run_ss << last_build_summary_
               << " image=" << image_path
               << " image_size=[" << image.width << "," << image.height << "]"
               << " letterbox_scale=" << letterbox.scale
               << " pad=[" << letterbox.pad_x << "," << letterbox.pad_y << "]"
               << " detections=" << detections->size()
               << " preprocess_ms=" << ElapsedMilliseconds(preprocess_begin, preprocess_end)
               << " input_copy_ms=" << input_copy_ms
               << " build_ms=0"
               << " lower_ms=0"
               << " rungraph_ms=" << ElapsedMilliseconds(rungraph_begin, rungraph_end)
               << " postprocess_ms=" << ElapsedMilliseconds(postprocess_begin, postprocess_end)
               << " total_ms=" << ElapsedMilliseconds(run_begin, postprocess_end);
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            run_ss << " cuda_active_tensors=" << cuda_cache_stats.active_tensor_count
                   << " cuda_persistent_tensors=" << cuda_cache_stats.persistent_tensor_count
                   << " cuda_free_blocks=" << cuda_cache_stats.free_block_count
                   << " cuda_active_bytes=" << cuda_cache_stats.active_bytes
                   << " cuda_pooled_bytes=" << cuda_cache_stats.pooled_bytes;
        }
#endif
        last_run_summary_ = run_ss.str();
    }
    return 0;
}

int32_t Yolov5Runner::RunPreparedImage(const ImageData& image, float conf_thresh, float iou_thresh,
                                       std::vector<Detection>* detections) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    last_error_.clear();
    if (detections == nullptr || !loaded_) {
        last_error_ = "YOLO runner must be loaded and detections must not be null";
        return -1;
    }
    return RunOnImage("prepared", image, conf_thresh, iou_thresh, detections, false);
}

int32_t Yolov5Runner::PrepareExecutableGraph() {
    auto status = static_graph_.BuildStatus();
    if (!status.ok()) { last_error_ = status.WithContext("YOLO graph build").message(); return -1; }
    status = static_graph_.ApplyPassesStatus();
    if (!status.ok()) { last_error_ = status.WithContext("YOLO graph passes").message(); return -1; }
    std::shared_ptr<const CompiledGraph> compiled;
    status = CompiledGraph::Compile(static_graph_, &compiled);
    if (!status.ok()) { last_error_ = status.WithContext("YOLO graph compilation").message(); return -1; }
    std::unique_ptr<ExecutionSession> session;
    SessionOptions options;
    status = compiled->CreateSession(&session, options);
    if (!status.ok()) { last_error_ = status.WithContext("YOLO session creation").message(); return -1; }
    session->runtime().SetProfilingEnabled(Runtime().ProfilingEnabled());
    compiled_graph_ = std::move(compiled);
    session_ = std::move(session);
    return 0;
}

int32_t Yolov5Runner::Load(const std::string& model_path, Yolov5Backend backend,
                          Yolov5LayoutOverride layout_override) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    last_error_.clear();
    last_build_summary_.clear();
    last_run_summary_.clear();
    loaded_ = false;
    backend_ = backend;
    layout_override_ = layout_override;
#ifndef FEATHER_WITH_CUDA
    if (backend_ == Yolov5Backend::kCuda) {
        last_error_ = "CUDA backend requested but this build has no CUDA support";
        return -1;
    }
#endif
    backend_device_ = ResolveBackendDevice(backend_);
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::WarmupCudaRuntime() != 0) {
            last_error_ = "YOLO CUDA initialization failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        kernel::cuda_detail::ClearTensorCache();
    }
#endif

    if (!loader_.Load(model_path)) {
        last_error_ = "failed to load YOLO model: " + model_path;
        return -1;
    }
    const auto& model = loader_.model();
    if (!ParseYoloMetadata(model, &detection_metadata_, &last_error_)) return -1;
    if (model.graph.inputs.empty() || model.graph.outputs.empty()) {
        last_error_ = "YOLO graph must declare an input and output";
        return -1;
    }

    input_name_ = model.graph.inputs.front();
    output_name_ = model.graph.outputs.front();
    model_name_ = model.name;
    const auto* input_value = FindValueDescByName(model, input_name_);
    if (input_value == nullptr || input_value->tensor.dims.size() != 4) {
        last_error_ = "YOLO input must have a static rank-four image shape";
        return -1;
    }
    if (!detection_metadata_.legacy_yolov5) {
        if (model.graph.inputs.size() != 1 || model.graph.outputs.size() != 1) {
            last_error_ = "YOLO detection models require exactly one input and one decoded output";
            return -1;
        }
        const auto* output = FindValueDescByName(model, output_name_);
        if (output == nullptr) { last_error_ = "YOLO output descriptor is missing"; return -1; }
        if (!ValidateYoloOutput(output->tensor.dims, output->tensor.data_type, detection_metadata_, &last_error_)) return -1;
    }
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(input_value->tensor.dims, input_value->tensor.layout, &input_shape) ||
        input_shape.n != 1 || input_shape.c != 3 || input_shape.h <= 0 ||
        input_shape.h > std::numeric_limits<int>::max() || input_shape.h != input_shape.w) {
        last_error_ = "YOLO requires a static batch-one three-channel square image input";
        return -1;
    }
    input_layout_ = ResolveInputLayout(input_value->tensor.layout, layout_override_);

    input_size_ = static_cast<int>(input_shape.h);
    input_dtype_ = input_value->tensor.data_type;

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    session_.reset();
    compiled_graph_.reset();
    Runtime().Clear();
    if (static_graph_.SetModel(model) != 0) {
        last_error_ = "failed to set YOLO graph model";
        return -1;
    }
    static_graph_.SetPassManager(CreateYoloPassManager());
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader_.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            last_error_ = "failed to attach YOLO weight: " + value.tensor.name;
            return -1;
        }
    }

    auto input_tensor = std::make_shared<Tensor>(EncodeImageShape4D(input_shape, input_layout_));
    if (input_tensor == nullptr) {
        last_error_ = "failed to allocate YOLO input tensor";
        return -1;
    }
    input_tensor->set_layout(input_layout_);
    switch (input_dtype_) {
        case DataType::FP16:
            (void)input_tensor->mutable_data<uint16_t>();
            break;
        case DataType::FP32:
            (void)input_tensor->mutable_data<float>();
            break;
        default:
            last_error_ = "YOLO image input must be FP32 or FP16";
            return -1;
    }
    if (static_graph_.SetTensor(input_name_, input_tensor) != 0) {
        last_error_ = "failed to bind YOLO input tensor";
        return -1;
    }

    const auto prepare_begin = std::chrono::steady_clock::now();
    if (PrepareExecutableGraph() != 0) {
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::TensorCacheStats cuda_cache_stats;
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::ExecutionContextScope session_scope(Runtime().CudaContext());
        if (PrimeConstantTensorDevices(model, &Runtime()) != 0) {
            last_error_ = "failed to initialize YOLO CUDA weights: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        if (WarmupRuntimeGraph(&Runtime(), output_name_, &last_error_) != 0) return -1;
        cuda_cache_stats = kernel::cuda_detail::GetTensorCacheStats();
    }
#endif
    const auto prepare_end = std::chrono::steady_clock::now();

    std::ostringstream ss;
    ss << "model=" << model_name_
       << " family=" << detection_metadata_.family
       << " backend=" << DeviceBackendName(backend_device_)
       << " input=" << input_name_
       << " output=" << output_name_
       << " input_shape=[";
    for (size_t i = 0; i < input_value->tensor.dims.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << input_value->tensor.dims[i];
    }
    ss << "]"
       << " input_layout=" << DataLayoutName(input_layout_)
       << " input_dtype=" << static_cast<int>(input_dtype_)
       << " static_nodes=" << static_graph_.NodeSize()
       << " runtime_nodes=" << Runtime().NodeSize()
       << " prepare_ms=" << ElapsedMilliseconds(prepare_begin, prepare_end);
    AppendNodeDevices(ss, static_graph_, Runtime());
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        ss << " cuda_active_tensors=" << cuda_cache_stats.active_tensor_count
           << " cuda_persistent_tensors=" << cuda_cache_stats.persistent_tensor_count
           << " cuda_free_blocks=" << cuda_cache_stats.free_block_count
           << " cuda_active_bytes=" << cuda_cache_stats.active_bytes
           << " cuda_pooled_bytes=" << cuda_cache_stats.pooled_bytes;
    }
#endif
    last_build_summary_ = ss.str();
    loaded_ = true;
    return 0;
}

int32_t Yolov5Runner::Run(const std::string& image_path, float conf_thresh, float iou_thresh,
                          std::vector<Detection>* detections) {
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::ExecutionContextScope context_scope(Runtime().CudaContext());
#endif
    last_error_.clear();
    if (detections == nullptr || !loaded_) {
        last_error_ = "YOLO runner must be loaded and detections must not be null";
        return -1;
    }

    ImageData image;
    const auto load_begin = std::chrono::steady_clock::now();
    if (LoadImage(image_path, &image) != 0) {
        last_error_ = "failed to load YOLO input image: " + image_path;
        return -1;
    }
    const auto load_end = std::chrono::steady_clock::now();
    const auto status = RunOnImage(image_path, image, conf_thresh, iou_thresh, detections, true);
    if (status != 0) {
        return status;
    }

    std::ostringstream run_ss;
    run_ss << "load_ms=" << ElapsedMilliseconds(load_begin, load_end) << ' ' << last_run_summary_;
    last_run_summary_ = run_ss.str();
    return 0;
}

const model::ValueDesc* Yolov5Runner::FindValueDesc(const std::string& name) const {
    return FindValueDescByName(loader_.model(), name);
}

}  // namespace demo
}  // namespace feather
