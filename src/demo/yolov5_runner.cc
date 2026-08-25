#include "demo/yolov5_runner.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
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

int WarmupRuntimeGraph(RuntimeGraph* runtime_graph, const std::string& output_name) {
    if (runtime_graph == nullptr) {
        return -1;
    }
    {
        kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
        if (runtime_graph->Run() != 0) {
            return -1;
        }
    }
    if (kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
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
    if (detections == nullptr) {
        return -1;
    }

    const auto run_begin = std::chrono::steady_clock::now();

    LetterboxInfo letterbox;
    const auto preprocess_begin = std::chrono::steady_clock::now();
    auto runtime_input = runtime_graph_.GetTensor(input_name_);
    if (runtime_input == nullptr) {
        return -1;
    }
    if (PreprocessImageToTensor(image, input_size_, input_dtype_, runtime_input.get(), &letterbox) != 0) {
        return -1;
    }
    double input_copy_ms = 0.0;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(runtime_input.get());
        const auto input_copy_begin = std::chrono::steady_clock::now();
        if (PrimeTensorDevice(runtime_input.get()) != 0 || kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
            return -1;
        }
        const auto input_copy_end = std::chrono::steady_clock::now();
        input_copy_ms = ElapsedMilliseconds(input_copy_begin, input_copy_end);
    }
#endif
    const auto preprocess_end = std::chrono::steady_clock::now();

    const auto rungraph_begin = std::chrono::steady_clock::now();
    int32_t run_status = 0;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
        run_status = runtime_graph_.Run();
    } else
#endif
    {
        run_status = runtime_graph_.Run();
    }
    if (run_status != 0) {
        return -1;
    }
    const auto rungraph_end = std::chrono::steady_clock::now();

    auto output_tensor = runtime_graph_.GetTensor(output_name_);
    if (output_tensor == nullptr) {
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::SyncTensorToHost(output_tensor.get(),
                                                 static_cast<size_t>(output_tensor->numel()) *
                                                     DataTypeBytes(output_tensor->data_type()),
                                                 output_tensor->mutable_data(output_tensor->numel() *
                                                                            DataTypeBytes(output_tensor->data_type()))) != 0) {
            return -1;
        }
    }
#endif
    const auto postprocess_begin = std::chrono::steady_clock::now();
    *detections = DecodeYolov5Detections(*output_tensor, letterbox, image.width, image.height,
                                         conf_thresh, iou_thresh);
    const auto postprocess_end = std::chrono::steady_clock::now();
#ifdef FEATHER_WITH_CUDA
    kernel::cuda_detail::TensorCacheStats cuda_cache_stats;
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::ReleaseTensorDevice(output_tensor.get());
        cuda_cache_stats = kernel::cuda_detail::GetTensorCacheStats();
    }
#endif

    if (record_summary) {
        std::ostringstream build_ss;
        build_ss << "model=" << model_name_
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
                 << " runtime_nodes=" << runtime_graph_.NodeSize();
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
    if (detections == nullptr || input_name_.empty() || output_name_.empty()) {
        return -1;
    }
    return RunOnImage("prepared", image, conf_thresh, iou_thresh, detections, false);
}

int32_t Yolov5Runner::PrepareExecutableGraph() {
    if (static_graph_.Build() != 0) {
        return -1;
    }
    if (static_graph_.ApplyPasses() != 0) {
        return -1;
    }
    runtime_graph_.Clear();
    if (lowering_.Lower(static_graph_, &runtime_graph_) != 0) {
        return -1;
    }
    return 0;
}

int32_t Yolov5Runner::Load(const std::string& model_path, Yolov5Backend backend,
                          Yolov5LayoutOverride layout_override) {
    backend_ = backend;
    layout_override_ = layout_override;
#ifndef FEATHER_WITH_CUDA
    if (backend_ == Yolov5Backend::kCuda) {
        return -1;
    }
#endif
    backend_device_ = ResolveBackendDevice(backend_);
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::WarmupCudaRuntime() != 0) {
            return -1;
        }
        kernel::cuda_detail::ClearTensorCache();
    }
#endif

    if (!loader_.Load(model_path)) {
        return -1;
    }
    const auto& model = loader_.model();
    if (model.graph.inputs.empty() || model.graph.outputs.empty()) {
        return -1;
    }

    input_name_ = model.graph.inputs.front();
    output_name_ = model.graph.outputs.front();
    model_name_ = model.name;
    const auto* input_value = FindValueDescByName(model, input_name_);
    if (input_value == nullptr || input_value->tensor.dims.size() != 4) {
        return -1;
    }
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(input_value->tensor.dims, input_value->tensor.layout, &input_shape) ||
        input_shape.n != 1 || input_shape.c != 3 || input_shape.h != input_shape.w) {
        return -1;
    }
    input_layout_ = ResolveInputLayout(input_value->tensor.layout, layout_override_);

    input_size_ = static_cast<int>(input_shape.h);
    input_dtype_ = input_value->tensor.data_type;

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    runtime_graph_.Clear();
    if (static_graph_.SetModel(model) != 0) {
        return -1;
    }
    static_graph_.SetPassManager(CreateYoloPassManager());
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader_.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            return -1;
        }
    }

    auto input_tensor = std::make_shared<Tensor>(EncodeImageShape4D(input_shape, input_layout_));
    if (input_tensor == nullptr) {
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
            return -1;
    }
    if (static_graph_.SetTensor(input_name_, input_tensor) != 0) {
        return -1;
    }

    const auto prepare_begin = std::chrono::steady_clock::now();
    if (PrepareExecutableGraph() != 0) {
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (PrimeConstantTensorDevices(model, &runtime_graph_) != 0 ||
            WarmupRuntimeGraph(&runtime_graph_, output_name_) != 0) {
            return -1;
        }
    }
#endif
    const auto prepare_end = std::chrono::steady_clock::now();

    std::ostringstream ss;
    ss << "model=" << model_name_
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
       << " runtime_nodes=" << runtime_graph_.NodeSize()
       << " prepare_ms=" << ElapsedMilliseconds(prepare_begin, prepare_end);
    last_build_summary_ = ss.str();
    return 0;
}

int32_t Yolov5Runner::Run(const std::string& image_path, float conf_thresh, float iou_thresh,
                          std::vector<Detection>* detections) {
    if (detections == nullptr || input_name_.empty() || output_name_.empty()) {
        return -1;
    }

    ImageData image;
    const auto load_begin = std::chrono::steady_clock::now();
    if (LoadImage(image_path, &image) != 0) {
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
