#include "demo/classification_runner.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include "pass/graph_pass.h"
#include "util/fp16.h"
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

DeviceType ResolveBackendDevice(ClassificationBackend backend) {
    switch (backend) {
        case ClassificationBackend::kCommon:
            return DeviceType::COMMON;
        case ClassificationBackend::kX86:
            return DeviceType::X86;
        case ClassificationBackend::kCuda:
            return DeviceType::CUDA;
        case ClassificationBackend::kHost:
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

void WriteTensorValue(Tensor* tensor, int64_t index, float value) {
    if (tensor->data_type() == DataType::FP16) {
        tensor->mutable_data<uint16_t>()[index] = FloatToHalf(value);
    } else {
        tensor->mutable_data<float>()[index] = value;
    }
}

float ReadTensorValue(const Tensor& tensor, int64_t index) {
    if (tensor.data_type() == DataType::FP16) {
        return HalfToFloat(tensor.data<uint16_t>()[index]);
    }
    if (tensor.data_type() == DataType::FP32) {
        return tensor.data<float>()[index];
    }
    return 0.0f;
}

std::vector<uint8_t> ResizeRgbBilinear(const ImageData& image, int width, int height) {
    std::vector<uint8_t> resized(static_cast<size_t>(width) * height * 3, 0);
    const float scale_x = static_cast<float>(image.width) / static_cast<float>(width);
    const float scale_y = static_cast<float>(image.height) / static_cast<float>(height);
    for (int y = 0; y < height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, image.height - 1);
        const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
        const float ly = source_y - std::floor(source_y);
        for (int x = 0; x < width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, image.width - 1);
            const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
            const float lx = source_x - std::floor(source_x);
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int px, int py) {
                    return static_cast<float>(image.pixels[static_cast<size_t>((py * image.width + px) * 3 + c)]);
                };
                const float top = at(x0, y0) * (1.0f - lx) + at(x1, y0) * lx;
                const float bottom = at(x0, y1) * (1.0f - lx) + at(x1, y1) * lx;
                resized[static_cast<size_t>((y * width + x) * 3 + c)] =
                    static_cast<uint8_t>(std::clamp(top * (1.0f - ly) + bottom * ly, 0.0f, 255.0f));
            }
        }
    }
    return resized;
}

}  // namespace

bool ParseClassificationBackend(const std::string& value, ClassificationBackend* backend) {
    if (backend == nullptr) {
        return false;
    }
    std::string normalized;
    normalized.reserve(value.size());
    for (const auto ch : value) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (normalized == "host") {
        *backend = ClassificationBackend::kHost;
        return true;
    }
    if (normalized == "common") {
        *backend = ClassificationBackend::kCommon;
        return true;
    }
    if (normalized == "x86") {
        *backend = ClassificationBackend::kX86;
        return true;
    }
    if (normalized == "cuda") {
        *backend = ClassificationBackend::kCuda;
        return true;
    }
    return false;
}

const char* ClassificationBackendName(ClassificationBackend backend) {
    switch (backend) {
        case ClassificationBackend::kCommon:
            return "common";
        case ClassificationBackend::kX86:
            return "x86";
        case ClassificationBackend::kCuda:
            return "cuda";
        case ClassificationBackend::kHost:
        default:
            return "host";
    }
}

int32_t PreprocessImageNetToTensor(const ImageData& image, int input_size, int resize_shorter_side,
                                   const ImageNetPreprocessConfig& config, DataType dtype, Tensor* tensor) {
    if (tensor == nullptr || image.width <= 0 || image.height <= 0 || image.channels != 3 ||
        input_size <= 0 || resize_shorter_side < input_size ||
        (dtype != DataType::FP16 && dtype != DataType::FP32)) {
        return -1;
    }

    const float resize_scale = static_cast<float>(resize_shorter_side) /
                               static_cast<float>(std::min(image.width, image.height));
    const int resized_width = std::max(input_size, static_cast<int>(std::round(image.width * resize_scale)));
    const int resized_height = std::max(input_size, static_cast<int>(std::round(image.height * resize_scale)));
    const auto resized = ResizeRgbBilinear(image, resized_width, resized_height);
    const int crop_x = (resized_width - input_size) / 2;
    const int crop_y = (resized_height - input_size) / 2;

    const DataLayout layout = NormalizeDataLayout(tensor->layout());
    tensor->Resize(EncodeImageShape4D(ImageShape4D{1, 3, input_size, input_size}, layout));
    tensor->set_data_type(dtype);
    if (!tensor->IsInitialized() || tensor->memory_size() <
                                      static_cast<size_t>(tensor->numel()) * DataTypeBytes(dtype)) {
        return -1;
    }
    if (dtype == DataType::FP16) {
        (void)tensor->mutable_data<uint16_t>();
    } else {
        (void)tensor->mutable_data<float>();
    }

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_size; ++y) {
            for (int x = 0; x < input_size; ++x) {
                const auto pixel = static_cast<float>(resized[static_cast<size_t>(((y + crop_y) * resized_width +
                                                                                    (x + crop_x)) *
                                                                                   3 + c)]) /
                                   255.0f;
                const float normalized = (pixel - config.mean[static_cast<size_t>(c)]) /
                                         config.std[static_cast<size_t>(c)];
                const auto offset = OffsetForImage4D(layout, 0, c, y, x, 3, input_size, input_size);
                WriteTensorValue(tensor, offset, normalized);
            }
        }
    }
    return 0;
}

std::vector<std::string> LoadClassificationLabels(const std::string& path) {
    std::ifstream input(path);
    if (!input.good()) {
        return {};
    }
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        labels.push_back(std::move(line));
    }
    return labels;
}

int32_t ClassificationRunner::Load(const std::string& model_path, ClassificationBackend backend) {
    last_error_.clear();
    backend_ = backend;
#ifndef FEATHER_WITH_CUDA
    if (backend_ == ClassificationBackend::kCuda) {
        last_error_ = "CUDA backend requested but this build was compiled without CUDA support";
        return -1;
    }
#endif
    backend_device_ = ResolveBackendDevice(backend);
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::WarmupCudaRuntime() != 0) {
            last_error_ = "CUDA backend initialization failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        kernel::cuda_detail::ClearTensorCache();
    }
#endif
    if (!loader_.Load(model_path)) {
        last_error_ = "failed to load Feather model: " + model_path;
        return -1;
    }
    const auto& model = loader_.model();
    if (model.graph.inputs.size() != 1 || model.graph.outputs.size() != 1) {
        last_error_ = "classification model must have exactly one input and one output";
        return -1;
    }
    input_name_ = model.graph.inputs.front();
    output_name_ = model.graph.outputs.front();
    model_name_ = model.name;
    const auto* input_value = FindValueDescByName(model, input_name_);
    if (input_value == nullptr || input_value->tensor.dims.size() != 4) {
        last_error_ = "classification input must be a declared 4D tensor";
        return -1;
    }
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(input_value->tensor.dims, input_value->tensor.layout, &input_shape) ||
        input_shape.n != 1 || input_shape.c != 3 || input_shape.h <= 0 || input_shape.w <= 0 ||
        input_shape.h != input_shape.w) {
        last_error_ = "classification input must have shape [1,3,H,H]";
        return -1;
    }
    input_size_ = static_cast<int>(input_shape.h);
    input_dtype_ = input_value->tensor.data_type;
    if (input_dtype_ != DataType::FP16 && input_dtype_ != DataType::FP32) {
        last_error_ = "classification input must use FP16 or FP32";
        return -1;
    }
    input_layout_ = NormalizeDataLayout(input_value->tensor.layout);

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    static_graph_.SetPassManager(CreateDefaultPassManager());
    runtime_graph_.Clear();
    runtime_graph_.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    if (static_graph_.SetModel(model) != 0) {
        last_error_ = "failed to set classification model on static graph";
        return -1;
    }
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader_.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            last_error_ = "failed to create model weight tensor: " + value.tensor.name;
            return -1;
        }
    }

    auto input_tensor = std::make_shared<Tensor>(input_value->tensor.dims);
    input_tensor->set_layout(input_layout_);
    input_tensor->set_data_type(input_dtype_);
    if (input_dtype_ == DataType::FP16) {
        (void)input_tensor->mutable_data<uint16_t>();
    } else {
        (void)input_tensor->mutable_data<float>();
    }
    if (static_graph_.SetTensor(input_name_, input_tensor) != 0) {
        last_error_ = "failed to attach classification input tensor";
        return -1;
    }

    const auto prepare_begin = std::chrono::steady_clock::now();
    if (PrepareExecutableGraph() != 0) {
        last_error_ = "failed to build classification graph for backend " +
                      std::string(DeviceBackendName(backend_device_));
        return -1;
    }
    const auto prepare_end = std::chrono::steady_clock::now();
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (PrimeConstantTensorDevices(model, &runtime_graph_) != 0 ||
            WarmupRuntimeGraph(&runtime_graph_, output_name_) != 0) {
            last_error_ = "CUDA graph warmup failed: " + kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
    }
#endif
    std::ostringstream summary;
    summary << "model=" << model_name_
            << " backend=" << DeviceBackendName(backend_device_)
            << " input=" << input_name_
            << " output=" << output_name_
            << " input_shape=[" << input_value->tensor.dims[0] << "," << input_value->tensor.dims[1]
            << "," << input_value->tensor.dims[2] << "," << input_value->tensor.dims[3] << "]"
            << " input_layout=" << (IsChannelLastLayout(input_layout_) ? "nhwc" : "nchw")
            << " input_dtype=" << static_cast<int>(input_dtype_)
            << " static_nodes=" << static_graph_.NodeSize()
            << " runtime_nodes=" << runtime_graph_.NodeSize()
            << " prepare_ms=" << ElapsedMilliseconds(prepare_begin, prepare_end);
    last_build_summary_ = summary.str();
    return 0;
}

int32_t ClassificationRunner::PrepareExecutableGraph() {
    if (static_graph_.Build() != 0 || static_graph_.ApplyPasses() != 0) {
        return -1;
    }
    runtime_graph_.Clear();
    runtime_graph_.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    return lowering_.Lower(static_graph_, &runtime_graph_);
}

int32_t ClassificationRunner::Run(const std::string& image_path, const ImageNetPreprocessConfig& config, int top_k,
                                   std::vector<ClassificationResult>* results) {
    last_error_.clear();
    if (results == nullptr || input_name_.empty() || output_name_.empty()) {
        last_error_ = "classification runner is not loaded or results is null";
        return -1;
    }
    ImageData image;
    const auto load_begin = std::chrono::steady_clock::now();
    if (LoadImage(image_path, &image) != 0) {
        last_error_ = "failed to load image: " + image_path;
        return -1;
    }
    const auto load_end = std::chrono::steady_clock::now();
    const auto status = RunOnImage(image_path, image, config, top_k, results, true);
    if (status != 0) {
        return status;
    }
    last_run_summary_ = "load_ms=" + std::to_string(ElapsedMilliseconds(load_begin, load_end)) + " " + last_run_summary_;
    return 0;
}

int32_t ClassificationRunner::RunPreparedImage(const ImageData& image, const ImageNetPreprocessConfig& config, int top_k,
                                                std::vector<ClassificationResult>* results) {
    last_error_.clear();
    if (results == nullptr || input_name_.empty() || output_name_.empty()) {
        last_error_ = "classification runner is not loaded or results is null";
        return -1;
    }
    return RunOnImage("prepared", image, config, top_k, results, false);
}

int32_t ClassificationRunner::RunOnImage(const std::string& image_path, const ImageData& image,
                                          const ImageNetPreprocessConfig& config, int top_k,
                                          std::vector<ClassificationResult>* results, bool record_summary) {
    if (results == nullptr || top_k <= 0 || config.input_size != input_size_ ||
        config.resize_shorter_side < config.input_size) {
        last_error_ = "invalid classification input or preprocessing configuration";
        return -1;
    }
    auto input_tensor = runtime_graph_.GetTensor(input_name_);
    if (input_tensor == nullptr) {
        last_error_ = "classification input tensor is missing from runtime graph";
        return -1;
    }
    const auto begin = std::chrono::steady_clock::now();
    const auto preprocess_begin = std::chrono::steady_clock::now();
    if (PreprocessImageNetToTensor(image, config.input_size, config.resize_shorter_side, config, input_dtype_,
                                   input_tensor.get()) != 0) {
        last_error_ = "failed to preprocess image into classification input tensor";
        return -1;
    }
    const auto preprocess_end = std::chrono::steady_clock::now();

    double input_copy_ms = 0.0;
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        kernel::cuda_detail::InvalidateTensorDevice(input_tensor.get());
        const auto input_copy_begin = std::chrono::steady_clock::now();
        if (PrimeTensorDevice(input_tensor.get()) != 0 || kernel::cuda_detail::SynchronizeInferenceStream() != 0) {
            last_error_ = "failed to copy classification input to CUDA: " +
                          kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
        const auto input_copy_end = std::chrono::steady_clock::now();
        input_copy_ms = ElapsedMilliseconds(input_copy_begin, input_copy_end);
    }
#endif

    const auto run_begin = std::chrono::steady_clock::now();
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
        last_error_ = "classification graph execution failed";
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            last_error_ += ": " + kernel::cuda_detail::CudaLastErrorMessage();
        }
#endif
        return -1;
    }
    const auto run_end = std::chrono::steady_clock::now();
    auto output = runtime_graph_.GetTensor(output_name_);
    if (output == nullptr || (output->dims().size() != 1 && output->dims().size() != 2) ||
        (output->dims().size() == 2 && output->dims()[0] != 1) ||
        (output->data_type() != DataType::FP16 && output->data_type() != DataType::FP32)) {
        last_error_ = "classification output must be a 1D or [1,C] FP16/FP32 tensor";
        return -1;
    }
#ifdef FEATHER_WITH_CUDA
    if (backend_device_ == DeviceType::CUDA) {
        if (kernel::cuda_detail::SyncTensorToHost(output.get(),
                                                  static_cast<size_t>(output->numel()) *
                                                      DataTypeBytes(output->data_type()),
                                                      output->mutable_data(output->numel() *
                                                                       DataTypeBytes(output->data_type()))) != 0) {
            last_error_ = "failed to copy classification output from CUDA: " +
                          kernel::cuda_detail::CudaLastErrorMessage();
            return -1;
        }
    }
#endif
    const int64_t class_count = output->dims()[static_cast<int>(output->dims().size() - 1)];
    if (class_count <= 0) {
        last_error_ = "classification output has no classes";
        return -1;
    }
    std::vector<float> logits(static_cast<size_t>(class_count));
    for (int64_t i = 0; i < class_count; ++i) {
        logits[static_cast<size_t>(i)] = ReadTensorValue(*output, i);
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float normalizer = 0.0f;
    for (const float value : logits) {
        normalizer += std::exp(value - max_logit);
    }
    std::vector<int> indices(static_cast<size_t>(class_count));
    std::iota(indices.begin(), indices.end(), 0);
    const auto count = std::min<size_t>(static_cast<size_t>(top_k), indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(count), indices.end(),
                      [&](int lhs, int rhs) { return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)]; });
    results->clear();
    results->reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto class_id = indices[i];
        results->push_back(ClassificationResult{class_id, logits[static_cast<size_t>(class_id)],
                                                std::exp(logits[static_cast<size_t>(class_id)] - max_logit) / normalizer});
    }
    if (record_summary) {
        std::ostringstream summary;
        summary << "model=" << model_name_ << " backend=" << DeviceBackendName(backend_device_)
                << " image=" << image_path
                << " image_size=[" << image.width << "," << image.height << "]"
                << " top_k=" << count
                << " preprocess_ms=" << ElapsedMilliseconds(preprocess_begin, preprocess_end)
                << " input_copy_ms=" << input_copy_ms
                << " rungraph_ms=" << ElapsedMilliseconds(run_begin, run_end)
                << " total_ms=" << ElapsedMilliseconds(begin, run_end);
#ifdef FEATHER_WITH_CUDA
        if (backend_device_ == DeviceType::CUDA) {
            const auto cache_stats = kernel::cuda_detail::GetTensorCacheStats();
            summary << " cuda_active_tensors=" << cache_stats.active_tensor_count
                    << " cuda_persistent_tensors=" << cache_stats.persistent_tensor_count
                    << " cuda_free_blocks=" << cache_stats.free_block_count
                    << " cuda_active_bytes=" << cache_stats.active_bytes
                    << " cuda_pooled_bytes=" << cache_stats.pooled_bytes;
            kernel::cuda_detail::ReleaseTensorDevice(output.get());
        }
#endif
        last_run_summary_ = summary.str();
    }
    return 0;
}

const model::ValueDesc* ClassificationRunner::FindValueDesc(const std::string& name) const {
    return FindValueDescByName(loader_.model(), name);
}

}  // namespace demo
}  // namespace feather
