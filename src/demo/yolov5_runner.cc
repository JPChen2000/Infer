#include "demo/yolov5_runner.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <sstream>

#include "demo/image_io.h"

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
        default:
            return "host";
    }
}

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

    if (normalized == "common") {
        *backend = Yolov5Backend::kCommon;
        return true;
    }
    if (normalized == "x86") {
        *backend = Yolov5Backend::kX86;
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
        case Yolov5Backend::kHost:
        default:
            return "host";
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
    const auto preprocess_end = std::chrono::steady_clock::now();

    const auto rungraph_begin = std::chrono::steady_clock::now();
    if (runtime_graph_.Run() != 0) {
        return -1;
    }
    const auto rungraph_end = std::chrono::steady_clock::now();

    auto output_tensor = runtime_graph_.GetTensor(output_name_);
    if (output_tensor == nullptr) {
        return -1;
    }
    const auto postprocess_begin = std::chrono::steady_clock::now();
    *detections = DecodeYolov5Detections(*output_tensor, letterbox, image.width, image.height,
                                         conf_thresh, iou_thresh);
    const auto postprocess_end = std::chrono::steady_clock::now();

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
               << " input_copy_ms=0"
               << " build_ms=0"
               << " lower_ms=0"
               << " rungraph_ms=" << ElapsedMilliseconds(rungraph_begin, rungraph_end)
               << " postprocess_ms=" << ElapsedMilliseconds(postprocess_begin, postprocess_end)
               << " total_ms=" << ElapsedMilliseconds(run_begin, postprocess_end);
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
    runtime_graph_.Clear();
    if (lowering_.Lower(static_graph_, &runtime_graph_) != 0) {
        return -1;
    }
    return 0;
}

int32_t Yolov5Runner::Load(const std::string& model_path, Yolov5Backend backend) {
    backend_ = backend;
    backend_device_ = ResolveBackendDevice(backend_);

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
    if (input_value == nullptr || input_value->tensor.dims.size() != 4 ||
        input_value->tensor.dims[0] != 1 || input_value->tensor.dims[1] != 3) {
        return -1;
    }

    input_size_ = static_cast<int>(input_value->tensor.dims[2]);
    input_dtype_ = input_value->tensor.data_type;

    static_graph_ = StaticGraph();
    static_graph_.SetKernelDevice(backend_device_);
    runtime_graph_.Clear();
    if (static_graph_.SetModel(model) != 0) {
        return -1;
    }
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader_.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph_.SetTensor(value.tensor.name, tensor) != 0) {
            return -1;
        }
    }

    auto input_tensor = std::make_shared<Tensor>(input_value->tensor.dims);
    if (input_tensor == nullptr) {
        return -1;
    }
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
