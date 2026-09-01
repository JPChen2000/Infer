#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "demo/image_io.h"
#include "model/model_io.h"
#include "util/bf16.h"

namespace {

struct Output {
    std::vector<int64_t> dims;
    std::vector<float> values;
};

void Usage(const char* program) {
    std::cerr << "usage: " << program
              << " --model-a MODEL --model-b MODEL --image IMAGE [--backend common|x86]\n";
}

bool ValueAfter(int argc, char** argv, int* index, std::string* value) {
    if (index == nullptr || value == nullptr || *index + 1 >= argc) return false;
    *value = argv[++*index];
    return true;
}

const feather::model::ValueDesc* FindValue(const feather::model::ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) return &value;
    }
    return nullptr;
}

float DecodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t result = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            const float value = std::ldexp(static_cast<float>(fraction), -24);
            return sign == 0 ? value : -value;
        }
    } else if (exponent == 0x1fu) {
        result |= 0x7f800000u | (fraction << 13);
    } else {
        result |= ((exponent + 112u) << 23) | (fraction << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

bool ReadOutput(const feather::Tensor& tensor, Output* output) {
    if (output == nullptr || !tensor.IsInitialized() || tensor.numel() <= 0) return false;
    output->dims = tensor.dims().data();
    output->values.resize(static_cast<size_t>(tensor.numel()));
    for (int64_t index = 0; index < tensor.numel(); ++index) {
        switch (tensor.data_type()) {
            case feather::DataType::FP32:
                output->values[static_cast<size_t>(index)] = tensor.data<float>()[index];
                break;
            case feather::DataType::FP16:
                output->values[static_cast<size_t>(index)] = DecodeFloat16(tensor.data<uint16_t>()[index]);
                break;
            case feather::DataType::BF16: {
                const uint32_t bits = static_cast<uint32_t>(tensor.data<feather::BFloat16>()[index].bits) << 16;
                std::memcpy(&output->values[static_cast<size_t>(index)], &bits, sizeof(float));
                break;
            }
            default:
                return false;
        }
    }
    return true;
}

bool RunModel(const std::string& path, const feather::demo::ImageData& image,
              feather::DeviceType device, Output* output) {
    if (output == nullptr) return false;
    feather::model::ModelLoader loader;
    if (!loader.Load(path)) return false;
    const auto& model = loader.model();
    if (model.graph.inputs.empty() || model.graph.outputs.empty()) return false;
    const auto* input = FindValue(model, model.graph.inputs.front());
    if (input == nullptr || input->tensor.dims.size() != 4 ||
        (input->tensor.data_type != feather::DataType::FP32 &&
         input->tensor.data_type != feather::DataType::FP16)) {
        return false;
    }
    feather::ImageShape4D shape;
    if (!feather::DecodeImageShape4D(input->tensor.dims, input->tensor.layout, &shape) ||
        shape.n != 1 || shape.c != 3 || shape.h <= 0 || shape.h != shape.w) {
        return false;
    }

    feather::StaticGraph static_graph;
    static_graph.SetKernelDevice(device);
    static_graph.SetPassManager(nullptr);
    if (static_graph.SetModel(model) != 0) return false;
    static_graph.SetPassManager(nullptr);
    for (const auto& value : model.graph.values) {
        if (!value.constant) continue;
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr || static_graph.SetTensor(value.tensor.name, std::move(tensor)) != 0) return false;
    }
    auto input_tensor = std::make_shared<feather::Tensor>(input->tensor.dims);
    input_tensor->set_layout(input->tensor.layout);
    input_tensor->set_data_type(input->tensor.data_type);
    if (input->tensor.data_type == feather::DataType::FP16) {
        (void)input_tensor->mutable_data<uint16_t>();
    } else {
        (void)input_tensor->mutable_data<float>();
    }
    if (static_graph.SetTensor(model.graph.inputs.front(), input_tensor) != 0 || static_graph.Build() != 0) {
        return false;
    }
    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    if (lowering.Lower(static_graph, &runtime_graph) != 0) return false;
    feather::demo::LetterboxInfo letterbox;
    if (feather::demo::PreprocessImageToTensor(image, static_cast<int>(shape.h), input->tensor.data_type,
                                               input_tensor.get(), &letterbox) != 0) {
        return false;
    }
    if (runtime_graph.Run() != 0) return false;
    const auto result = runtime_graph.GetTensor(model.graph.outputs.front());
    return result != nullptr && ReadOutput(*result, output);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_a;
    std::string model_b;
    std::string image_path;
    std::string backend = "common";
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model-a") {
            if (!ValueAfter(argc, argv, &index, &model_a)) { Usage(argv[0]); return 2; }
        } else if (argument == "--model-b") {
            if (!ValueAfter(argc, argv, &index, &model_b)) { Usage(argv[0]); return 2; }
        } else if (argument == "--image") {
            if (!ValueAfter(argc, argv, &index, &image_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--backend") {
            if (!ValueAfter(argc, argv, &index, &backend)) { Usage(argv[0]); return 2; }
        } else {
            Usage(argv[0]);
            return 2;
        }
    }
    if (model_a.empty() || model_b.empty() || image_path.empty() ||
        (backend != "common" && backend != "x86")) {
        Usage(argv[0]);
        return 2;
    }
    feather::demo::ImageData image;
    if (feather::demo::LoadImage(image_path, &image) != 0) {
        std::cerr << "failed to load image: " << image_path << '\n';
        return 3;
    }
    const auto device = backend == "x86" ? feather::DeviceType::X86 : feather::DeviceType::COMMON;
    Output reference;
    Output candidate;
    if (!RunModel(model_a, image, device, &reference) || !RunModel(model_b, image, device, &candidate)) {
        std::cerr << "failed to run one of the models\n";
        return 4;
    }
    if (reference.dims != candidate.dims || reference.values.size() != candidate.values.size() ||
        reference.values.empty()) {
        std::cerr << "output shape or size mismatch\n";
        return 5;
    }
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_reference_abs = 0.0;
    float max_abs = 0.0f;
    size_t max_index = 0;
    for (size_t index = 0; index < reference.values.size(); ++index) {
        const float diff = std::fabs(reference.values[index] - candidate.values[index]);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * diff;
        sum_reference_abs += std::fabs(reference.values[index]);
        if (diff > max_abs) {
            max_abs = diff;
            max_index = index;
        }
    }
    const double count = static_cast<double>(reference.values.size());
    std::cout << "backend=" << backend << " elements=" << reference.values.size()
              << " max_abs=" << max_abs << " mean_abs=" << sum_abs / count
              << " rmse=" << std::sqrt(sum_sq / count)
              << " relative_l1=" << sum_abs / std::max(sum_reference_abs, 1e-12)
              << " max_index=" << max_index
              << " reference=" << reference.values[max_index]
              << " candidate=" << candidate.values[max_index] << '\n';
    return 0;
}
