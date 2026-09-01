#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "demo/image_io.h"
#include "model/model_io.h"
#include "util/bf16.h"

namespace {

struct Snapshot {
    std::vector<int64_t> dims;
    std::vector<float> values;
    feather::DataType data_type{feather::DataType::UNKNOWN};
    feather::QuantizationParams quantization{};
    std::vector<int8_t> int8_values;
    std::vector<int32_t> integer_values;
};

struct Error {
    std::string name;
    float max_abs{0.0f};
    double mean_abs{0.0};
    double rmse{0.0};
    double relative_l1{0.0};
    size_t max_index{0};
    float reference{0.0f};
    float candidate{0.0f};
    size_t order{0};
};

void Usage(const char* program) {
    std::cerr << "usage: " << program
              << " --model-a MODEL --model-b MODEL --image IMAGE [--backend common|x86]\n";
}

const char* DataTypeName(feather::DataType data_type) {
    switch (data_type) {
        case feather::DataType::INT8: return "INT8";
        case feather::DataType::UINT8: return "UINT8";
        case feather::DataType::FP16: return "FP16";
        case feather::DataType::FP32: return "FP32";
        case feather::DataType::INT32: return "INT32";
        case feather::DataType::INT64: return "INT64";
        case feather::DataType::BF16: return "BF16";
        case feather::DataType::BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
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

bool ReadTensor(const feather::Tensor& tensor, Snapshot* snapshot) {
    if (snapshot == nullptr || !tensor.IsInitialized() || tensor.numel() <= 0) return false;
    snapshot->dims = tensor.dims().data();
    snapshot->values.resize(static_cast<size_t>(tensor.numel()));
    snapshot->int8_values.clear();
    if (tensor.data_type() == feather::DataType::INT8) {
        snapshot->int8_values.resize(static_cast<size_t>(tensor.numel()));
    }
    snapshot->integer_values.clear();
    snapshot->data_type = tensor.data_type();
    const auto& quantization = tensor.quantization();
    snapshot->quantization = quantization;
    for (int64_t index = 0; index < tensor.numel(); ++index) {
        float value = 0.0f;
        switch (tensor.data_type()) {
            case feather::DataType::FP32:
                value = tensor.data<float>()[index];
                break;
            case feather::DataType::FP16:
                value = DecodeFloat16(tensor.data<uint16_t>()[index]);
                break;
            case feather::DataType::BF16: {
                const uint32_t bits = static_cast<uint32_t>(tensor.data<feather::BFloat16>()[index].bits) << 16;
                std::memcpy(&value, &bits, sizeof(value));
                break;
            }
            case feather::DataType::INT8: {
                const float scale = quantization.scale_at(0);
                const int32_t zero_point = quantization.zero_point_at(0);
                const int32_t integer_value = static_cast<int32_t>(tensor.data<int8_t>()[index]);
                snapshot->int8_values[static_cast<size_t>(index)] = static_cast<int8_t>(integer_value);
                if (snapshot->integer_values.size() < 16) snapshot->integer_values.push_back(integer_value);
                value = (integer_value - zero_point) * scale;
                break;
            }
            case feather::DataType::INT32: {
                const int32_t integer_value = tensor.data<int32_t>()[index];
                if (snapshot->integer_values.size() < 16) snapshot->integer_values.push_back(integer_value);
                value = static_cast<float>(integer_value);
                break;
            }
            case feather::DataType::INT64:
                value = static_cast<float>(tensor.data<int64_t>()[index]);
                break;
            case feather::DataType::UINT8:
            case feather::DataType::BOOL:
                value = static_cast<float>(tensor.data<uint8_t>()[index]);
                break;
            default:
                return false;
        }
        if (!std::isfinite(value)) return false;
        snapshot->values[static_cast<size_t>(index)] = value;
    }
    return true;
}

int32_t GetIntAttribute(const feather::model::NodeDesc& node, const std::string& name, int32_t fallback) {
    const auto it = node.attributes.find(name);
    if (it == node.attributes.end()) return fallback;
    if (const auto* value = std::get_if<int64_t>(&it->second)) return static_cast<int32_t>(*value);
    return fallback;
}

int8_t QuantizeReference(double real_value, double scale, int32_t zero_point) {
    const double quantized = std::round(real_value / scale) + static_cast<double>(zero_point);
    const double clamped = std::max(-128.0, std::min(127.0, quantized));
    return static_cast<int8_t>(clamped);
}

bool IsReferenceUnaryOp(const std::string& op_type) {
    return op_type == "Relu" || op_type == "ReLU" || op_type == "Neg" || op_type == "Sigmoid" ||
           op_type == "SiLU" || op_type == "Exp" || op_type == "Sqrt" || op_type == "Tanh" ||
           op_type == "Erf" || op_type == "Sin" || op_type == "Cos" || op_type == "Softplus";
}

bool IsReferenceBinaryOp(const std::string& op_type) {
    return op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div";
}

float ApplyReferenceUnary(const std::string& op_type, float value) {
    if (op_type == "Relu" || op_type == "ReLU") return std::max(value, 0.0f);
    if (op_type == "Neg") return -value;
    if (op_type == "Sigmoid") return 1.0f / (1.0f + std::exp(-value));
    if (op_type == "SiLU") return value / (1.0f + std::exp(-value));
    if (op_type == "Exp") return std::exp(value);
    if (op_type == "Sqrt") return std::sqrt(value);
    if (op_type == "Tanh") return std::tanh(value);
    if (op_type == "Erf") return std::erf(value);
    if (op_type == "Sin") return std::sin(value);
    if (op_type == "Cos") return std::cos(value);
    if (op_type == "Softplus") return std::max(value, 0.0f) + std::log1p(std::exp(-std::fabs(value)));
    return std::numeric_limits<float>::quiet_NaN();
}

float ApplyReferenceBinary(const std::string& op_type, float lhs, float rhs) {
    if (op_type == "Add") return lhs + rhs;
    if (op_type == "Sub") return lhs - rhs;
    if (op_type == "Mul") return lhs * rhs;
    if (op_type == "Div") return lhs / rhs;
    return std::numeric_limits<float>::quiet_NaN();
}

void ValidateCandidateStandardInt8Nodes(const feather::model::ModelDesc& model,
                                        const std::unordered_map<std::string, Snapshot>& snapshots) {
    for (const auto& node : model.graph.nodes) {
        const bool unary = IsReferenceUnaryOp(node.op_type);
        const bool binary = IsReferenceBinaryOp(node.op_type);
        if ((!unary && !binary) || node.inputs.empty() || node.outputs.size() != 1) continue;
        if (binary && node.inputs.size() < 2) continue;
        const auto output_it = snapshots.find(node.outputs[0]);
        if (output_it == snapshots.end() || output_it->second.data_type != feather::DataType::INT8) continue;
        const Snapshot& output = output_it->second;
        if (output.int8_values.size() != output.values.size()) continue;
        const auto lhs_it = snapshots.find(node.inputs[0]);
        if (lhs_it == snapshots.end() || lhs_it->second.data_type != feather::DataType::INT8 ||
            lhs_it->second.int8_values.size() != lhs_it->second.values.size() ||
            lhs_it->second.values.size() != output.values.size()) continue;
        const Snapshot& lhs = lhs_it->second;
        const Snapshot* rhs = nullptr;
        if (binary) {
            const auto rhs_it = snapshots.find(node.inputs[1]);
            if (rhs_it == snapshots.end() || rhs_it->second.data_type != feather::DataType::INT8 ||
                rhs_it->second.int8_values.size() != rhs_it->second.values.size() ||
                rhs_it->second.values.size() != output.values.size()) continue;
            rhs = &rhs_it->second;
        }
        size_t mismatch_count = 0;
        int32_t max_q_diff = 0;
        int8_t first_expected = 0;
        int8_t first_actual = 0;
        size_t first_index = 0;
        for (size_t index = 0; index < output.values.size(); ++index) {
            const float lhs_real =
                (static_cast<int32_t>(lhs.int8_values[index]) - lhs.quantization.zero_point_at(0)) *
                lhs.quantization.scale_at(0);
            const float rhs_real = binary
                                      ? (static_cast<int32_t>(rhs->int8_values[index]) -
                                         rhs->quantization.zero_point_at(0)) *
                                            rhs->quantization.scale_at(0)
                                      : 0.0f;
            const float real_value = binary
                                         ? ApplyReferenceBinary(node.op_type, lhs_real, rhs_real)
                                         : ApplyReferenceUnary(node.op_type, lhs_real);
            if (!std::isfinite(real_value)) continue;
            const int8_t expected = QuantizeReference(real_value, output.quantization.scale_at(0),
                                                       output.quantization.zero_point_at(0));
            const int32_t q_diff = std::abs(static_cast<int32_t>(expected) -
                                            static_cast<int32_t>(output.int8_values[index]));
            if (q_diff != 0) {
                if (mismatch_count == 0) {
                    first_expected = expected;
                    first_actual = output.int8_values[index];
                    first_index = index;
                }
                ++mismatch_count;
                max_q_diff = std::max(max_q_diff, q_diff);
            }
        }
        std::cout << "INT8_STANDARD_REFERENCE node=" << node.name << " op=" << node.op_type
                  << " input=" << node.inputs[0] << " output=" << node.outputs[0]
                  << " compared=" << output.values.size() << " mismatches=" << mismatch_count
                  << " max_q_diff=" << max_q_diff;
        if (mismatch_count != 0) {
            std::cout << " first_index=" << first_index << " expected=" << static_cast<int>(first_expected)
                      << " actual=" << static_cast<int>(first_actual);
        }
        std::cout << '\n';
    }
}

void ValidateCandidateInt8Convs(const std::string& model_path, const feather::model::ModelDesc& model,
                                const std::unordered_map<std::string, Snapshot>& snapshots) {
    feather::model::ModelLoader loader;
    if (!loader.Load(model_path)) {
        std::cout << "INT8_CONV_REFERENCE failed_to_load_model\n";
        return;
    }
    for (const auto& node : model.graph.nodes) {
        if (node.op_type != "Conv" && node.op_type != "Conv2D") continue;
        if (node.inputs.size() < 2 || node.outputs.size() != 1) continue;
        const auto* input_desc = FindValue(model, node.inputs[0]);
        const auto* weight_desc = FindValue(model, node.inputs[1]);
        const auto* output_desc = FindValue(model, node.outputs[0]);
        const auto input_it = snapshots.find(node.inputs[0]);
        const auto output_it = snapshots.find(node.outputs[0]);
        if (input_desc == nullptr || weight_desc == nullptr || output_desc == nullptr ||
            input_it == snapshots.end() || output_it == snapshots.end() ||
            input_desc->tensor.data_type != feather::DataType::INT8 ||
            weight_desc->tensor.data_type != feather::DataType::INT8 ||
            output_desc->tensor.data_type != feather::DataType::INT8 ||
            input_it->second.int8_values.size() != static_cast<size_t>(input_it->second.values.size()) ||
            output_it->second.int8_values.size() != static_cast<size_t>(output_it->second.values.size())) {
            continue;
        }
        auto weight = loader.CreateWeightTensor(node.inputs[1]);
        if (weight == nullptr || weight->data_type() != feather::DataType::INT8 || weight->dims().size() != 4) continue;
        std::shared_ptr<feather::Tensor> bias;
        if (node.inputs.size() > 2 && !node.inputs[2].empty()) bias = loader.CreateWeightTensor(node.inputs[2]);

        feather::ImageShape4D input_shape;
        feather::ImageShape4D output_shape;
        if (!feather::DecodeImageShape4D(input_desc->tensor.dims, input_desc->tensor.layout, &input_shape) ||
            !feather::DecodeImageShape4D(output_desc->tensor.dims, output_desc->tensor.layout, &output_shape)) continue;
        const int64_t out_channels = weight->dims()[0];
        const int64_t input_channels_per_group = weight->dims()[1];
        const int64_t kernel_h = weight->dims()[2];
        const int64_t kernel_w = weight->dims()[3];
        const int32_t group = GetIntAttribute(node, "group", 1);
        if (input_shape.n <= 0 || input_shape.c <= 0 || input_shape.h <= 0 || input_shape.w <= 0 ||
            output_shape.n != input_shape.n || output_shape.c != out_channels || group <= 0 ||
            input_shape.c % group != 0 || out_channels % group != 0 ||
            input_channels_per_group != input_shape.c / group) continue;

        const auto& input_quantization = input_it->second.quantization;
        const auto& output_quantization = output_it->second.quantization;
        if (!input_quantization.enabled || !output_quantization.enabled ||
            input_quantization.scale_at(0) <= 0.0f || output_quantization.scale_at(0) <= 0.0f) continue;
        const int8_t* input = input_it->second.int8_values.data();
        const int8_t* weight_data = weight->data<int8_t>();
        const int8_t* actual_output = output_it->second.int8_values.data();
        const int64_t output_channels_per_group = out_channels / group;
        const auto input_layout = feather::NormalizeDataLayout(input_desc->tensor.layout);
        const auto output_layout = feather::NormalizeDataLayout(output_desc->tensor.layout);
        size_t mismatch_count = 0;
        int32_t max_q_diff = 0;
        int8_t first_expected = 0;
        int8_t first_actual = 0;
        size_t first_index = 0;
        size_t compared = 0;
        for (int64_t batch = 0; batch < input_shape.n; ++batch) {
            for (int64_t group_index = 0; group_index < group; ++group_index) {
                for (int64_t channel = 0; channel < output_channels_per_group; ++channel) {
                    const int64_t output_channel = group_index * output_channels_per_group + channel;
                    const int32_t weight_zero_point = weight->quantization().zero_point_at(static_cast<size_t>(output_channel));
                    const double real_scale = static_cast<double>(input_quantization.scale_at(0)) *
                                              static_cast<double>(weight->quantization().scale_at(static_cast<size_t>(output_channel)));
                    for (int64_t output_h = 0; output_h < output_shape.h; ++output_h) {
                        for (int64_t output_w = 0; output_w < output_shape.w; ++output_w) {
                            int64_t accumulator = 0;
                            if (bias != nullptr && bias->IsInitialized() && bias->data_type() == feather::DataType::INT32 &&
                                bias->numel() == out_channels) {
                                accumulator = bias->data<int32_t>()[output_channel];
                            }
                            for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
                                const int64_t global_input_channel = group_index * input_channels_per_group + input_channel;
                                for (int64_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                                    const int64_t input_y = output_h * GetIntAttribute(node, "stride_h", 1) +
                                                            kernel_y * GetIntAttribute(node, "dilation_h", 1) -
                                                            GetIntAttribute(node, "pad_h", 0);
                                    if (input_y < 0 || input_y >= input_shape.h) continue;
                                    for (int64_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                                        const int64_t input_x = output_w * GetIntAttribute(node, "stride_w", 1) +
                                                                kernel_x * GetIntAttribute(node, "dilation_w", 1) -
                                                                GetIntAttribute(node, "pad_w", 0);
                                        if (input_x < 0 || input_x >= input_shape.w) continue;
                                        const int64_t input_offset = feather::OffsetForImage4D(
                                            input_layout, batch, global_input_channel, input_y, input_x,
                                            input_shape.c, input_shape.h, input_shape.w);
                                        const int64_t weight_offset =
                                            ((output_channel * input_channels_per_group + input_channel) * kernel_h + kernel_y) *
                                                kernel_w + kernel_x;
                                        accumulator += (static_cast<int32_t>(input[input_offset]) - input_quantization.zero_point_at(0)) *
                                                       (static_cast<int32_t>(weight_data[weight_offset]) - weight_zero_point);
                                    }
                                }
                            }
                            const int8_t expected = QuantizeReference(
                                static_cast<double>(accumulator) * real_scale,
                                static_cast<double>(output_quantization.scale_at(0)),
                                output_quantization.zero_point_at(0));
                            const int64_t output_offset = feather::OffsetForImage4D(
                                output_layout, batch, output_channel, output_h, output_w,
                                output_shape.c, output_shape.h, output_shape.w);
                            const int32_t q_diff = std::abs(static_cast<int32_t>(expected) -
                                                            static_cast<int32_t>(actual_output[output_offset]));
                            if (q_diff != 0) {
                                if (mismatch_count == 0) {
                                    first_expected = expected;
                                    first_actual = actual_output[output_offset];
                                    first_index = static_cast<size_t>(output_offset);
                                }
                                ++mismatch_count;
                                max_q_diff = std::max(max_q_diff, q_diff);
                            }
                            ++compared;
                        }
                    }
                }
            }
        }
        std::cout << "INT8_CONV_REFERENCE node=" << node.name << " input=" << node.inputs[0]
                  << " output=" << node.outputs[0] << " compared=" << compared
                  << " mismatches=" << mismatch_count << " max_q_diff=" << max_q_diff;
        if (mismatch_count != 0) {
            std::cout << " first_index=" << first_index << " expected=" << static_cast<int>(first_expected)
                      << " actual=" << static_cast<int>(first_actual);
        }
        std::cout << '\n';
    }
}

void PrintCandidateOnlyInt8Tensors(const feather::model::ModelDesc& reference_model,
                                   const feather::model::ModelDesc& candidate_model,
                                   const std::unordered_map<std::string, Snapshot>& candidate) {
    std::unordered_set<std::string> reference_names;
    for (const auto& value : reference_model.graph.values) reference_names.insert(value.tensor.name);

    std::vector<std::string> names;
    for (const auto& value : candidate_model.graph.values) {
        if (value.constant || reference_names.count(value.tensor.name) != 0) continue;
        if (value.tensor.name.find("__int8_") == std::string::npos) continue;
        if (candidate.find(value.tensor.name) != candidate.end()) names.push_back(value.tensor.name);
    }
    std::sort(names.begin(), names.end());
    std::cout << "-- candidate-only internal INT8 tensors --\n";
    for (const auto& name : names) {
        const auto it = candidate.find(name);
        if (it == candidate.end()) continue;
        const Snapshot& snapshot = it->second;
        std::cout << name << " dtype=" << DataTypeName(snapshot.data_type)
                  << " scale=" << snapshot.quantization.scale_at(0)
                  << " zero_point=" << snapshot.quantization.zero_point_at(0)
                  << " values=";
        for (size_t index = 0; index < std::min<size_t>(8, snapshot.values.size()); ++index) {
            if (index != 0) std::cout << ",";
            std::cout << snapshot.values[index];
        }
        if (!snapshot.integer_values.empty()) {
            std::cout << " q=";
            for (size_t index = 0; index < std::min<size_t>(8, snapshot.integer_values.size()); ++index) {
                if (index != 0) std::cout << ",";
                std::cout << snapshot.integer_values[index];
            }
        }
        std::cout << '\n';
    }
}

bool RunModel(const std::string& path, const feather::demo::ImageData& image, feather::DeviceType device,
              feather::model::ModelDesc* model, std::unordered_map<std::string, Snapshot>* snapshots) {
    if (model == nullptr || snapshots == nullptr) return false;
    feather::model::ModelLoader loader;
    if (!loader.Load(path)) return false;
    *model = loader.model();
    if (model->graph.inputs.empty() || model->graph.outputs.empty()) return false;
    const auto* input = FindValue(*model, model->graph.inputs.front());
    if (input == nullptr || input->tensor.dims.size() != 4 ||
        (input->tensor.data_type != feather::DataType::FP32 && input->tensor.data_type != feather::DataType::FP16)) {
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
    if (static_graph.SetModel(*model) != 0) return false;
    static_graph.SetPassManager(nullptr);
    for (const auto& value : model->graph.values) {
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
    if (static_graph.SetTensor(model->graph.inputs.front(), input_tensor) != 0 || static_graph.Build() != 0) return false;
    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    if (lowering.Lower(static_graph, &runtime_graph) != 0) return false;
    feather::demo::LetterboxInfo letterbox;
    if (feather::demo::PreprocessImageToTensor(image, static_cast<int>(shape.h), input->tensor.data_type,
                                               input_tensor.get(), &letterbox) != 0) return false;
    if (runtime_graph.Run() != 0) return false;
    snapshots->clear();
    for (const auto& value : model->graph.values) {
        if (value.constant) continue;
        const auto tensor = runtime_graph.GetTensor(value.tensor.name);
        if (tensor == nullptr) continue;
        Snapshot snapshot;
        if (ReadTensor(*tensor, &snapshot)) (*snapshots)[value.tensor.name] = std::move(snapshot);
    }
    return true;
}

Error Compare(const std::string& name, const Snapshot& reference, const Snapshot& candidate, size_t order) {
    Error error;
    error.name = name;
    error.order = order;
    if (reference.dims != candidate.dims || reference.values.size() != candidate.values.size() ||
        reference.values.empty()) {
        error.max_abs = std::numeric_limits<float>::infinity();
        return error;
    }
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_reference_abs = 0.0;
    for (size_t index = 0; index < reference.values.size(); ++index) {
        const float diff = std::fabs(reference.values[index] - candidate.values[index]);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * diff;
        sum_reference_abs += std::fabs(reference.values[index]);
        if (diff > error.max_abs) {
            error.max_abs = diff;
            error.max_index = index;
            error.reference = reference.values[index];
            error.candidate = candidate.values[index];
        }
    }
    error.mean_abs = sum_abs / static_cast<double>(reference.values.size());
    error.rmse = std::sqrt(sum_sq / static_cast<double>(reference.values.size()));
    error.relative_l1 = sum_abs / std::max(sum_reference_abs, 1e-12);
    return error;
}

void PrintError(const Error& error) {
    std::cout << error.order << " " << error.name << " max_abs=" << error.max_abs
              << " mean_abs=" << error.mean_abs << " rmse=" << error.rmse
              << " relative_l1=" << error.relative_l1 << " max_index=" << error.max_index
              << " reference=" << error.reference << " candidate=" << error.candidate << '\n';
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
    if (model_a.empty() || model_b.empty() || image_path.empty() || (backend != "common" && backend != "x86")) {
        Usage(argv[0]);
        return 2;
    }
    feather::demo::ImageData image;
    if (feather::demo::LoadImage(image_path, &image) != 0) return 3;
    const auto device = backend == "x86" ? feather::DeviceType::X86 : feather::DeviceType::COMMON;
    feather::model::ModelDesc model_a_desc;
    feather::model::ModelDesc model_b_desc;
    std::unordered_map<std::string, Snapshot> reference;
    std::unordered_map<std::string, Snapshot> candidate;
    if (!RunModel(model_a, image, device, &model_a_desc, &reference) ||
        !RunModel(model_b, image, device, &model_b_desc, &candidate)) {
        std::cerr << "failed to run one of the models\n";
        return 4;
    }
    std::vector<Error> errors;
    for (size_t order = 0; order < model_a_desc.graph.values.size(); ++order) {
        const auto& value = model_a_desc.graph.values[order];
        if (value.constant) continue;
        const auto a = reference.find(value.tensor.name);
        const auto b = candidate.find(value.tensor.name);
        if (a == reference.end() || b == candidate.end()) continue;
        errors.push_back(Compare(value.tensor.name, a->second, b->second, order));
    }
    std::cout << "backend=" << backend << " compared_values=" << errors.size() << '\n';
    std::cout << "-- graph order, errors >= 1e-3 --\n";
    for (const auto& error : errors) {
        if (error.max_abs >= 1e-3f) PrintError(error);
    }
    std::sort(errors.begin(), errors.end(), [](const Error& lhs, const Error& rhs) {
        return lhs.max_abs > rhs.max_abs;
    });
    std::cout << "-- top 30 by max_abs --\n";
    for (size_t index = 0; index < std::min<size_t>(30, errors.size()); ++index) PrintError(errors[index]);
    PrintCandidateOnlyInt8Tensors(model_a_desc, model_b_desc, candidate);
    ValidateCandidateStandardInt8Nodes(model_b_desc, candidate);
    ValidateCandidateInt8Convs(model_b, model_b_desc, candidate);
    return 0;
}
