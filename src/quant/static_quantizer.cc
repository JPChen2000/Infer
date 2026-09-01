#include "quant/static_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "quant/quantization.h"

namespace feather {
namespace {

using model::ModelDesc;
using model::NodeDesc;
using model::TensorDesc;
using model::ValueDesc;

bool IsValidActivation(const ActivationQuantization& value) {
    return std::isfinite(value.scale) && value.scale > 0.0f && value.zero_point >= -128 && value.zero_point <= 127;
}

int64_t Product(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0 || result > std::numeric_limits<int64_t>::max() / dim) return 0;
        result *= dim;
    }
    return result;
}

std::string NodeLabel(const NodeDesc& node, size_t index) {
    return node.name.empty() ? node.op_type + "_" + std::to_string(index) : node.name;
}

bool KeepNodeInFp32(const StaticQuantizationConfig& config, const NodeDesc& node, size_t index) {
    const std::string label = NodeLabel(node, index);
    return std::any_of(config.keep_fp32_node_prefixes.begin(), config.keep_fp32_node_prefixes.end(),
                       [&label](const std::string& prefix) {
                           return !prefix.empty() && label.compare(0, prefix.size(), prefix) == 0;
                       });
}

bool IsQuantizable(const std::string& op_type) {
    // The ONNX importer preserves source operator names.  Conv is the ONNX
    // alias for the runtime's Conv2D implementation and must enter the same
    // INT8 weight/bias path.
    return op_type == "FC" || op_type == "Gemm" || op_type == "MatMul" || op_type == "Conv2D" ||
           op_type == "Conv";
}

const ValueDesc* FindValue(const ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) return &value;
    }
    return nullptr;
}

ValueDesc* FindValue(ModelDesc* model, const std::string& name) {
    if (model == nullptr) return nullptr;
    for (auto& value : model->graph.values) {
        if (value.tensor.name == name) return &value;
    }
    return nullptr;
}

std::string UniqueName(const std::unordered_set<std::string>& used, const std::string& base) {
    if (used.count(base) == 0) return base;
    for (size_t suffix = 1;; ++suffix) {
        const std::string candidate = base + "_" + std::to_string(suffix);
        if (used.count(candidate) == 0) return candidate;
    }
}

void AddValue(ModelDesc* model, std::unordered_set<std::string>* used, ValueDesc value) {
    if (model == nullptr || used == nullptr || used->count(value.tensor.name) != 0) return;
    used->insert(value.tensor.name);
    model->graph.values.push_back(std::move(value));
}

ActivationQuantizationTable SelectActivations(const StaticQuantizationConfig& config) {
    if (!config.activations.empty()) return config.activations;
    if (!config.activation_table.empty()) return config.activation_table;
    return config.activation_quantization;
}

bool LookupActivation(const ActivationQuantizationTable& table, const std::string& name,
                      ActivationQuantization* value) {
    if (value == nullptr) return false;
    const auto it = table.find(name);
    if (it == table.end() || !IsValidActivation(it->second)) return false;
    *value = it->second;
    return true;
}

QuantizationParams ToParams(const ActivationQuantization& value) {
    QuantizationParams result;
    result.enabled = true;
    result.scale = value.scale;
    result.zero_point = value.zero_point;
    result.granularity = QuantizationGranularity::kPerTensor;
    result.axis = -1;
    result.scales = {value.scale};
    result.zero_points = {value.zero_point};
    return result;
}

float DecodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t result = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            float value = std::ldexp(static_cast<float>(fraction), -24);
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

float ReadFloat(const Tensor& tensor, size_t index) {
    const auto* raw = static_cast<const uint8_t*>(tensor.raw_data());
    switch (tensor.data_type()) {
        case DataType::FP32: {
            float value = 0.0f;
            std::memcpy(&value, raw + index * sizeof(float), sizeof(value));
            return value;
        }
        case DataType::FP16: {
            uint16_t bits = 0;
            std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
            return DecodeFloat16(bits);
        }
        case DataType::BF16: {
            uint16_t bits = 0;
            std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
            const uint32_t full = static_cast<uint32_t>(bits) << 16;
            float value = 0.0f;
            std::memcpy(&value, &full, sizeof(value));
            return value;
        }
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

bool ReadFloatValues(const std::shared_ptr<Tensor>& tensor, std::vector<float>* values) {
    if (tensor == nullptr || values == nullptr || !tensor->IsInitialized() || tensor->numel() <= 0 ||
        (tensor->data_type() != DataType::FP32 && tensor->data_type() != DataType::FP16 &&
         tensor->data_type() != DataType::BF16)) return false;
    values->resize(static_cast<size_t>(tensor->numel()));
    for (size_t i = 0; i < values->size(); ++i) (*values)[i] = ReadFloat(*tensor, i);
    return std::all_of(values->begin(), values->end(), [](float value) { return std::isfinite(value); });
}

std::shared_ptr<Tensor> MakeFloatTensor(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::shared_ptr<Tensor> MakeInt32Tensor(const std::vector<int32_t>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<int32_t>(values, dims);
    return tensor;
}

int64_t WeightQuantizationAxis(const NodeDesc& node, const TensorDesc& desc) {
    if (desc.dims.empty()) return -1;
    if (node.op_type == "FC") return desc.dims.size() == 2 ? 1 : 0;
    if (node.op_type == "Gemm") {
        auto it = node.attributes.find("transB");
        const bool trans_b = it != node.attributes.end() &&
                             std::get_if<int64_t>(&it->second) != nullptr &&
                             *std::get_if<int64_t>(&it->second) != 0;
        return trans_b ? 0 : static_cast<int64_t>(desc.dims.size() - 1);
    }
    if (node.op_type == "MatMul") return static_cast<int64_t>(desc.dims.size() - 1);
    return 0;
}

bool QuantizeWeight(const std::shared_ptr<Tensor>& source, const TensorDesc& desc,
                    const StaticQuantizationConfig& config, int64_t channel_axis,
                    QuantizationParams* params,
                    std::vector<int8_t>* values) {
    if (source == nullptr || params == nullptr || values == nullptr || Product(desc.dims) <= 0) return false;
    std::vector<float> source_values;
    if (!ReadFloatValues(source, &source_values) || source_values.size() != static_cast<size_t>(Product(desc.dims))) return false;
    const bool per_channel = config.per_channel_weights && channel_axis >= 0 &&
                              channel_axis < static_cast<int64_t>(desc.dims.size()) &&
                              desc.dims[static_cast<size_t>(channel_axis)] > 1;
    const size_t channels = per_channel ? static_cast<size_t>(desc.dims[static_cast<size_t>(channel_axis)]) : 1;
    params->enabled = true;
    params->granularity = per_channel ? QuantizationGranularity::kPerChannel : QuantizationGranularity::kPerTensor;
    params->axis = per_channel ? channel_axis : -1;
    params->block_size = 0;
    params->scales.assign(channels, 1.0f);
    params->zero_points.assign(channels, 0);
    std::vector<float> minimums(channels, std::numeric_limits<float>::max());
    std::vector<float> maximums(channels, std::numeric_limits<float>::lowest());
    for (size_t index = 0; index < source_values.size(); ++index) {
        const size_t channel = per_channel ? QuantizationParameterIndex(desc.dims, index, *params) : 0;
        minimums[channel] = std::min(minimums[channel], source_values[index]);
        maximums[channel] = std::max(maximums[channel], source_values[index]);
    }
    for (size_t channel = 0; channel < channels; ++channel) {
        const float minimum = minimums[channel];
        const float maximum = maximums[channel];
        float scale = 1.0f;
        int32_t zero_point = 0;
        if (config.symmetric) {
            const float maximum_abs = std::max(std::fabs(minimum), std::fabs(maximum));
            scale = maximum_abs == 0.0f ? 1.0f : maximum_abs / 127.0f;
        } else if (maximum != minimum) {
            scale = (maximum - minimum) / 255.0f;
            zero_point = static_cast<int32_t>(std::round(-128.0f - minimum / scale));
            zero_point = std::max(-128, std::min(127, zero_point));
        }
        if (!std::isfinite(scale) || scale <= 0.0f) return false;
        params->scales[channel] = scale;
        params->zero_points[channel] = zero_point;
    }
    params->scale = params->scales.front();
    params->zero_point = params->zero_points.front();
    values->resize(source_values.size());
    for (size_t i = 0; i < source_values.size(); ++i) {
        const size_t channel = per_channel ? QuantizationParameterIndex(desc.dims, i, *params) : 0;
        (*values)[i] = QuantizeInt8Value(source_values[i], params->scales[channel], params->zero_points[channel]);
    }
    return ValidateQuantizationParams(*params, desc.dims);
}

bool QuantizeBias(const std::shared_ptr<Tensor>& source, const TensorDesc& desc,
                  const ActivationQuantization& input_activation, const QuantizationParams& weight_params,
                  std::shared_ptr<Tensor>* result) {
    if (source == nullptr || result == nullptr || Product(desc.dims) <= 0 || !IsValidActivation(input_activation) ||
        weight_params.scales.empty()) return false;
    std::vector<float> source_values;
    if (!ReadFloatValues(source, &source_values)) return false;
    const bool per_channel = weight_params.scales.size() > 1;
    const size_t channels = weight_params.scales.size();
    // A per-tensor weight scale applies to every bias element. For
    // per-channel weights, a vector bias must match the channel count;
    // a scalar bias remains valid for the first channel for legacy graphs.
    if (per_channel && source_values.size() != channels && source_values.size() != 1) return false;
    std::vector<int32_t> values(source_values.size());
    for (size_t i = 0; i < source_values.size(); ++i) {
        const size_t channel = per_channel ? std::min(i, channels - 1) : 0;
        const double scale = static_cast<double>(input_activation.scale) * weight_params.scales[channel];
        if (!(scale > 0.0) || !std::isfinite(scale)) return false;
        const double rounded = std::round(static_cast<double>(source_values[i]) / scale);
        if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
            rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) return false;
        values[i] = static_cast<int32_t>(rounded);
    }
    *result = MakeInt32Tensor(values, desc.dims);
    return true;
}

void AddConstants(ModelDesc* model, std::unordered_set<std::string>* used,
                  std::unordered_map<std::string, std::shared_ptr<Tensor>>* weights,
                  const std::string& base, const QuantizationParams& params,
                  std::string* scale_name, std::string* zero_name) {
    *scale_name = UniqueName(*used, base + "__int8_scale");
    *zero_name = UniqueName(*used, base + "__int8_zero_point");
    const std::vector<int64_t> dims{static_cast<int64_t>(params.scales.size())};
    ValueDesc scale;
    scale.tensor.name = *scale_name;
    scale.tensor.dims = dims;
    scale.tensor.data_type = DataType::FP32;
    scale.constant = true;
    AddValue(model, used, scale);
    (*weights)[*scale_name] = MakeFloatTensor(params.scales, dims);
    ValueDesc zero;
    zero.tensor.name = *zero_name;
    zero.tensor.dims = dims;
    zero.tensor.data_type = DataType::INT32;
    zero.constant = true;
    AddValue(model, used, zero);
    (*weights)[*zero_name] = MakeInt32Tensor(params.zero_points, dims);
}

bool IsStandardInt8Node(const std::string& op_type) {
    if (op_type == "Shape" || op_type == "ResizeConcat" || op_type == "YoloDecode") return true;

    return op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div" ||
           op_type == "ReLU" || op_type == "Relu" || op_type == "Neg" || op_type == "Sigmoid" ||
           op_type == "SiLU" || op_type == "Silu" || op_type == "Exp" || op_type == "Sqrt" ||
           op_type == "Tanh" || op_type == "Erf" || op_type == "Sin" || op_type == "Cos" ||
           op_type == "Softplus" || op_type == "Pow" || op_type == "BatchNormalization" ||
           op_type == "AvgPool" || op_type == "AveragePool" || op_type == "MaxPool" ||
           op_type == "GlobalAveragePool" ||
           op_type == "ReduceSum" || op_type == "ReduceMean" || op_type == "Identity" ||
           op_type == "Reshape" || op_type == "Flatten" || op_type == "Transpose" ||
           op_type == "Squeeze" || op_type == "Unsqueeze" || op_type == "Slice" ||
           op_type == "Split" || op_type == "Concat" || op_type == "Expand" || op_type == "Gather" ||
           op_type == "Where" || op_type == "Resize" || op_type == "Softmax" || op_type == "Equal" ||
           op_type == "Cast" || op_type == "ConstantOfShape";
}

bool IsStandardControlInput(const std::string& op_type, size_t index) {
    if (op_type == "YoloDecode") return index > 0;

    if (op_type == "Where") return index == 0;
    if (op_type == "BatchNormalization") return index > 0;
    if (op_type == "Reshape" || op_type == "Expand" || op_type == "Gather") return index > 0;
    if (op_type == "Slice") return index > 0;
    if (op_type == "Split") return index > 0;
    if (op_type == "Resize") return index > 0;
    if (op_type == "Squeeze" || op_type == "Unsqueeze") return index > 0;
    if (op_type == "ConstantOfShape") return true;
    if (op_type == "Pow") return index > 0;
    return false;
}

bool StandardOutputNeedsInt8(const NodeDesc& node, const ValueDesc* output_value) {
    if (node.op_type == "Shape") return false;
    if (node.op_type == "ResizeConcat" || node.op_type == "YoloDecode") return true;

    if (node.op_type == "Equal") return false;
    if (node.op_type == "Cast") return false;
    return true;
}

bool ConvertStandardInt8Node(
    const ModelDesc& input_model, ModelDesc* converted, const ActivationQuantizationTable& activations,
    std::unordered_set<std::string>* used, std::unordered_map<std::string, std::shared_ptr<Tensor>>* weights,
    const NodeDesc& original, const std::string& label,
    std::vector<NodeDesc>* nodes, std::string* diagnostic) {
    if (converted == nullptr || used == nullptr || weights == nullptr || nodes == nullptr || original.outputs.empty()) {
        if (diagnostic != nullptr) *diagnostic = "malformed standard operator";
        return false;
    }

    std::vector<ActivationQuantization> input_quantizations(original.inputs.size());
    std::vector<const ValueDesc*> input_values(original.inputs.size(), nullptr);
    for (size_t index = 0; index < original.inputs.size(); ++index) {
        if (original.inputs[index].empty() || IsStandardControlInput(original.op_type, index)) continue;
        if (!LookupActivation(activations, original.inputs[index], &input_quantizations[index])) {
            if (diagnostic != nullptr) *diagnostic = "missing activation scale for input " + original.inputs[index];
            return false;
        }
        input_values[index] = FindValue(input_model, original.inputs[index]);
        if (input_values[index] == nullptr) {
            if (diagnostic != nullptr) *diagnostic = "missing tensor description for input " + original.inputs[index];
            return false;
        }
    }

    std::vector<ActivationQuantization> output_quantizations(original.outputs.size());
    std::vector<const ValueDesc*> output_values(original.outputs.size(), nullptr);
    for (size_t index = 0; index < original.outputs.size(); ++index) {
        output_values[index] = FindValue(input_model, original.outputs[index]);
        if (output_values[index] == nullptr) {
            if (diagnostic != nullptr) *diagnostic = "missing tensor description for output " + original.outputs[index];
            return false;
        }
    }
    if (original.op_type == "Cast" && output_values.front()->tensor.data_type == DataType::INT8) {
        if (diagnostic != nullptr) *diagnostic = "Cast to INT8 remains an explicit non-affine conversion";
        return false;
    }
    const bool quantized_outputs = StandardOutputNeedsInt8(original, output_values.front());
    if (quantized_outputs) {
        for (size_t index = 0; index < original.outputs.size(); ++index) {
            if (!LookupActivation(activations, original.outputs[index], &output_quantizations[index])) {
                if (diagnostic != nullptr) *diagnostic = "missing activation scale for output " + original.outputs[index];
                return false;
            }
        }
    }

    NodeDesc arithmetic = original;
    std::vector<std::string> input_scale_names(original.inputs.size());
    std::vector<std::string> input_zero_names(original.inputs.size());
    for (size_t index = 0; index < original.inputs.size(); ++index) {
        if (original.inputs[index].empty() || IsStandardControlInput(original.op_type, index)) continue;
        const QuantizationParams params = ToParams(input_quantizations[index]);
        std::string scale_name;
        std::string zero_name;
        AddConstants(converted, used, weights, original.inputs[index], params, &scale_name, &zero_name);
        input_scale_names[index] = scale_name;
        input_zero_names[index] = zero_name;
        const std::string q_input_name = UniqueName(*used, label + "__int8_input_" + std::to_string(index));
        ValueDesc q_input = *input_values[index];
        q_input.tensor.name = q_input_name;
        q_input.tensor.data_type = DataType::INT8;
        q_input.tensor.quantization = params;
        q_input.constant = false;
        q_input.weight = {};
        AddValue(converted, used, q_input);
        NodeDesc quantize;
        quantize.name = UniqueName(*used, label + "__QuantizeLinear_" + std::to_string(index));
        used->insert(quantize.name);
        quantize.op_type = "QuantizeLinear";
        quantize.inputs = {original.inputs[index], scale_name, zero_name};
        quantize.outputs = {q_input_name};
        nodes->push_back(std::move(quantize));
        arithmetic.inputs[index] = q_input_name;
    }

    std::vector<std::string> output_scale_names(original.outputs.size());
    std::vector<std::string> output_zero_names(original.outputs.size());
    std::vector<std::string> quantized_output_names(original.outputs.size());
    if (quantized_outputs) {
        for (size_t index = 0; index < original.outputs.size(); ++index) {
            const QuantizationParams params = ToParams(output_quantizations[index]);
            if (output_values[index]->tensor.data_type == DataType::INT8) {
                if (auto* output_desc = FindValue(converted, original.outputs[index]); output_desc != nullptr) {
                    output_desc->tensor.data_type = DataType::INT8;
                    output_desc->tensor.quantization = params;
                }
                arithmetic.outputs[index] = original.outputs[index];
                continue;
            }
            std::string scale_name;
            std::string zero_name;
            AddConstants(converted, used, weights, original.outputs[index], params, &scale_name, &zero_name);
            output_scale_names[index] = scale_name;
            output_zero_names[index] = zero_name;
            const std::string q_output_name = UniqueName(*used, label + "__int8_output_" + std::to_string(index));
            ValueDesc q_output = *output_values[index];
            q_output.tensor.name = q_output_name;
            q_output.tensor.data_type = DataType::INT8;
            q_output.tensor.quantization = params;
            q_output.constant = false;
            q_output.weight = {};
            AddValue(converted, used, q_output);
            arithmetic.outputs[index] = q_output_name;
            quantized_output_names[index] = q_output_name;
        }
    }
    const std::vector<std::string> arithmetic_outputs = arithmetic.outputs;
    nodes->push_back(std::move(arithmetic));

    if (quantized_outputs) {
        for (size_t index = 0; index < original.outputs.size(); ++index) {
            if (quantized_output_names[index].empty()) continue;
            NodeDesc dequantize;
            dequantize.name = UniqueName(*used, label + "__DequantizeLinear_" + std::to_string(index));
            used->insert(dequantize.name);
            dequantize.op_type = "DequantizeLinear";
            dequantize.inputs = {arithmetic_outputs[index], output_scale_names[index], output_zero_names[index]};
            dequantize.outputs = {original.outputs[index]};
            dequantize.attributes["to"] = int64_t{1};
            nodes->push_back(std::move(dequantize));
        }
    }
    return true;
}

void FuseRedundantInt8Boundaries(ModelDesc* model) {
    if (model == nullptr) return;
    auto& nodes = model->graph.nodes;
    std::unordered_map<std::string, std::vector<size_t>> consumers;
    for (size_t index = 0; index < nodes.size(); ++index) {
        for (const auto& input : nodes[index].inputs) {
            if (!input.empty()) consumers[input].push_back(index);
        }
    }
    const auto same_quantization = [](const ValueDesc* lhs, const ValueDesc* rhs) {
        if (lhs == nullptr || rhs == nullptr || lhs->tensor.data_type != DataType::INT8 ||
            rhs->tensor.data_type != DataType::INT8) return false;
        const auto& lhs_q = lhs->tensor.quantization;
        const auto& rhs_q = rhs->tensor.quantization;
        return lhs_q.enabled == rhs_q.enabled && lhs_q.scale == rhs_q.scale &&
               lhs_q.zero_point == rhs_q.zero_point && lhs_q.granularity == rhs_q.granularity &&
               lhs_q.axis == rhs_q.axis && lhs_q.block_size == rhs_q.block_size &&
               lhs_q.scales == rhs_q.scales && lhs_q.zero_points == rhs_q.zero_points;
    };
    const auto is_graph_output = [model](const std::string& name) {
        return std::find(model->graph.outputs.begin(), model->graph.outputs.end(), name) !=
               model->graph.outputs.end();
    };

    std::vector<bool> removed(nodes.size(), false);
    for (size_t dequantize_index = 0; dequantize_index < nodes.size(); ++dequantize_index) {
        const NodeDesc& dequantize = nodes[dequantize_index];
        if (dequantize.op_type != "DequantizeLinear" || dequantize.inputs.empty() ||
            dequantize.outputs.size() != 1) continue;
        const auto consumer_it = consumers.find(dequantize.outputs.front());
        if (consumer_it == consumers.end()) continue;
        const ValueDesc* producer_int8 = FindValue(*model, dequantize.inputs.front());
        if (producer_int8 == nullptr) continue;
        for (const size_t quantize_index : consumer_it->second) {
            if (quantize_index >= nodes.size()) continue;
            NodeDesc& quantize = nodes[quantize_index];
            if (quantize.op_type != "QuantizeLinear" || quantize.inputs.size() < 1 ||
                quantize.outputs.size() != 1) continue;
            const auto quantized_consumer_it = consumers.find(quantize.outputs.front());
            if (quantized_consumer_it == consumers.end() || quantized_consumer_it->second.size() != 1) continue;
            const size_t arithmetic_index = quantized_consumer_it->second.front();
            if (arithmetic_index >= nodes.size() || arithmetic_index == quantize_index ||
                nodes[arithmetic_index].op_type == "QuantizeLinear" ||
                nodes[arithmetic_index].op_type == "DequantizeLinear") continue;
            const ValueDesc* consumer_int8 = FindValue(*model, quantize.outputs.front());
            if (!same_quantization(producer_int8, consumer_int8)) continue;
            NodeDesc& arithmetic = nodes[arithmetic_index];
            bool rewired = false;
            for (auto& input : arithmetic.inputs) {
                if (input == quantize.outputs.front()) {
                    input = dequantize.inputs.front();
                    rewired = true;
                }
            }
            if (rewired) removed[quantize_index] = true;
        }
    }
    for (size_t dequantize_index = 0; dequantize_index < nodes.size(); ++dequantize_index) {
        if (removed[dequantize_index]) continue;
        const NodeDesc& dequantize = nodes[dequantize_index];
        if (dequantize.op_type != "DequantizeLinear" || dequantize.outputs.size() != 1 ||
            is_graph_output(dequantize.outputs.front())) continue;
        const auto consumer_it = consumers.find(dequantize.outputs.front());
        bool has_live_consumer = false;
        if (consumer_it != consumers.end()) {
            for (const size_t consumer : consumer_it->second) {
                if (consumer < removed.size() && !removed[consumer]) {
                    has_live_consumer = true;
                    break;
                }
            }
        }
        if (!has_live_consumer) removed[dequantize_index] = true;
    }
    std::vector<NodeDesc> fused_nodes;
    fused_nodes.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        if (!removed[index]) fused_nodes.push_back(std::move(nodes[index]));
    }
    nodes = std::move(fused_nodes);
}

}  // namespace

int32_t LoadActivationQuantizationTable(const std::string& path, ActivationQuantizationTable* table,
                                        std::vector<std::string>* diagnostics) {
    if (table == nullptr) return -1;
    table->clear();
    std::ifstream input(path);
    if (!input.good()) {
        if (diagnostics != nullptr) diagnostics->push_back("unable to open scale table: " + path);
        return -1;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream stream(line);
        std::string name;
        float scale = 0.0f;
        int32_t zero_point = 0;
        if (!(stream >> name)) continue;
        if (!(stream >> scale >> zero_point)) {
            if (diagnostics != nullptr) diagnostics->push_back("malformed scale row at line " + std::to_string(line_number));
            table->clear();
            return -1;
        }
        std::string extra;
        if (stream >> extra || !std::isfinite(scale) || scale <= 0.0f || zero_point < -128 || zero_point > 127 ||
            table->count(name) != 0) {
            if (diagnostics != nullptr) diagnostics->push_back("invalid scale row at line " + std::to_string(line_number));
            table->clear();
            return -1;
        }
        (*table)[name] = ActivationQuantization{scale, zero_point};
    }
    return input.good() || input.eof() ? 0 : -1;
}

int32_t StaticQuantizeModel(const ModelDesc& input_model,
                            const std::unordered_map<std::string, std::shared_ptr<Tensor>>& source_weights,
                            const StaticQuantizationConfig& config,
                            ModelDesc* output_model,
                            std::unordered_map<std::string, std::shared_ptr<Tensor>>* output_weights,
                            StaticQuantizationReport* report) {
    if (output_model == nullptr || output_weights == nullptr) return -1;
    StaticQuantizationReport working;
    ModelDesc converted = input_model;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> weights = source_weights;
    const auto activations = SelectActivations(config);
    for (const auto& value : input_model.graph.values) {
        if (!value.constant) continue;
        const auto it = source_weights.find(value.tensor.name);
        if (it == source_weights.end() || it->second == nullptr || !it->second->IsInitialized()) {
            working.diagnostics.push_back("missing constant tensor: " + value.tensor.name);
            if (config.strict) {
                if (report != nullptr) *report = working;
                return -1;
            }
        }
    }
    std::unordered_set<std::string> used;
    for (const auto& value : converted.graph.values) used.insert(value.tensor.name);
    for (const auto& node : converted.graph.nodes) if (!node.name.empty()) used.insert(node.name);
    std::vector<NodeDesc> nodes;
    nodes.reserve(input_model.graph.nodes.size() * 3);
    for (size_t index = 0; index < input_model.graph.nodes.size(); ++index) {
        const NodeDesc& original = input_model.graph.nodes[index];
        const std::string label = NodeLabel(original, index);
        if (KeepNodeInFp32(config, original, index)) {
            working.skipped_nodes.push_back(label);
            working.diagnostics.push_back("kept FP32 node: " + label);
            nodes.push_back(original);
            continue;
        }
        if (IsStandardInt8Node(original.op_type)) {
            std::string diagnostic;
            if (!ConvertStandardInt8Node(input_model, &converted, activations, &used, &weights, original, label, &nodes,
                                         &diagnostic)) {
                working.skipped_nodes.push_back(label);
                working.diagnostics.push_back(diagnostic.empty() ? "unable to quantize standard node: " + label
                                                                  : diagnostic + ": " + label);
                if (config.strict) {
                    if (report != nullptr) *report = working;
                    return -1;
                }
                nodes.push_back(original);
                continue;
            }
            working.quantized_nodes.push_back(label);
            continue;
        }
        if (!IsQuantizable(original.op_type)) {
            nodes.push_back(original);
            continue;
        }
        if (original.inputs.size() < 2 || original.outputs.empty()) {
            working.skipped_nodes.push_back(label);
            working.diagnostics.push_back("cannot quantize malformed node: " + label);
            if (config.strict) {
                if (report != nullptr) *report = working;
                return -1;
            }
            nodes.push_back(original);
            continue;
        }
        ActivationQuantization input_activation;
        ActivationQuantization output_activation;
        if (!LookupActivation(activations, original.inputs[0], &input_activation) ||
            !LookupActivation(activations, original.outputs[0], &output_activation)) {
            working.skipped_nodes.push_back(label);
            working.diagnostics.push_back("missing activation scale for node: " + label);
            if (config.strict) {
                if (report != nullptr) *report = working;
                return -1;
            }
            nodes.push_back(original);
            continue;
        }
        const ValueDesc* input_value = FindValue(input_model, original.inputs[0]);
        const ValueDesc* output_value = FindValue(input_model, original.outputs[0]);
        const ValueDesc* weight_value = FindValue(input_model, original.inputs[1]);
        const auto weight_it = source_weights.find(original.inputs[1]);
        if (input_value == nullptr || output_value == nullptr || weight_value == nullptr ||
            weight_it == source_weights.end() || weight_it->second == nullptr) {
            working.skipped_nodes.push_back(label);
            working.diagnostics.push_back("missing tensor description or weight for node: " + label);
            if (config.strict) {
                if (report != nullptr) *report = working;
                return -1;
            }
            nodes.push_back(original);
            continue;
        }
        QuantizationParams weight_params;
        std::vector<int8_t> quantized_weight;
        const int64_t weight_axis = WeightQuantizationAxis(original, weight_value->tensor);
        if (!QuantizeWeight(weight_it->second, weight_value->tensor, config, weight_axis,
                            &weight_params, &quantized_weight)) {
            working.skipped_nodes.push_back(label);
            working.diagnostics.push_back("invalid weight tensor for node: " + label);
            if (config.strict) {
                if (report != nullptr) *report = working;
                return -1;
            }
            nodes.push_back(original);
            continue;
        }
        auto weight_tensor = std::make_shared<Tensor>();
        weight_tensor->Assign<int8_t>(quantized_weight, weight_value->tensor.dims);
        weight_tensor->set_quantization(weight_params);
        std::shared_ptr<Tensor> bias_tensor;
        const bool has_bias = original.inputs.size() >= 3 && !original.inputs[2].empty();
        if (has_bias) {
            const auto bias_it = source_weights.find(original.inputs[2]);
            const ValueDesc* bias_value = FindValue(input_model, original.inputs[2]);
            if (bias_it == source_weights.end() || bias_it->second == nullptr || bias_value == nullptr) {
                working.skipped_nodes.push_back(label);
                working.diagnostics.push_back("missing bias tensor for node: " + label);
                if (config.strict) {
                    if (report != nullptr) *report = working;
                    return -1;
                }
                nodes.push_back(original);
                continue;
            }
            if (!QuantizeBias(bias_it->second, bias_value->tensor, input_activation, weight_params, &bias_tensor)) {
                working.skipped_nodes.push_back(label);
                working.diagnostics.push_back("invalid bias tensor for node: " + label);
                if (config.strict) {
                    if (report != nullptr) *report = working;
                    return -1;
                }
                nodes.push_back(original);
                continue;
            }
        }
        // Apply tensor and model-description mutations only after all per-node
        // validation succeeds. In non-strict mode a skipped node must retain
        // its original floating-point constants.
        weights[original.inputs[1]] = weight_tensor;
        if (auto* desc = FindValue(&converted, original.inputs[1]); desc != nullptr) {
            desc->tensor.data_type = DataType::INT8;
            desc->tensor.quantization = weight_params;
        }
        if (has_bias) {
            weights[original.inputs[2]] = bias_tensor;
            if (auto* desc = FindValue(&converted, original.inputs[2]); desc != nullptr) {
                desc->tensor.data_type = DataType::INT32;
                desc->tensor.quantization = QuantizationParams{};
            }
        }
        const QuantizationParams input_params = ToParams(input_activation);
        const QuantizationParams output_params = ToParams(output_activation);
        std::string input_scale_name;
        std::string input_zero_name;
        std::string output_scale_name;
        std::string output_zero_name;
        AddConstants(&converted, &used, &weights, original.inputs[0], input_params, &input_scale_name, &input_zero_name);
        AddConstants(&converted, &used, &weights, original.outputs[0], output_params, &output_scale_name, &output_zero_name);
        const std::string q_input_name = UniqueName(used, label + "__int8_input");
        const std::string q_output_name = UniqueName(used, label + "__int8_output");
        ValueDesc q_input = *input_value;
        q_input.tensor.name = q_input_name;
        q_input.tensor.data_type = DataType::INT8;
        q_input.tensor.quantization = input_params;
        q_input.constant = false;
        q_input.weight = {};
        AddValue(&converted, &used, q_input);
        ValueDesc q_output = *output_value;
        q_output.tensor.name = q_output_name;
        q_output.tensor.data_type = DataType::INT8;
        q_output.tensor.quantization = output_params;
        q_output.constant = false;
        q_output.weight = {};
        AddValue(&converted, &used, q_output);
        NodeDesc quantize;
        quantize.name = UniqueName(used, label + "__QuantizeLinear");
        used.insert(quantize.name);
        quantize.op_type = "QuantizeLinear";
        quantize.inputs = {original.inputs[0], input_scale_name, input_zero_name};
        quantize.outputs = {q_input_name};
        NodeDesc arithmetic = original;
        arithmetic.inputs[0] = q_input_name;
        arithmetic.outputs[0] = q_output_name;
        NodeDesc dequantize;
        dequantize.name = UniqueName(used, label + "__DequantizeLinear");
        used.insert(dequantize.name);
        dequantize.op_type = "DequantizeLinear";
        dequantize.inputs = {q_output_name, output_scale_name, output_zero_name};
        dequantize.outputs = {original.outputs[0]};
        dequantize.attributes["to"] = int64_t{1};
        nodes.push_back(std::move(quantize));
        nodes.push_back(std::move(arithmetic));
        nodes.push_back(std::move(dequantize));
        working.quantized_nodes.push_back(label);
    }
    converted.graph.nodes = std::move(nodes);
    FuseRedundantInt8Boundaries(&converted);
    *output_model = std::move(converted);
    *output_weights = std::move(weights);
    if (report != nullptr) *report = std::move(working);
    return 0;
}

}  // namespace feather
