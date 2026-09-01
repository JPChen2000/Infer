#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "model/model_io.h"
#include "quant/static_quantizer.h"

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::StaticGraph;
using feather::Tensor;
using feather::ActivationQuantization;
using feather::StaticQuantizationConfig;
using feather::StaticQuantizationReport;
using feather::model::ModelDesc;
using feather::model::ModelLoader;
using feather::model::ModelWriter;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

constexpr const char* kModelPath = "/tmp/feather_static_quantization_e2e.fth";

ValueDesc Value(const std::string& name, const std::vector<int64_t>& dims, DataType type, bool constant = false) {
    ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = type;
    value.constant = constant;
    return value;
}

ModelDesc BuildModel() {
    ModelDesc model;
    model.name = "static_quantization_e2e";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"probabilities"};
    model.graph.values = {
        Value("input", {1, 1, 3, 3}, DataType::FP32),
        Value("conv_weight", {1, 1, 1, 1}, DataType::FP32, true),
        Value("conv_bias", {1}, DataType::FP32, true),
        Value("conv_out", {1, 1, 3, 3}, DataType::FP32),
        Value("bn_scale", {1}, DataType::FP32, true),
        Value("bn_bias", {1}, DataType::FP32, true),
        Value("bn_mean", {1}, DataType::FP32, true),
        Value("bn_var", {1}, DataType::FP32, true),
        Value("bn_out", {1, 1, 3, 3}, DataType::FP32),
        Value("relu_out", {1, 1, 3, 3}, DataType::FP32),
        Value("pool_out", {1, 1, 2, 2}, DataType::FP32),
        Value("flat_out", {1, 4}, DataType::FP32),
        Value("fc_weight", {4, 2}, DataType::FP32, true),
        Value("fc_bias", {2}, DataType::FP32, true),
        Value("logits", {1, 2}, DataType::FP32),
        Value("probabilities", {1, 2}, DataType::FP32),
    };

    NodeDesc conv;
    conv.name = "conv";
    conv.op_type = "Conv2D";
    conv.inputs = {"input", "conv_weight", "conv_bias"};
    conv.outputs = {"conv_out"};
    conv.attributes["stride_h"] = int64_t{1};
    conv.attributes["stride_w"] = int64_t{1};
    conv.attributes["pad_h"] = int64_t{0};
    conv.attributes["pad_w"] = int64_t{0};
    conv.attributes["dilation_h"] = int64_t{1};
    conv.attributes["dilation_w"] = int64_t{1};
    conv.attributes["group"] = int64_t{1};

    NodeDesc bn;
    bn.name = "batch_norm";
    bn.op_type = "BatchNormalization";
    bn.inputs = {"conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"};
    bn.outputs = {"bn_out"};
    bn.attributes["epsilon"] = 1.0e-5f;

    NodeDesc relu;
    relu.name = "relu";
    relu.op_type = "ReLU";
    relu.inputs = {"bn_out"};
    relu.outputs = {"relu_out"};

    NodeDesc pool;
    pool.name = "pool";
    pool.op_type = "MaxPool";
    pool.inputs = {"relu_out"};
    pool.outputs = {"pool_out"};
    pool.attributes["kernel_h"] = int64_t{2};
    pool.attributes["kernel_w"] = int64_t{2};
    pool.attributes["stride_h"] = int64_t{1};
    pool.attributes["stride_w"] = int64_t{1};
    pool.attributes["pad_h"] = int64_t{0};
    pool.attributes["pad_w"] = int64_t{0};

    NodeDesc flatten;
    flatten.name = "flatten";
    flatten.op_type = "Flatten";
    flatten.inputs = {"pool_out"};
    flatten.outputs = {"flat_out"};
    flatten.attributes["axis"] = int64_t{1};

    NodeDesc gemm;
    gemm.name = "classifier";
    gemm.op_type = "Gemm";
    gemm.inputs = {"flat_out", "fc_weight", "fc_bias"};
    gemm.outputs = {"logits"};

    NodeDesc softmax;
    softmax.name = "probability";
    softmax.op_type = "Softmax";
    softmax.inputs = {"logits"};
    softmax.outputs = {"probabilities"};
    softmax.attributes["axis"] = int64_t{1};

    model.graph.nodes = {conv, bn, relu, pool, flatten, gemm, softmax};
    return model;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> BuildWeights() {
    auto conv_weight = std::make_shared<Tensor>();
    conv_weight->Assign<float>({1.0f}, {1, 1, 1, 1});
    auto conv_bias = std::make_shared<Tensor>();
    conv_bias->Assign<float>({0.1f}, {1});
    auto bn_scale = std::make_shared<Tensor>();
    bn_scale->Assign<float>({1.0f}, {1});
    auto bn_bias = std::make_shared<Tensor>();
    bn_bias->Assign<float>({0.0f}, {1});
    auto bn_mean = std::make_shared<Tensor>();
    bn_mean->Assign<float>({0.0f}, {1});
    auto bn_var = std::make_shared<Tensor>();
    bn_var->Assign<float>({1.0f}, {1});
    auto fc_weight = std::make_shared<Tensor>();
    fc_weight->Assign<float>({0.5f, -0.25f, 0.3f, 0.6f, -0.4f, 0.2f, 0.7f, -0.1f}, {4, 2});
    auto fc_bias = std::make_shared<Tensor>();
    fc_bias->Assign<float>({0.05f, -0.1f}, {2});
    return {{"conv_weight", conv_weight}, {"conv_bias", conv_bias}, {"bn_scale", bn_scale},
            {"bn_bias", bn_bias}, {"bn_mean", bn_mean}, {"bn_var", bn_var}, {"fc_weight", fc_weight},
            {"fc_bias", fc_bias}};
}

std::shared_ptr<Tensor> BuildFloatInput() {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({0.7f, -0.2f, 0.4f, 1.1f, -0.3f, 0.8f, 0.5f, 1.2f, -0.6f}, {1, 1, 3, 3});
    return input;
}

const ValueDesc* FindValue(const ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) return &value;
    }
    return nullptr;
}

int32_t RunModel(const ModelDesc& model,
                 const std::unordered_map<std::string, std::shared_ptr<Tensor>>& weights,
                 const std::shared_ptr<Tensor>& input,
                 DeviceType device,
                 std::shared_ptr<Tensor>* output) {
    if (output == nullptr) return -1;
    StaticGraph static_graph;
    static_graph.SetKernelDevice(device);
    if (static_graph.SetModel(model) != 0 || static_graph.SetTensor("input", input) != 0) return -1;
    for (const auto& item : weights) {
        if (static_graph.SetTensor(item.first, item.second) != 0) return -1;
    }
    if (static_graph.Build() != 0) return -1;

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    if (lowering.Lower(static_graph, &runtime_graph) != 0 || runtime_graph.Run() != 0) return -1;
    *output = runtime_graph.GetTensor("probabilities");
    return *output == nullptr ? -1 : 0;
}

TEST(static_quantization_e2e_test, QuantizesSavesReloadsAndRunsWithCommonAndX86) {
    const ModelDesc source_model = BuildModel();
    const auto source_weights = BuildWeights();
    const auto source_input = BuildFloatInput();

    std::shared_ptr<Tensor> fp32_output;
    ASSERT_EQ(RunModel(source_model, source_weights, source_input, DeviceType::COMMON, &fp32_output), 0);
    ASSERT_EQ(fp32_output->data_type(), DataType::FP32);

    StaticQuantizationConfig config;
    config.strict = true;
    config.symmetric = false;
    config.per_channel_weights = true;
    for (const auto& name : {"input", "conv_out", "bn_out", "relu_out", "pool_out", "flat_out", "logits",
                             "probabilities", "conv", "batch_norm", "relu", "pool", "flatten", "classifier",
                             "probability"}) {
        config.activations[name] = ActivationQuantization{0.01f, -3};
    }

    ModelDesc quantized_model;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> quantized_weights;
    StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(source_model, source_weights, config, &quantized_model,
                                           &quantized_weights, &report), 0);
    for (const auto& name : {"conv", "batch_norm", "relu", "pool", "flatten", "classifier", "probability"}) {
        EXPECT_NE(std::find(report.quantized_nodes.begin(), report.quantized_nodes.end(), name),
                  report.quantized_nodes.end()) << name;
    }

    const auto* quantized_weight_desc = FindValue(quantized_model, "conv_weight");
    const auto* quantized_bias_desc = FindValue(quantized_model, "conv_bias");
    ASSERT_NE(quantized_weight_desc, nullptr);
    ASSERT_NE(quantized_bias_desc, nullptr);
    EXPECT_EQ(quantized_weight_desc->tensor.data_type, DataType::INT8);
    EXPECT_EQ(quantized_bias_desc->tensor.data_type, DataType::INT32);
    EXPECT_TRUE(quantized_weight_desc->tensor.quantization.enabled);

    bool has_quantize = false;
    bool has_dequantize = false;
    std::unordered_set<std::string> standard_ops = {"Conv2D", "BatchNormalization", "ReLU", "MaxPool", "Flatten", "Gemm", "Softmax"};
    for (const auto& node : quantized_model.graph.nodes) {
        has_quantize = has_quantize || node.op_type == "QuantizeLinear";
        has_dequantize = has_dequantize || node.op_type == "DequantizeLinear";
        if (standard_ops.count(node.op_type) != 0) {
            for (size_t index = 0; index < node.inputs.size(); ++index) {
                const auto* value = FindValue(quantized_model, node.inputs[index]);
                ASSERT_NE(value, nullptr) << node.name << " input " << index << " " << node.inputs[index];
                if (node.op_type == "Conv2D" || node.op_type == "Gemm") {
                    if (index == 0 || index == 1) {
                        EXPECT_EQ(value->tensor.data_type, DataType::INT8) << node.name << " input " << index;
                    } else if (index == 2) {
                        EXPECT_EQ(value->tensor.data_type, DataType::INT32) << node.name << " bias";
                    }
                } else if (node.op_type == "BatchNormalization") {
                    if (index == 0) {
                        EXPECT_EQ(value->tensor.data_type, DataType::INT8) << node.name << " data input";
                    } else {
                        EXPECT_EQ(value->tensor.data_type, DataType::FP32) << node.name << " parameter " << index;
                    }
                } else {
                    EXPECT_EQ(value->tensor.data_type, DataType::INT8) << node.name << " data input";
                }
            }
            for (const auto& output : node.outputs) {
                const auto* value = FindValue(quantized_model, output);
                ASSERT_NE(value, nullptr) << node.name << " output " << output;
                EXPECT_EQ(value->tensor.data_type, DataType::INT8) << node.name << " output " << output;
            }
        }
        if (node.op_type == "QuantizeLinear") {
            ASSERT_EQ(node.inputs.size(), 3U);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[0])->tensor.data_type, DataType::FP32);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[1])->tensor.data_type, DataType::FP32);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[2])->tensor.data_type, DataType::INT32);
            EXPECT_EQ(FindValue(quantized_model, node.outputs[0])->tensor.data_type, DataType::INT8);
        }
        if (node.op_type == "DequantizeLinear") {
            ASSERT_EQ(node.inputs.size(), 3U);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[0])->tensor.data_type, DataType::INT8);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[1])->tensor.data_type, DataType::FP32);
            EXPECT_EQ(FindValue(quantized_model, node.inputs[2])->tensor.data_type, DataType::INT32);
            EXPECT_EQ(FindValue(quantized_model, node.outputs[0])->tensor.data_type, DataType::FP32);
        }
    }
    EXPECT_TRUE(has_quantize);
    EXPECT_TRUE(has_dequantize);

    ASSERT_TRUE((ModelWriter{}).Save(kModelPath, quantized_model, quantized_weights));
    ModelLoader loader;
    ASSERT_TRUE(loader.Load(kModelPath));
    ASSERT_EQ(loader.model().graph.nodes.size(), quantized_model.graph.nodes.size());

    std::unordered_map<std::string, std::shared_ptr<Tensor>> loaded_weights;
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) continue;
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        loaded_weights[value.tensor.name] = tensor;
    }

    for (const DeviceType device : {DeviceType::COMMON, DeviceType::X86}) {
        std::shared_ptr<Tensor> int8_output;
        ASSERT_EQ(RunModel(loader.model(), loaded_weights, source_input, device, &int8_output), 0)
            << static_cast<int>(device);
        ASSERT_EQ(int8_output->data_type(), DataType::FP32);
        ASSERT_EQ(int8_output->numel(), fp32_output->numel());
        for (int64_t index = 0; index < int8_output->numel(); ++index) {
            EXPECT_NEAR(int8_output->data<float>()[index], fp32_output->data<float>()[index], 0.2f)
                << "device=" << static_cast<int>(device) << " index=" << index;
        }
    }

    std::remove(kModelPath);
}

}  // namespace
