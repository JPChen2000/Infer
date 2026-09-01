#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "quant/static_quantizer.h"

namespace {

feather::model::ValueDesc Value(const std::string& name, const std::vector<int64_t>& dims,
                                feather::DataType dtype, bool constant) {
    feather::model::ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = dtype;
    value.constant = constant;
    return value;
}

struct Fixture {
    feather::model::ModelDesc model;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
};

Fixture MakeFixture() {
    Fixture fixture;
    fixture.model.name = "static_int8_fixture";
    fixture.model.version = 1;
    fixture.model.graph.name = "main";
    fixture.model.graph.inputs = {"input"};
    fixture.model.graph.outputs = {"output"};
    fixture.model.graph.values = {
        Value("input", {1, 2}, feather::DataType::FP32, false),
        Value("weight", {2, 2}, feather::DataType::FP32, true),
        Value("bias", {2}, feather::DataType::FP32, true),
        Value("output", {1, 2}, feather::DataType::FP32, false),
    };
    fixture.model.graph.nodes = {{"fc", "FC", "", {"input", "weight", "bias"}, {"output"}, {}}};
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<float>({0.5f, -1.0f, 2.0f, 0.25f}, {2, 2});
    auto bias = std::make_shared<feather::Tensor>();
    bias->Assign<float>({1.0f, -2.0f}, {2});
    fixture.weights["weight"] = weight;
    fixture.weights["bias"] = bias;
    return fixture;
}

feather::StaticQuantizationConfig Config() {
    feather::StaticQuantizationConfig config;
    config.activations["input"] = {0.5f, 0};
    config.activations["output"] = {1.0f, 0};
    return config;
}

int64_t QuantizedWeightAxis(const std::string& op_type, const std::vector<int64_t>& dims, bool trans_b) {
    auto fixture = MakeFixture();
    fixture.model.graph.nodes[0].op_type = op_type;
    fixture.model.graph.nodes[0].inputs = {"input", "weight"};
    fixture.model.graph.nodes[0].attributes.clear();
    if (trans_b) {
        fixture.model.graph.nodes[0].attributes["transB"] = int64_t{1};
    }
    fixture.model.graph.values[1].tensor.dims = dims;
    size_t count = 1;
    for (const int64_t dim : dims) {
        count *= static_cast<size_t>(dim);
    }
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<float>(std::vector<float>(count, 1.0f), dims);
    fixture.weights["weight"] = weight;

    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    if (feather::StaticQuantizeModel(fixture.model, fixture.weights, Config(), &output, &output_weights, &report) != 0) {
        return -2;
    }
    const auto weight_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                           [](const auto& value) { return value.tensor.name == "weight"; });
    return weight_desc == output.graph.values.end() ? -3 : weight_desc->tensor.quantization.axis;
}

TEST(static_quantizer_test, QuantizesWeightsBiasAndAddsQdqBoundaries) {
    auto fixture = MakeFixture();
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, Config(), &output, &output_weights, &report), 0);
    ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"fc"}));
    EXPECT_TRUE(report.skipped_nodes.empty());
    EXPECT_EQ(output.graph.outputs, (std::vector<std::string>{"output"}));
    ASSERT_EQ(output.graph.nodes.size(), 3U);
    EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
    EXPECT_EQ(output.graph.nodes[1].op_type, "FC");
    EXPECT_EQ(output.graph.nodes[2].op_type, "DequantizeLinear");
    const auto weight_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                           [](const auto& value) { return value.tensor.name == "weight"; });
    ASSERT_NE(weight_desc, output.graph.values.end());
    EXPECT_EQ(weight_desc->tensor.data_type, feather::DataType::INT8);
    EXPECT_EQ(weight_desc->tensor.quantization.granularity, feather::QuantizationGranularity::kPerChannel);
    EXPECT_EQ(weight_desc->tensor.quantization.axis, 1);
    ASSERT_NE(output_weights["weight"], nullptr);
    EXPECT_EQ(output_weights["weight"]->data_type(), feather::DataType::INT8);
    ASSERT_NE(output_weights["bias"], nullptr);
    EXPECT_EQ(output_weights["bias"]->data_type(), feather::DataType::INT32);
}

TEST(static_quantizer_test, ReusesMatchingInt8BoundaryAcrossStandardNodes) {
    feather::model::ModelDesc model;
    model.name = "standard_chain";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};
    model.graph.values = {
        Value("input", {1, 4}, feather::DataType::FP32, false),
        Value("hidden", {1, 4}, feather::DataType::FP32, false),
        Value("output", {1, 4}, feather::DataType::FP32, false),
    };
    model.graph.nodes = {
        {"relu0", "ReLU", "", {"input"}, {"hidden"}, {}},
        {"relu1", "ReLU", "", {"hidden"}, {"output"}, {}},
    };

    feather::StaticQuantizationConfig config;
    config.activations["input"] = {0.25f, 0};
    config.activations["hidden"] = {0.25f, 0};
    config.activations["output"] = {0.25f, 0};
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;

    ASSERT_EQ(feather::StaticQuantizeModel(model, {}, config, &output, &output_weights, &report), 0);
    ASSERT_EQ(output.graph.nodes.size(), 4U);
    EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
    EXPECT_EQ(output.graph.nodes[1].op_type, "ReLU");
    EXPECT_EQ(output.graph.nodes[2].op_type, "ReLU");
    EXPECT_EQ(output.graph.nodes[3].op_type, "DequantizeLinear");
    EXPECT_EQ(output.graph.nodes[2].inputs[0], output.graph.nodes[1].outputs[0]);
}

TEST(static_quantizer_test, CanKeepSelectedNodesInFp32) {
    auto fixture = MakeFixture();
    auto config = Config();
    config.keep_fp32_node_prefixes = {"fc"};

    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, config, &output, &output_weights, &report), 0);
    EXPECT_TRUE(report.quantized_nodes.empty());
    EXPECT_EQ(report.skipped_nodes, (std::vector<std::string>{"fc"}));
    ASSERT_EQ(output.graph.nodes.size(), 1U);
    EXPECT_EQ(output.graph.nodes[0].op_type, "FC");
    EXPECT_EQ(output_weights.at("weight")->data_type(), feather::DataType::FP32);
    EXPECT_EQ(output_weights.at("bias")->data_type(), feather::DataType::FP32);
    const auto weight_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                           [](const auto& value) { return value.tensor.name == "weight"; });
    const auto bias_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                         [](const auto& value) { return value.tensor.name == "bias"; });
    ASSERT_NE(weight_desc, output.graph.values.end());
    ASSERT_NE(bias_desc, output.graph.values.end());
    EXPECT_EQ(weight_desc->tensor.data_type, feather::DataType::FP32);
    EXPECT_EQ(bias_desc->tensor.data_type, feather::DataType::FP32);
}

TEST(static_quantizer_test, StrictModeRejectsMissingActivationScale) {
    auto fixture = MakeFixture();
    auto config = Config();
    config.activations.erase("output");
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    EXPECT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, config, &output, &output_weights, &report), -1);
    EXPECT_FALSE(report.diagnostics.empty());
}

TEST(static_quantizer_test, NonStrictModeReportsSkippedNode) {
    auto fixture = MakeFixture();
    auto config = Config();
    config.activations.erase("output");
    config.strict = false;
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, config, &output, &output_weights, &report), 0);
    EXPECT_EQ(report.skipped_nodes, (std::vector<std::string>{"fc"}));
    EXPECT_EQ(output.graph.nodes.size(), 1U);
}

TEST(static_quantizer_test, NonStrictModeDoesNotPartiallyQuantizeSkippedNode) {
    auto fixture = MakeFixture();
    fixture.weights.erase("bias");
    auto config = Config();
    config.strict = false;
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, config, &output, &output_weights, &report), 0);
    ASSERT_EQ(output.graph.nodes.size(), 1U);
    const auto weight_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                           [](const auto& value) { return value.tensor.name == "weight"; });
    ASSERT_NE(weight_desc, output.graph.values.end());
    EXPECT_EQ(weight_desc->tensor.data_type, feather::DataType::FP32);
    ASSERT_NE(output_weights.find("weight"), output_weights.end());
    EXPECT_EQ(output_weights.at("weight")->data_type(), feather::DataType::FP32);
}

TEST(static_quantizer_test, UsesOperationAwareWeightQuantizationAxes) {
    EXPECT_EQ(QuantizedWeightAxis("FC", {2, 2}, false), 1);
    EXPECT_EQ(QuantizedWeightAxis("Gemm", {2, 2}, false), 1);
    EXPECT_EQ(QuantizedWeightAxis("Gemm", {2, 2}, true), 0);
    EXPECT_EQ(QuantizedWeightAxis("MatMul", {2, 2, 2}, false), 2);
    EXPECT_EQ(QuantizedWeightAxis("Conv2D", {2, 1, 1, 1}, false), 0);
}

TEST(static_quantizer_test, PerTensorWeightsQuantizeVectorBias) {
    auto fixture = MakeFixture();
    auto config = Config();
    config.per_channel_weights = false;
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;

    ASSERT_EQ(feather::StaticQuantizeModel(fixture.model, fixture.weights, config, &output, &output_weights, &report),
              0);
    ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"fc"}));
    ASSERT_NE(output_weights.find("bias"), output_weights.end());
    EXPECT_EQ(output_weights.at("bias")->data_type(), feather::DataType::INT32);
    EXPECT_EQ(output_weights.at("bias")->numel(), 2);
}

TEST(static_quantizer_test, ParsesCommentsAndRejectsMalformedRows) {
    const std::string valid_path = "/tmp/static_quantizer_scales.txt";
    {
        std::ofstream file(valid_path);
        file << "# comment\n\ninput 0.5 0\noutput 1.0 -2\n";
    }
    feather::ActivationQuantizationTable table;
    std::vector<std::string> diagnostics;
    ASSERT_EQ(feather::LoadActivationQuantizationTable(valid_path, &table, &diagnostics), 0);
    EXPECT_FLOAT_EQ(table.at("input").scale, 0.5f);
    EXPECT_EQ(table.at("output").zero_point, -2);
    const std::string invalid_path = "/tmp/static_quantizer_invalid_scales.txt";
    {
        std::ofstream file(invalid_path);
        file << "input 0 0\n";
    }
    EXPECT_EQ(feather::LoadActivationQuantizationTable(invalid_path, &table, &diagnostics), -1);
}

TEST(static_quantizer_test, QuantizesStandardBinaryNodeAndStoresAllQuantizationConstants) {
    feather::model::ModelDesc model;
    model.name = "standard_binary";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs"};
    model.graph.outputs = {"out"};
    model.graph.values = {
        Value("lhs", {1, 2}, feather::DataType::FP32, false),
        Value("rhs", {1, 2}, feather::DataType::FP32, false),
        Value("out", {1, 2}, feather::DataType::FP32, false),
    };
    model.graph.nodes = {{"add", "Add", "", {"lhs", "rhs"}, {"out"}, {}}};

    feather::StaticQuantizationConfig config;
    config.activations["lhs"] = {0.5f, 0};
    config.activations["rhs"] = {0.25f, -2};
    config.activations["out"] = {0.125f, 3};
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(model, {}, config, &output, &output_weights, &report), 0);
    ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"add"}));
    ASSERT_EQ(output.graph.nodes.size(), 4U);
    EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
    EXPECT_EQ(output.graph.nodes[1].op_type, "QuantizeLinear");
    EXPECT_EQ(output.graph.nodes[2].op_type, "Add");
    EXPECT_EQ(output.graph.nodes[3].op_type, "DequantizeLinear");
    EXPECT_NE(output_weights.find("lhs__int8_scale"), output_weights.end());
    EXPECT_NE(output_weights.find("lhs__int8_zero_point"), output_weights.end());
    EXPECT_NE(output_weights.find("rhs__int8_scale"), output_weights.end());
    EXPECT_NE(output_weights.find("rhs__int8_zero_point"), output_weights.end());
    EXPECT_NE(output_weights.find("out__int8_scale"), output_weights.end());
    EXPECT_NE(output_weights.find("out__int8_zero_point"), output_weights.end());
}

TEST(static_quantizer_test, QuantizesStandardSplitWithIndependentOutputScales) {
    feather::model::ModelDesc model;
    model.name = "standard_split";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"out0", "out1"};
    model.graph.values = {
        Value("input", {1, 4}, feather::DataType::FP32, false),
        Value("split", {2}, feather::DataType::INT64, true),
        Value("out0", {1, 2}, feather::DataType::FP32, false),
        Value("out1", {1, 2}, feather::DataType::FP32, false),
    };
    model.graph.nodes = {{"split_node", "Split", "", {"input", "split"}, {"out0", "out1"}, {}}};
    auto split = std::make_shared<feather::Tensor>();
    split->Assign<int64_t>({2, 2}, {2});
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights{{"split", split}};
    feather::StaticQuantizationConfig config;
    config.activations["input"] = {0.5f, 0};
    config.activations["out0"] = {0.25f, 0};
    config.activations["out1"] = {0.125f, 1};
    feather::model::ModelDesc output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    ASSERT_EQ(feather::StaticQuantizeModel(model, weights, config, &output, &output_weights, &report), 0);
    ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"split_node"}));
    ASSERT_EQ(output.graph.nodes.size(), 4U);
    EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
    EXPECT_EQ(output.graph.nodes[1].op_type, "Split");
    EXPECT_EQ(output.graph.nodes[2].op_type, "DequantizeLinear");
    EXPECT_EQ(output.graph.nodes[3].op_type, "DequantizeLinear");
    EXPECT_NE(output_weights.find("out0__int8_scale"), output_weights.end());
    EXPECT_NE(output_weights.find("out1__int8_scale"), output_weights.end());
    EXPECT_EQ(output_weights.at("split")->data_type(), feather::DataType::INT64);
}

TEST(static_quantizer_test, CoversEveryStandardDataOperator) {
    const std::vector<std::string> standard_ops = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "ReLU",
        "Relu",
        "Neg",
        "Sigmoid",
        "SiLU",
        "Silu",
        "Exp",
        "Sqrt",
        "Tanh",
        "Erf",
        "Sin",
        "Cos",
        "Softplus",
        "Pow",
        "BatchNormalization",
        "AvgPool",
        "MaxPool",
        "GlobalAveragePool",
        "ReduceSum",
        "ReduceMean",
        "Identity",
        "Reshape",
        "Flatten",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Slice",
        "Split",
        "Concat",
        "Expand",
        "Gather",
        "Where",
        "Resize",
        "Softmax",
        "Equal",
        "Cast",
        "ConstantOfShape",
        "Shape",
        "ResizeConcat",
        "YoloDecode",
    };
    ASSERT_FALSE(standard_ops.empty());
    for (const auto& op_type : standard_ops) {
        feather::model::ModelDesc model;
        model.name = "standard_" + op_type;
        model.version = 1;
        model.graph.name = "main";
        model.graph.outputs = {"output"};
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
        std::vector<std::string> numeric_inputs;
        std::vector<std::string> control_inputs;
        std::vector<std::string> node_inputs;
        auto add_numeric_input = [&](const std::string& name) {
            model.graph.inputs.push_back(name);
            model.graph.values.push_back(Value(name, {1, 2}, feather::DataType::FP32, false));
            numeric_inputs.push_back(name);
            node_inputs.push_back(name);
        };
        auto add_condition_input = [&](const std::string& name) {
            model.graph.inputs.push_back(name);
            model.graph.values.push_back(Value(name, {1, 2}, feather::DataType::BOOL, false));
            control_inputs.push_back(name);
            node_inputs.push_back(name);
        };
        auto add_control_input = [&](const std::string& name, feather::DataType data_type) {
            model.graph.values.push_back(Value(name, {1}, data_type, true));
            auto tensor = std::make_shared<feather::Tensor>();
            if (data_type == feather::DataType::INT64) {
                tensor->Assign<int64_t>({1}, {1});
            } else {
                tensor->Assign<float>({1.0f}, {1});
            }
            weights[name] = tensor;
            control_inputs.push_back(name);
            node_inputs.push_back(name);
        };

        if (op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div" ||
            op_type == "Equal") {
            add_numeric_input("lhs");
            add_numeric_input("rhs");
        } else if (op_type == "Concat") {
            add_numeric_input("lhs");
            add_numeric_input("rhs");
        } else if (op_type == "Where") {
            add_condition_input("condition");
            add_numeric_input("lhs");
            add_numeric_input("rhs");
        } else if (op_type == "BatchNormalization") {
            add_numeric_input("input");
            add_control_input("bn_scale", feather::DataType::FP32);
            add_control_input("bn_bias", feather::DataType::FP32);
            add_control_input("bn_mean", feather::DataType::FP32);
            add_control_input("bn_var", feather::DataType::FP32);
        } else if (op_type == "Pow") {
            add_numeric_input("input");
            add_control_input("exponent", feather::DataType::FP32);
        } else if (op_type == "Reshape" || op_type == "Expand") {
            add_numeric_input("input");
            add_control_input("shape", feather::DataType::INT64);
        } else if (op_type == "Gather") {
            add_numeric_input("input");
            add_control_input("indices", feather::DataType::INT64);
        } else if (op_type == "Slice") {
            add_numeric_input("input");
            add_control_input("starts", feather::DataType::INT64);
            add_control_input("ends", feather::DataType::INT64);
            add_control_input("axes", feather::DataType::INT64);
            add_control_input("steps", feather::DataType::INT64);
        } else if (op_type == "Split") {
            add_numeric_input("input");
            add_control_input("split", feather::DataType::INT64);
        } else if (op_type == "Resize") {
            add_numeric_input("input");
            add_control_input("roi", feather::DataType::FP32);
            add_control_input("scales", feather::DataType::FP32);
            add_control_input("sizes", feather::DataType::INT64);
        } else if (op_type == "Squeeze" || op_type == "Unsqueeze") {
            add_numeric_input("input");
            add_control_input("axes", feather::DataType::INT64);
        } else if (op_type == "ConstantOfShape") {
            add_control_input("shape", feather::DataType::INT64);
        } else {
            add_numeric_input("input");
        }

        const auto output_type = op_type == "Equal" ? feather::DataType::BOOL : feather::DataType::FP32;
        model.graph.values.push_back(Value("output", {1, 2}, output_type, false));
        feather::model::NodeDesc node;
        node.name = "node_" + op_type;
        node.op_type = op_type;
        node.inputs = node_inputs;
        node.outputs = {"output"};
        if (op_type == "Flatten") node.attributes["axis"] = int64_t{1};
        if (op_type == "Softmax") node.attributes["axis"] = int64_t{-1};
        if (op_type == "Transpose") node.attributes["perm"] = std::vector<int64_t>{1, 0};
        if (op_type == "Cast") node.attributes["to"] = int64_t{1};
        model.graph.nodes = {node};

        feather::StaticQuantizationConfig config;
        for (const auto& input : numeric_inputs) config.activations[input] = {0.25f, 0};
        config.activations["output"] = {0.25f, 0};
        feather::model::ModelDesc output;
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
        feather::StaticQuantizationReport report;
        ASSERT_EQ(feather::StaticQuantizeModel(model, weights, config, &output, &output_weights, &report), 0) << op_type;
        EXPECT_TRUE(report.skipped_nodes.empty()) << op_type;
        ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{node.name})) << op_type;

        bool has_quantize = false;
        bool has_dequantize = false;
        for (const auto& converted : output.graph.nodes) {
            has_quantize = has_quantize || converted.op_type == "QuantizeLinear";
            has_dequantize = has_dequantize || converted.op_type == "DequantizeLinear";
        }
        const bool preserves_non_numeric_semantics = op_type == "Equal" || op_type == "Cast" ||
                                                     op_type == "ConstantOfShape";
        if (op_type == "Shape") {
            EXPECT_TRUE(has_quantize) << op_type;
            EXPECT_FALSE(has_dequantize) << op_type;
            ASSERT_GE(output.graph.nodes.size(), 2U) << op_type;
            EXPECT_EQ(output.graph.nodes.back().op_type, "Shape");
        } else if (!preserves_non_numeric_semantics) {
            EXPECT_EQ(has_quantize, !numeric_inputs.empty()) << op_type;
            EXPECT_TRUE(has_dequantize) << op_type;
        } else if (op_type == "Equal") {
            EXPECT_FALSE(has_dequantize) << op_type;
            EXPECT_TRUE(has_quantize) << op_type;
            ASSERT_GE(output.graph.nodes.size(), 2U) << op_type;
            EXPECT_EQ(output.graph.nodes.back().op_type, "Equal");
        } else if (op_type == "Cast") {
            EXPECT_FALSE(has_dequantize) << op_type;
            EXPECT_TRUE(has_quantize) << op_type;
            ASSERT_GE(output.graph.nodes.size(), 2U) << op_type;
            EXPECT_EQ(output.graph.nodes.back().op_type, "Cast");
        } else {
            EXPECT_FALSE(has_quantize) << op_type;
            EXPECT_TRUE(has_dequantize) << op_type;
            ASSERT_GE(output.graph.nodes.size(), 2U) << op_type;
            EXPECT_EQ(output.graph.nodes.front().op_type, "ConstantOfShape");
        }
        for (const auto& converted : output.graph.nodes) {
            if (converted.op_type != "QuantizeLinear" || converted.inputs.empty()) continue;
            EXPECT_EQ(std::find(control_inputs.begin(), control_inputs.end(), converted.inputs.front()),
                      control_inputs.end()) << op_type;
        }
    }
}

TEST(static_quantizer_test, QuantizesEveryLinearOperatorWithInt8WeightsAndInt32Bias) {
    struct LinearCase {
        const char* op_type;
        std::vector<int64_t> weight_dims;
        bool has_bias;
    };
    const std::vector<LinearCase> cases = {
        {"FC", {2, 2}, true},
        {"Gemm", {2, 2}, true},
        {"MatMul", {2, 2}, false},
        {"Conv2D", {1, 1, 1, 1}, true},
    };
    for (const auto& linear : cases) {
        feather::model::ModelDesc model;
        model.name = std::string("linear_") + linear.op_type;
        model.version = 1;
        model.graph.name = "main";
        model.graph.inputs = {"input"};
        model.graph.outputs = {"output"};
        model.graph.values = {
            Value("input", linear.op_type == std::string("Conv2D") ? std::vector<int64_t>{1, 1, 2, 2}
                                                                       : std::vector<int64_t>{1, 2},
                  feather::DataType::FP32, false),
            Value("weight", linear.weight_dims, feather::DataType::FP32, true),
            Value("output", linear.op_type == std::string("Conv2D") ? std::vector<int64_t>{1, 1, 2, 2}
                                                                        : std::vector<int64_t>{1, 2},
                  feather::DataType::FP32, false),
        };
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
        auto weight = std::make_shared<feather::Tensor>();
        weight->Assign<float>(std::vector<float>(linear.op_type == std::string("Conv2D") ? 1U : 4U, 0.5f),
                              linear.weight_dims);
        weights["weight"] = weight;
        feather::model::NodeDesc node;
        node.name = std::string("node_") + linear.op_type;
        node.op_type = linear.op_type;
        node.inputs = {"input", "weight"};
        if (linear.has_bias) {
            const int64_t bias_channels = linear.op_type == std::string("Conv2D") ? 1 : 2;
            model.graph.values.push_back(Value("bias", {bias_channels}, feather::DataType::FP32, true));
            auto bias = std::make_shared<feather::Tensor>();
            if (bias_channels == 1) {
                bias->Assign<float>({0.25f}, {1});
            } else {
                bias->Assign<float>({0.25f, -0.5f}, {2});
            }
            weights["bias"] = bias;
            node.inputs.push_back("bias");
        }
        node.outputs = {"output"};
        if (linear.op_type == std::string("Conv2D")) {
            node.attributes["stride_h"] = int64_t{1};
            node.attributes["stride_w"] = int64_t{1};
            node.attributes["pad_h"] = int64_t{0};
            node.attributes["pad_w"] = int64_t{0};
            node.attributes["dilation_h"] = int64_t{1};
            node.attributes["dilation_w"] = int64_t{1};
            node.attributes["group"] = int64_t{1};
        }
        model.graph.nodes = {node};

        feather::StaticQuantizationConfig config;
        config.activations["input"] = {0.25f, 0};
        config.activations["output"] = {0.5f, 0};
        feather::model::ModelDesc output;
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
        feather::StaticQuantizationReport report;
        ASSERT_EQ(feather::StaticQuantizeModel(model, weights, config, &output, &output_weights, &report), 0)
            << linear.op_type;
        ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{node.name})) << linear.op_type;
        EXPECT_TRUE(report.skipped_nodes.empty()) << linear.op_type;
        ASSERT_EQ(output.graph.nodes.size(), 3U) << linear.op_type;
        EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear") << linear.op_type;
        EXPECT_EQ(output.graph.nodes[1].op_type, linear.op_type) << linear.op_type;
        EXPECT_EQ(output.graph.nodes[2].op_type, "DequantizeLinear") << linear.op_type;
        ASSERT_NE(output_weights.find("weight"), output_weights.end()) << linear.op_type;
        EXPECT_EQ(output_weights.at("weight")->data_type(), feather::DataType::INT8) << linear.op_type;
        if (linear.has_bias) {
            ASSERT_NE(output_weights.find("bias"), output_weights.end()) << linear.op_type;
            EXPECT_EQ(output_weights.at("bias")->data_type(), feather::DataType::INT32) << linear.op_type;
        }
        const auto weight_desc = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                               [](const auto& value) { return value.tensor.name == "weight"; });
        ASSERT_NE(weight_desc, output.graph.values.end()) << linear.op_type;
        EXPECT_EQ(weight_desc->tensor.data_type, feather::DataType::INT8) << linear.op_type;
    }
}

TEST(static_quantizer_test, QuantizesOnnxOperatorAliases) {
    auto conv_fixture = MakeFixture();
    conv_fixture.model.graph.nodes[0].op_type = "Conv";
    conv_fixture.model.graph.values[0].tensor.dims = {1, 1, 1, 1};
    conv_fixture.model.graph.values[1].tensor.dims = {2, 1, 1, 1};
    conv_fixture.model.graph.values[3].tensor.dims = {1, 2, 1, 1};
    auto conv_weight = std::make_shared<feather::Tensor>();
    conv_weight->Assign<float>({0.5f, -1.0f}, {2, 1, 1, 1});
    conv_fixture.weights["weight"] = conv_weight;

    feather::model::ModelDesc conv_output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> conv_weights;
    feather::StaticQuantizationReport conv_report;
    ASSERT_EQ(feather::StaticQuantizeModel(conv_fixture.model, conv_fixture.weights, Config(), &conv_output,
                                           &conv_weights, &conv_report), 0);
    EXPECT_EQ(conv_report.quantized_nodes, (std::vector<std::string>{"fc"}));
    EXPECT_TRUE(conv_report.skipped_nodes.empty());
    ASSERT_EQ(conv_output.graph.nodes.size(), 3U);
    EXPECT_EQ(conv_output.graph.nodes[1].op_type, "Conv");
    ASSERT_NE(conv_weights.find("weight"), conv_weights.end());
    EXPECT_EQ(conv_weights.at("weight")->data_type(), feather::DataType::INT8);
    ASSERT_NE(conv_weights.find("bias"), conv_weights.end());
    EXPECT_EQ(conv_weights.at("bias")->data_type(), feather::DataType::INT32);

    feather::model::ModelDesc pool_model;
    pool_model.name = "average_pool_alias";
    pool_model.version = 1;
    pool_model.graph.name = "main";
    pool_model.graph.inputs = {"input"};
    pool_model.graph.outputs = {"output"};
    pool_model.graph.values = {
        Value("input", {1, 1, 2, 2}, feather::DataType::FP32, false),
        Value("output", {1, 1, 1, 1}, feather::DataType::FP32, false),
    };
    feather::model::NodeDesc pool;
    pool.name = "average_pool";
    pool.op_type = "AveragePool";
    pool.inputs = {"input"};
    pool.outputs = {"output"};
    pool.attributes["kernel_h"] = int64_t{2};
    pool.attributes["kernel_w"] = int64_t{2};
    pool.attributes["stride_h"] = int64_t{2};
    pool.attributes["stride_w"] = int64_t{2};
    pool_model.graph.nodes = {pool};

    auto pool_config = Config();
    feather::model::ModelDesc pool_output;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> pool_weights;
    feather::StaticQuantizationReport pool_report;
    ASSERT_EQ(feather::StaticQuantizeModel(pool_model, {}, pool_config, &pool_output, &pool_weights, &pool_report), 0);
    EXPECT_EQ(pool_report.quantized_nodes, (std::vector<std::string>{"average_pool"}));
    EXPECT_TRUE(pool_report.skipped_nodes.empty());
    ASSERT_EQ(pool_output.graph.nodes.size(), 3U);
    EXPECT_EQ(pool_output.graph.nodes[1].op_type, "AveragePool");
}



TEST(static_quantizer_test, QuantizesShapeResizeConcatAndYoloDecode) {
    {
        feather::model::ModelDesc model;
        model.name = "shape_quant";
        model.version = 1;
        model.graph.name = "main";
        model.graph.inputs = {"input"};
        model.graph.outputs = {"output"};
        model.graph.values = {
            Value("input", {1, 2, 3}, feather::DataType::FP32, false),
            Value("output", {3}, feather::DataType::INT64, false),
        };
        model.graph.nodes = {{"shape", "Shape", "", {"input"}, {"output"}, {}}};
        feather::StaticQuantizationConfig config;
        config.activations["input"] = {0.5f, 0};
        feather::model::ModelDesc output;
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
        feather::StaticQuantizationReport report;
        ASSERT_EQ(feather::StaticQuantizeModel(model, {}, config, &output, &weights, &report), 0);
        ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"shape"}));
        ASSERT_EQ(output.graph.nodes.size(), 2U);
        EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
        EXPECT_EQ(output.graph.nodes[1].op_type, "Shape");
        const auto value = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                        [](const auto& item) { return item.tensor.name == "output"; });
        ASSERT_NE(value, output.graph.values.end());
        EXPECT_EQ(value->tensor.data_type, feather::DataType::INT64);
    }

    {
        feather::model::ModelDesc model;
        model.name = "resize_concat_quant";
        model.version = 1;
        model.graph.name = "main";
        model.graph.inputs = {"resize_input", "concat_input"};
        model.graph.outputs = {"output"};
        model.graph.values = {
            Value("resize_input", {1, 1, 1, 1}, feather::DataType::FP32, false),
            Value("concat_input", {1, 1, 2, 2}, feather::DataType::FP32, false),
            Value("output", {1, 2, 2, 2}, feather::DataType::FP32, false),
        };
        feather::model::NodeDesc node;
        node.name = "resize_concat";
        node.op_type = "ResizeConcat";
        node.inputs = {"resize_input", "concat_input"};
        node.outputs = {"output"};
        node.attributes["scales"] = std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f};
        node.attributes["axis"] = int64_t{1};
        node.attributes["resize_input_index"] = int64_t{0};
        model.graph.nodes = {node};
        feather::StaticQuantizationConfig config;
        config.activations["resize_input"] = {0.5f, 0};
        config.activations["concat_input"] = {0.25f, 0};
        config.activations["output"] = {0.5f, 0};
        feather::model::ModelDesc output;
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
        feather::StaticQuantizationReport report;
        ASSERT_EQ(feather::StaticQuantizeModel(model, {}, config, &output, &weights, &report), 0);
        ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"resize_concat"}));
        ASSERT_EQ(output.graph.nodes.size(), 4U);
        EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
        EXPECT_EQ(output.graph.nodes[1].op_type, "QuantizeLinear");
        EXPECT_EQ(output.graph.nodes[2].op_type, "ResizeConcat");
        EXPECT_EQ(output.graph.nodes[3].op_type, "DequantizeLinear");
    }

    {
        feather::model::ModelDesc model;
        model.name = "yolo_decode_quant";
        model.version = 1;
        model.graph.name = "main";
        model.graph.inputs = {"input"};
        model.graph.outputs = {"output"};
        model.graph.values = {
            Value("input", {1, 12, 1, 1}, feather::DataType::FP32, false),
            Value("xy_scale", {2}, feather::DataType::FP32, true),
            Value("grid", {2}, feather::DataType::FP32, true),
            Value("stride", {1}, feather::DataType::FP32, true),
            Value("wh_scale", {2}, feather::DataType::FP32, true),
            Value("anchor_grid", {2}, feather::DataType::FP32, true),
            Value("output", {1, 2, 6}, feather::DataType::FP32, false),
        };
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> source_weights;
        auto add_constant = [&](const std::string& name, const std::vector<float>& data,
                                const std::vector<int64_t>& dims) {
            auto tensor = std::make_shared<feather::Tensor>();
            tensor->Assign<float>(data, dims);
            source_weights[name] = tensor;
        };
        add_constant("xy_scale", {2.0f, 2.0f}, {2});
        add_constant("grid", {1.0f, 1.0f}, {2});
        add_constant("stride", {4.0f}, {1});
        add_constant("wh_scale", {2.0f, 2.0f}, {2});
        add_constant("anchor_grid", {4.0f, 4.0f}, {2});
        model.graph.nodes = {{"yolo", "YoloDecode", "", {"input", "xy_scale", "grid", "stride", "wh_scale", "anchor_grid"}, {"output"}, {}}};
        feather::StaticQuantizationConfig config;
        config.activations["input"] = {0.5f, 0};
        config.activations["output"] = {0.5f, 0};
        feather::model::ModelDesc output;
        std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
        feather::StaticQuantizationReport report;
        ASSERT_EQ(feather::StaticQuantizeModel(model, source_weights, config, &output, &weights, &report), 0);
        ASSERT_EQ(report.quantized_nodes, (std::vector<std::string>{"yolo"}));
        ASSERT_EQ(output.graph.nodes.size(), 3U);
        EXPECT_EQ(output.graph.nodes[0].op_type, "QuantizeLinear");
        EXPECT_EQ(output.graph.nodes[1].op_type, "YoloDecode");
        EXPECT_EQ(output.graph.nodes[2].op_type, "DequantizeLinear");
        for (const auto& name : {"xy_scale", "grid", "stride", "wh_scale", "anchor_grid"}) {
            const auto value = std::find_if(output.graph.values.begin(), output.graph.values.end(),
                                            [&](const auto& item) { return item.tensor.name == name; });
            ASSERT_NE(value, output.graph.values.end());
            EXPECT_EQ(value->tensor.data_type, feather::DataType::FP32);
        }
    }
}

}  // namespace
