#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/graph_lowering.h"
#include "core/graph.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_format.h"

using feather::DataType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

TEST(runtime_graph_pipeline_test, BuildAndRunConvReluFcGraph) {
    ModelDesc model;
    model.name = "conv_relu_fc_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc conv_weight;
    conv_weight.tensor.name = "conv_w";
    conv_weight.tensor.dims = {2, 2};
    conv_weight.tensor.data_type = DataType::FP32;
    conv_weight.constant = true;

    ValueDesc conv_bias;
    conv_bias.tensor.name = "conv_bias";
    conv_bias.tensor.dims = {2, 2};
    conv_bias.tensor.data_type = DataType::FP32;
    conv_bias.constant = true;

    ValueDesc conv_out;
    conv_out.tensor.name = "conv_out";
    conv_out.tensor.dims = {2, 2};
    conv_out.tensor.data_type = DataType::FP32;

    ValueDesc relu_out;
    relu_out.tensor.name = "relu_out";
    relu_out.tensor.dims = {2, 2};
    relu_out.tensor.data_type = DataType::FP32;

    ValueDesc fc_weight;
    fc_weight.tensor.name = "fc_w";
    fc_weight.tensor.dims = {2, 3};
    fc_weight.tensor.data_type = DataType::FP32;
    fc_weight.constant = true;

    ValueDesc fc_bias;
    fc_bias.tensor.name = "fc_bias";
    fc_bias.tensor.dims = {2, 3};
    fc_bias.tensor.data_type = DataType::FP32;
    fc_bias.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 3};
    output.tensor.data_type = DataType::FP32;

    NodeDesc conv;
    conv.name = "conv0";
    conv.op_type = "Conv2D";
    conv.inputs = {"input", "conv_w", "conv_bias"};
    conv.outputs = {"conv_out"};
    conv.attributes["stride_h"] = static_cast<int64_t>(1);
    conv.attributes["stride_w"] = static_cast<int64_t>(1);
    conv.attributes["pad_h"] = static_cast<int64_t>(0);
    conv.attributes["pad_w"] = static_cast<int64_t>(0);

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"conv_out"};
    relu.outputs = {"relu_out"};

    NodeDesc fc;
    fc.name = "fc0";
    fc.op_type = "FC";
    fc.inputs = {"relu_out", "fc_w", "fc_bias"};
    fc.outputs = {"output"};

    model.graph.values = {input, conv_weight, conv_bias, conv_out, relu_out, fc_weight, fc_bias, output};
    model.graph.nodes = {conv, relu, fc};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto conv_weight_tensor = std::make_shared<Tensor>();
    conv_weight_tensor->Assign<float>({
        1, 0,
        0, -1,
    }, {2, 2});

    auto conv_bias_tensor = std::make_shared<Tensor>();
    conv_bias_tensor->Assign<float>({
        1, 1,
        1, 1,
    }, {2, 2});

    auto fc_weight_tensor = std::make_shared<Tensor>();
    fc_weight_tensor->Assign<float>({
        1, 2, 3,
        4, 5, 6,
    }, {2, 3});

    auto fc_bias_tensor = std::make_shared<Tensor>();
    fc_bias_tensor->Assign<float>({
        1, 1, 1,
        2, 2, 2,
    }, {2, 3});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("conv_w", conv_weight_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("conv_bias", conv_bias_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("fc_w", fc_weight_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("fc_bias", fc_bias_tensor), 0);

    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.OperatorSize(), 3U);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 3U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 3}));

    std::vector<float> expected = {
        1, 1, 1,
        2, 2, 2,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunGemmSigmoidGraph) {
    ModelDesc model;
    model.name = "gemm_sigmoid_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {3, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc bias;
    bias.tensor.name = "bias";
    bias.tensor.dims = {2, 2};
    bias.tensor.data_type = DataType::FP32;
    bias.constant = true;

    ValueDesc gemm_out;
    gemm_out.tensor.name = "gemm_out";
    gemm_out.tensor.dims = {2, 2};
    gemm_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc gemm;
    gemm.name = "gemm0";
    gemm.op_type = "Gemm";
    gemm.inputs = {"input", "weight", "bias"};
    gemm.outputs = {"gemm_out"};

    NodeDesc sigmoid;
    sigmoid.name = "sigmoid0";
    sigmoid.op_type = "Sigmoid";
    sigmoid.inputs = {"gemm_out"};
    sigmoid.outputs = {"output"};

    model.graph.values = {input, weight, bias, gemm_out, output};
    model.graph.nodes = {gemm, sigmoid};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->Assign<float>({1, 1, 2, 2}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("bias", bias_tensor), 0);

    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.OperatorSize(), 2U);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.NodeSize(), 2U);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 2}));

    EXPECT_GT(output_tensor->data<float>()[0], 0.99f);
    EXPECT_GT(output_tensor->data<float>()[1], 0.99f);
    EXPECT_GT(output_tensor->data<float>()[2], 0.99f);
    EXPECT_GT(output_tensor->data<float>()[3], 0.99f);
}

TEST(runtime_graph_pipeline_test, BuildAndRunReshapeGraph) {
    ModelDesc model;
    model.name = "reshape_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {3, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc reshape;
    reshape.name = "reshape0";
    reshape.op_type = "Reshape";
    reshape.inputs = {"input"};
    reshape.outputs = {"output"};
    reshape.attributes["shape"] = std::vector<int64_t>{3, 2};

    model.graph.values = {input, output};
    model.graph.nodes = {reshape};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 2}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], input_tensor->data<float>()[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunAvgPoolReluGraph) {
    ModelDesc model;
    model.name = "avgpool_relu_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc pool_out;
    pool_out.tensor.name = "pool_out";
    pool_out.tensor.dims = {2, 2};
    pool_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc avgpool;
    avgpool.name = "avgpool0";
    avgpool.op_type = "AvgPool";
    avgpool.inputs = {"input"};
    avgpool.outputs = {"pool_out"};
    avgpool.attributes["kernel_h"] = static_cast<int64_t>(2);
    avgpool.attributes["kernel_w"] = static_cast<int64_t>(2);
    avgpool.attributes["stride_h"] = static_cast<int64_t>(1);
    avgpool.attributes["stride_w"] = static_cast<int64_t>(1);
    avgpool.attributes["pad_h"] = static_cast<int64_t>(0);
    avgpool.attributes["pad_w"] = static_cast<int64_t>(0);

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"pool_out"};
    relu.outputs = {"output"};

    model.graph.values = {input, pool_out, output};
    model.graph.nodes = {avgpool, relu};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 2}));
    const std::vector<float> expected = {3.0f, 4.0f, 6.0f, 7.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunConcatSplitGraph) {
    ModelDesc model;
    model.name = "concat_split_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs"};
    model.graph.outputs = {"out0", "out1"};

    ValueDesc lhs;
    lhs.tensor.name = "lhs";
    lhs.tensor.dims = {2, 2};
    lhs.tensor.data_type = DataType::FP32;

    ValueDesc rhs;
    rhs.tensor.name = "rhs";
    rhs.tensor.dims = {2, 2};
    rhs.tensor.data_type = DataType::FP32;

    ValueDesc concat_out;
    concat_out.tensor.name = "concat_out";
    concat_out.tensor.dims = {2, 4};
    concat_out.tensor.data_type = DataType::FP32;

    ValueDesc out0;
    out0.tensor.name = "out0";
    out0.tensor.dims = {2, 2};
    out0.tensor.data_type = DataType::FP32;

    ValueDesc out1;
    out1.tensor.name = "out1";
    out1.tensor.dims = {2, 2};
    out1.tensor.data_type = DataType::FP32;

    NodeDesc concat;
    concat.name = "concat0";
    concat.op_type = "Concat";
    concat.inputs = {"lhs", "rhs"};
    concat.outputs = {"concat_out"};
    concat.attributes["axis"] = static_cast<int64_t>(1);

    NodeDesc split;
    split.name = "split0";
    split.op_type = "Split";
    split.inputs = {"concat_out"};
    split.outputs = {"out0", "out1"};
    split.attributes["axis"] = static_cast<int64_t>(1);
    split.attributes["split_sizes"] = std::vector<int64_t>{2, 2};

    model.graph.values = {lhs, rhs, concat_out, out0, out1};
    model.graph.nodes = {concat, split};

    auto lhs_tensor = std::make_shared<Tensor>();
    lhs_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto rhs_tensor = std::make_shared<Tensor>();
    rhs_tensor->Assign<float>({5, 6, 7, 8}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("lhs", lhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("rhs", rhs_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto out0_tensor = runtime_graph.GetTensor("out0");
    auto out1_tensor = runtime_graph.GetTensor("out1");
    ASSERT_NE(out0_tensor, nullptr);
    ASSERT_NE(out1_tensor, nullptr);
    EXPECT_EQ(out0_tensor->dims().data(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(out1_tensor->dims().data(), std::vector<int64_t>({2, 2}));

    const std::vector<float> expected0 = {1, 2, 3, 4};
    const std::vector<float> expected1 = {5, 6, 7, 8};
    for (size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_FLOAT_EQ(out0_tensor->data<float>()[i], expected0[i]);
        EXPECT_FLOAT_EQ(out1_tensor->data<float>()[i], expected1[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunAddMulGraph) {
    ModelDesc model;
    model.name = "add_mul_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs", "scale"};
    model.graph.outputs = {"output"};

    ValueDesc lhs;
    lhs.tensor.name = "lhs";
    lhs.tensor.dims = {2, 2};
    lhs.tensor.data_type = DataType::FP32;

    ValueDesc rhs;
    rhs.tensor.name = "rhs";
    rhs.tensor.dims = {2, 2};
    rhs.tensor.data_type = DataType::FP32;

    ValueDesc scale;
    scale.tensor.name = "scale";
    scale.tensor.dims = {2, 2};
    scale.tensor.data_type = DataType::FP32;

    ValueDesc add_out;
    add_out.tensor.name = "add_out";
    add_out.tensor.dims = {2, 2};
    add_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc add;
    add.name = "add0";
    add.op_type = "Add";
    add.inputs = {"lhs", "rhs"};
    add.outputs = {"add_out"};

    NodeDesc mul;
    mul.name = "mul0";
    mul.op_type = "Mul";
    mul.inputs = {"add_out", "scale"};
    mul.outputs = {"output"};

    model.graph.values = {lhs, rhs, scale, add_out, output};
    model.graph.nodes = {add, mul};

    auto lhs_tensor = std::make_shared<Tensor>();
    lhs_tensor->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto rhs_tensor = std::make_shared<Tensor>();
    rhs_tensor->Assign<float>({10, 20, 30, 40}, {2, 2});

    auto scale_tensor = std::make_shared<Tensor>();
    scale_tensor->Assign<float>({2, 2, 3, 3}, {2, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("lhs", lhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("rhs", rhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("scale", scale_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    const std::vector<float> expected = {22, 44, 99, 132};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunTransposeSoftmaxGraph) {
    ModelDesc model;
    model.name = "transpose_softmax_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc transpose_out;
    transpose_out.tensor.name = "transpose_out";
    transpose_out.tensor.dims = {3, 2};
    transpose_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {3, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc transpose;
    transpose.name = "transpose0";
    transpose.op_type = "Transpose";
    transpose.inputs = {"input"};
    transpose.outputs = {"transpose_out"};
    transpose.attributes["perm"] = std::vector<int64_t>{1, 0};

    NodeDesc softmax;
    softmax.name = "softmax0";
    softmax.op_type = "Softmax";
    softmax.inputs = {"transpose_out"};
    softmax.outputs = {"output"};
    softmax.attributes["axis"] = static_cast<int64_t>(1);

    model.graph.values = {input, transpose_out, output};
    model.graph.nodes = {transpose, softmax};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 2}));

    for (int row = 0; row < 3; ++row) {
        const float row_sum = output_tensor->data<float>()[row * 2] + output_tensor->data<float>()[row * 2 + 1];
        EXPECT_NEAR(row_sum, 1.0f, 1e-6f);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunMatMulIdentityGraph) {
    ModelDesc model;
    model.name = "matmul_identity_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"lhs", "rhs"};
    model.graph.outputs = {"output"};

    ValueDesc lhs;
    lhs.tensor.name = "lhs";
    lhs.tensor.dims = {2, 3};
    lhs.tensor.data_type = DataType::FP32;

    ValueDesc rhs;
    rhs.tensor.name = "rhs";
    rhs.tensor.dims = {3, 2};
    rhs.tensor.data_type = DataType::FP32;

    ValueDesc matmul_out;
    matmul_out.tensor.name = "matmul_out";
    matmul_out.tensor.dims = {2, 2};
    matmul_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc matmul;
    matmul.name = "matmul0";
    matmul.op_type = "MatMul";
    matmul.inputs = {"lhs", "rhs"};
    matmul.outputs = {"matmul_out"};

    NodeDesc identity;
    identity.name = "identity0";
    identity.op_type = "Identity";
    identity.inputs = {"matmul_out"};
    identity.outputs = {"output"};

    model.graph.values = {lhs, rhs, matmul_out, output};
    model.graph.nodes = {matmul, identity};

    auto lhs_tensor = std::make_shared<Tensor>();
    lhs_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto rhs_tensor = std::make_shared<Tensor>();
    rhs_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("lhs", lhs_tensor), 0);
    ASSERT_EQ(static_graph.SetTensor("rhs", rhs_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    const std::vector<float> expected = {22, 28, 49, 64};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(runtime_graph_pipeline_test, BuildAndRunFlattenSliceGraph) {
    ModelDesc model;
    model.name = "flatten_slice_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc flatten_out;
    flatten_out.tensor.name = "flatten_out";
    flatten_out.tensor.dims = {2, 3};
    flatten_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc flatten;
    flatten.name = "flatten0";
    flatten.op_type = "Flatten";
    flatten.inputs = {"input"};
    flatten.outputs = {"flatten_out"};
    flatten.attributes["axis"] = static_cast<int64_t>(1);

    NodeDesc slice;
    slice.name = "slice0";
    slice.op_type = "Slice";
    slice.inputs = {"flatten_out"};
    slice.outputs = {"output"};
    slice.attributes["axis"] = static_cast<int64_t>(1);
    slice.attributes["start"] = static_cast<int64_t>(1);
    slice.attributes["end"] = static_cast<int64_t>(3);

    model.graph.values = {input, flatten_out, output};
    model.graph.nodes = {flatten, slice};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(model), 0);
    ASSERT_EQ(static_graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(static_graph.Build(), 0);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("output");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({2, 2}));
    const std::vector<float> expected = {2, 3, 5, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor->data<float>()[i], expected[i]);
    }
}

TEST(static_graph_pipeline_test, RejectsInvalidConv2DAttributes) {
    ModelDesc model;
    model.name = "bad_conv";
    model.version = 1;
    model.graph.name = "main";

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc node;
    node.name = "conv0";
    node.op_type = "Conv2D";
    node.inputs = {"input", "weight"};
    node.outputs = {"output"};
    node.attributes["stride_h"] = static_cast<int64_t>(0);
    node.attributes["stride_w"] = static_cast<int64_t>(1);

    model.graph.values = {input, weight, output};
    model.graph.nodes = {node};

    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    }, {3, 3});

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->Assign<float>({
        1, 0,
        0, -1,
    }, {2, 2});

    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph.SetTensor("weight", weight_tensor), 0);
    EXPECT_EQ(graph.Build(), -1);
}
