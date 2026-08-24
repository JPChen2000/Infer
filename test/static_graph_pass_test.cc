#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "pass/graph_pass.h"
#include "pass/dead_node_elimination_pass.h"
#include "pass/identity_elimination_pass.h"
#include "pass/matmul_add_fusion_pass.h"
#include "pass/qwen_depthwise_conv_fusion_pass.h"
#include "pass/qwen_gated_delta_fusion_pass.h"
#include "pass/qwen_matmul_add_fusion_pass.h"
#include "pass/qwen_state_output_alias_pass.h"
#include "pass/no_op_elimination_pass.h"
#include "pass/resize_concat_fusion_pass.h"
#include "pass/reshape_chain_elimination_pass.h"
#include "pass/sigmoid_mul_fusion_pass.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "pass/yolo_decode_fusion_pass.h"
#include "model/model_format.h"
#include "util/bf16.h"

using feather::DataType;
using feather::DeadNodeEliminationPass;
using feather::DeviceType;
using feather::GraphPass;
using feather::IdentityEliminationPass;
using feather::MatMulAddFusionPass;
using feather::QwenDepthwiseConvFusionPass;
using feather::QwenGatedDeltaFusionPass;
using feather::QwenMatMulAddFusionPass;
using feather::QwenStateOutputAliasPass;
using feather::NoOpEliminationPass;
using feather::PassManager;
using feather::ResizeConcatFusionPass;
using feather::ReshapeChainEliminationPass;
using feather::SigmoidMulFusionPass;
using feather::StaticGraph;
using feather::Tensor;
using feather::YoloDecodeFusionPass;
using feather::model::ModelDesc;
using feather::model::NodeDesc;
using feather::model::ValueDesc;

namespace {

ModelDesc BuildConvReluModelDesc() {
    ModelDesc model;
    model.name = "conv_relu_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"relu_out"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {3, 3};
    input.tensor.data_type = DataType::FP32;

    ValueDesc weight;
    weight.tensor.name = "weight";
    weight.tensor.dims = {2, 2};
    weight.tensor.data_type = DataType::FP32;
    weight.constant = true;

    ValueDesc bias;
    bias.tensor.name = "bias";
    bias.tensor.dims = {2, 2};
    bias.tensor.data_type = DataType::FP32;
    bias.constant = true;

    ValueDesc conv_out;
    conv_out.tensor.name = "conv_out";
    conv_out.tensor.dims = {2, 2};
    conv_out.tensor.data_type = DataType::FP32;

    ValueDesc relu_out;
    relu_out.tensor.name = "relu_out";
    relu_out.tensor.dims = {2, 2};
    relu_out.tensor.data_type = DataType::FP32;

    NodeDesc conv;
    conv.name = "conv0";
    conv.op_type = "Conv2D";
    conv.inputs = {"input", "weight", "bias"};
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

    model.graph.values = {input, weight, bias, conv_out, relu_out};
    model.graph.nodes = {conv, relu};
    return model;
}

ModelDesc BuildConvReluDeadTailModelDesc() {
    ModelDesc model = BuildConvReluModelDesc();
    model.name = "conv_relu_dead_tail_graph";
    model.graph.outputs = {"conv_out"};
    return model;
}

ModelDesc BuildSigmoidMulModelDesc(bool reverse_mul_inputs) {
    ModelDesc model;
    model.name = "sigmoid_mul_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {4};
    input.tensor.data_type = DataType::FP32;

    ValueDesc sigmoid_out;
    sigmoid_out.tensor.name = "sigmoid_out";
    sigmoid_out.tensor.dims = {4};
    sigmoid_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {4};
    output.tensor.data_type = DataType::FP32;

    NodeDesc sigmoid;
    sigmoid.name = "sigmoid0";
    sigmoid.op_type = "Sigmoid";
    sigmoid.inputs = {"input"};
    sigmoid.outputs = {"sigmoid_out"};

    NodeDesc mul;
    mul.name = "mul0";
    mul.op_type = "Mul";
    mul.inputs = reverse_mul_inputs ? std::vector<std::string>{"sigmoid_out", "input"}
                                    : std::vector<std::string>{"input", "sigmoid_out"};
    mul.outputs = {"output"};

    model.graph.values = {input, sigmoid_out, output};
    model.graph.nodes = {sigmoid, mul};
    return model;
}

ModelDesc BuildIdentityRelayModelDesc(bool identity_output_is_graph_output = false) {
    ModelDesc model;
    model.name = "identity_relay_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {identity_output_is_graph_output ? "identity_out" : "output"};

    ValueDesc input;
    input.tensor.name = "input";
    input.tensor.dims = {2, 2};
    input.tensor.data_type = DataType::FP32;

    ValueDesc identity_out;
    identity_out.tensor.name = "identity_out";
    identity_out.tensor.dims = {2, 2};
    identity_out.tensor.data_type = DataType::FP32;

    ValueDesc output;
    output.tensor.name = "output";
    output.tensor.dims = {2, 2};
    output.tensor.data_type = DataType::FP32;

    NodeDesc identity;
    identity.name = "identity0";
    identity.op_type = "Identity";
    identity.inputs = {"input"};
    identity.outputs = {"identity_out"};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"identity_out"};
    relu.outputs = {"output"};

    model.graph.values = {input, identity_out, output};
    model.graph.nodes = identity_output_is_graph_output ? std::vector<NodeDesc>{identity} : std::vector<NodeDesc>{identity, relu};
    return model;
}

ModelDesc BuildQwenStateOutputIdentityModelDesc() {
    ModelDesc model;
    model.name = "qwen_state_output_alias_graph";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"cache_state"};
    model.graph.outputs = {"next_cache_state"};

    auto value = [](const std::string& name) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = {1, 4};
        desc.tensor.data_type = DataType::BF16;
        return desc;
    };
    model.graph.values = {value("cache_state"), value("state_raw"), value("next_cache_state")};

    NodeDesc produce;
    produce.name = "state_producer";
    produce.op_type = "Identity";
    produce.inputs = {"cache_state"};
    produce.outputs = {"state_raw"};

    NodeDesc output;
    output.name = "state_output";
    output.op_type = "Identity";
    output.inputs = {"state_raw"};
    output.outputs = {"next_cache_state"};
    model.graph.nodes = {produce, output};
    return model;
}

ModelDesc BuildMatMulAddModelDesc(bool keep_matmul_output_live = false) {
    ModelDesc model;
    model.name = "matmul_add_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"a", "b", "bias"};
    model.graph.outputs = keep_matmul_output_live ? std::vector<std::string>{"output", "matmul_out"}
                                                  : std::vector<std::string>{"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("a", {2, 3}), value("b", {3, 4}), value("bias", {4}), value("matmul_out", {2, 4}),
                          value("output", {2, 4})};

    NodeDesc matmul;
    matmul.name = "matmul0";
    matmul.op_type = "MatMul";
    matmul.inputs = {"a", "b"};
    matmul.outputs = {"matmul_out"};

    NodeDesc add;
    add.name = "add0";
    add.op_type = "Add";
    add.inputs = {"matmul_out", "bias"};
    add.outputs = {"output"};

    model.graph.nodes = {matmul, add};
    return model;
}

ModelDesc BuildQwenMatMulAddModelDesc() {
    ModelDesc model;
    model.name = "qwen_matmul_add_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"a", "b", "residual"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("a", {1, 1, 3}), value("b", {3, 4}), value("residual", {1, 1, 4}),
                          value("matmul_out", {1, 1, 4}), value("output", {1, 1, 4})};

    NodeDesc matmul;
    matmul.name = "qwen_matmul0";
    matmul.op_type = "MatMul";
    matmul.inputs = {"a", "b"};
    matmul.outputs = {"matmul_out"};

    NodeDesc add;
    add.name = "qwen_add0";
    add.op_type = "Add";
    add.inputs = {"matmul_out", "residual"};
    add.outputs = {"output"};

    model.graph.nodes = {matmul, add};
    return model;
}

ModelDesc BuildQwenDepthwiseConvModelDesc(bool prefix_has_user = false, bool prefix_is_graph_output = false) {
    constexpr int64_t kChannels = 3;
    ModelDesc model;
    model.name = "qwen_depthwise_conv_graph";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"state", "mixed", "weight", "reshape_shape"};
    model.graph.outputs = {"conv_out", "next_state"};
    if (prefix_is_graph_output) {
        model.graph.outputs.push_back("discarded_prefix");
    }

    auto value = [](const std::string& name, std::vector<int64_t> dims, DataType data_type) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = data_type;
        return desc;
    };
    model.graph.values = {
        value("state", {1, kChannels, 3}, DataType::BF16),
        value("mixed", {1, kChannels, 1}, DataType::BF16),
        value("weight", {kChannels, 1, 1, 4}, DataType::BF16),
        value("reshape_shape", {4}, DataType::INT64),
        value("joined", {1, kChannels, 4}, DataType::BF16),
        value("conv_input", {1, kChannels, 1, 4}, DataType::BF16),
        value("conv_out", {1, kChannels, 1, 1}, DataType::BF16),
        value("discarded_prefix", {1, kChannels, 1}, DataType::BF16),
        value("next_state", {1, kChannels, 3}, DataType::BF16),
        value("prefix_user_out", {1, kChannels, 1}, DataType::BF16),
    };

    NodeDesc concat;
    concat.name = "state_concat";
    concat.op_type = "Concat";
    concat.inputs = {"state", "mixed"};
    concat.outputs = {"joined"};
    concat.attributes["axis"] = static_cast<int64_t>(2);

    NodeDesc reshape;
    reshape.name = "state_reshape";
    reshape.op_type = "Reshape";
    reshape.inputs = {"joined", "reshape_shape"};
    reshape.outputs = {"conv_input"};

    NodeDesc conv;
    conv.name = "state_conv";
    conv.op_type = "Conv2D";
    conv.inputs = {"conv_input", "weight"};
    conv.outputs = {"conv_out"};
    conv.attributes["group"] = kChannels;
    conv.attributes["stride_h"] = static_cast<int64_t>(1);
    conv.attributes["stride_w"] = static_cast<int64_t>(1);
    conv.attributes["pad_h"] = static_cast<int64_t>(0);
    conv.attributes["pad_w"] = static_cast<int64_t>(0);
    conv.attributes["dilation_h"] = static_cast<int64_t>(1);
    conv.attributes["dilation_w"] = static_cast<int64_t>(1);

    NodeDesc split;
    split.name = "state_split";
    split.op_type = "Split";
    split.inputs = {"joined"};
    split.outputs = {"discarded_prefix", "next_state"};
    split.attributes["axis"] = static_cast<int64_t>(2);
    split.attributes["split_sizes"] = std::vector<int64_t>{1, 3};

    model.graph.nodes = {concat, reshape, conv, split};
    if (prefix_has_user) {
        NodeDesc prefix_user;
        prefix_user.name = "prefix_user";
        prefix_user.op_type = "Identity";
        prefix_user.inputs = {"discarded_prefix"};
        prefix_user.outputs = {"prefix_user_out"};
        model.graph.nodes.push_back(prefix_user);
    }
    return model;
}

ModelDesc BuildQwenGatedDeltaModelDesc() {
    ModelDesc model;
    model.name = "qwen_gated_delta_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"state", "k", "v", "beta", "decay", "q"};
    model.graph.outputs = {"next_state", "core"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {
        value("state", {1, 2, 3, 4}),
        value("k", {1, 2, 3}),
        value("v", {1, 2, 4}),
        value("beta", {1, 2}),
        value("decay", {1, 2}),
        value("q", {1, 2, 3}),
        value("decay_unsqueezed", {1, 2, 1}),
        value("decay_broadcast", {1, 2, 1, 1}),
        value("state_decay", {1, 2, 3, 4}),
        value("k_col", {1, 2, 3, 1}),
        value("kv_mem_full", {1, 2, 3, 4}),
        value("kv_mem", {1, 2, 4}),
        value("v_delta", {1, 2, 4}),
        value("beta_broadcast", {1, 2, 1}),
        value("delta", {1, 2, 4}),
        value("delta_row", {1, 2, 1, 4}),
        value("update", {1, 2, 3, 4}),
        value("next_state", {1, 2, 3, 4}),
        value("q_col", {1, 2, 3, 1}),
        value("output_full", {1, 2, 3, 4}),
        value("core", {1, 2, 4}),
    };

    NodeDesc decay_unsqueeze;
    decay_unsqueeze.name = "decay_unsqueeze";
    decay_unsqueeze.op_type = "Unsqueeze";
    decay_unsqueeze.inputs = {"decay"};
    decay_unsqueeze.outputs = {"decay_unsqueezed"};
    decay_unsqueeze.attributes["axes"] = std::vector<int64_t>{2};

    NodeDesc decay_unsqueeze_last;
    decay_unsqueeze_last.name = "decay_unsqueeze_last";
    decay_unsqueeze_last.op_type = "Unsqueeze";
    decay_unsqueeze_last.inputs = {"decay_unsqueezed"};
    decay_unsqueeze_last.outputs = {"decay_broadcast"};
    decay_unsqueeze_last.attributes["axes"] = std::vector<int64_t>{3};

    NodeDesc state_decay;
    state_decay.name = "state_decay";
    state_decay.op_type = "Mul";
    state_decay.inputs = {"state", "decay_broadcast"};
    state_decay.outputs = {"state_decay"};

    NodeDesc k_col;
    k_col.name = "k_col";
    k_col.op_type = "Unsqueeze";
    k_col.inputs = {"k"};
    k_col.outputs = {"k_col"};
    k_col.attributes["axes"] = std::vector<int64_t>{3};

    NodeDesc kv_mem_full;
    kv_mem_full.name = "kv_mem_full";
    kv_mem_full.op_type = "Mul";
    kv_mem_full.inputs = {"state_decay", "k_col"};
    kv_mem_full.outputs = {"kv_mem_full"};

    NodeDesc kv_mem;
    kv_mem.name = "kv_mem";
    kv_mem.op_type = "ReduceSum";
    kv_mem.inputs = {"kv_mem_full"};
    kv_mem.outputs = {"kv_mem"};
    kv_mem.attributes["axes"] = std::vector<int64_t>{2};
    kv_mem.attributes["keepdims"] = static_cast<int64_t>(0);

    NodeDesc v_delta;
    v_delta.name = "v_delta";
    v_delta.op_type = "Sub";
    v_delta.inputs = {"v", "kv_mem"};
    v_delta.outputs = {"v_delta"};

    NodeDesc beta_unsqueeze;
    beta_unsqueeze.name = "beta_unsqueeze";
    beta_unsqueeze.op_type = "Unsqueeze";
    beta_unsqueeze.inputs = {"beta"};
    beta_unsqueeze.outputs = {"beta_broadcast"};
    beta_unsqueeze.attributes["axes"] = std::vector<int64_t>{2};

    NodeDesc delta;
    delta.name = "delta";
    delta.op_type = "Mul";
    delta.inputs = {"v_delta", "beta_broadcast"};
    delta.outputs = {"delta"};

    NodeDesc delta_row;
    delta_row.name = "delta_row";
    delta_row.op_type = "Unsqueeze";
    delta_row.inputs = {"delta"};
    delta_row.outputs = {"delta_row"};
    delta_row.attributes["axes"] = std::vector<int64_t>{2};

    NodeDesc update;
    update.name = "update";
    update.op_type = "Mul";
    update.inputs = {"k_col", "delta_row"};
    update.outputs = {"update"};

    NodeDesc next_state;
    next_state.name = "next_state_update";
    next_state.op_type = "Add";
    next_state.inputs = {"state_decay", "update"};
    next_state.outputs = {"next_state"};

    NodeDesc q_col;
    q_col.name = "q_col";
    q_col.op_type = "Unsqueeze";
    q_col.inputs = {"q"};
    q_col.outputs = {"q_col"};
    q_col.attributes["axes"] = std::vector<int64_t>{3};

    NodeDesc output_full;
    output_full.name = "output_full";
    output_full.op_type = "Mul";
    output_full.inputs = {"next_state", "q_col"};
    output_full.outputs = {"output_full"};

    NodeDesc core;
    core.name = "core";
    core.op_type = "ReduceSum";
    core.inputs = {"output_full"};
    core.outputs = {"core"};
    core.attributes["axes"] = std::vector<int64_t>{2};
    core.attributes["keepdims"] = static_cast<int64_t>(0);

    model.graph.nodes = {decay_unsqueeze, decay_unsqueeze_last, state_decay, k_col, kv_mem_full, kv_mem, v_delta,
                         beta_unsqueeze, delta, delta_row, update, next_state, q_col, output_full, core};
    return model;
}

ModelDesc BuildQwenGatedDeltaReversedSubModelDesc() {
    auto model = BuildQwenGatedDeltaModelDesc();
    for (auto& node : model.graph.nodes) {
        if (node.name == "v_delta") {
            node.inputs = {"kv_mem", "v"};
            break;
        }
    }
    return model;
}

ModelDesc BuildQwenGatedDeltaIntermediateOutputModelDesc() {
    auto model = BuildQwenGatedDeltaModelDesc();
    model.graph.outputs.push_back("state_decay");
    return model;
}

ModelDesc BuildNoOpReshapeModelDesc() {
    ModelDesc model;
    model.name = "no_op_reshape_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {2, 2}), value("reshape_out", {2, 2}), value("output", {2, 2})};

    NodeDesc reshape;
    reshape.name = "reshape0";
    reshape.op_type = "Reshape";
    reshape.inputs = {"input"};
    reshape.outputs = {"reshape_out"};
    reshape.attributes["shape"] = std::vector<int64_t>{2, 2};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"reshape_out"};
    relu.outputs = {"output"};

    model.graph.nodes = {reshape, relu};
    return model;
}

ModelDesc BuildIdentityTransposeModelDesc() {
    ModelDesc model;
    model.name = "identity_transpose_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {1, 2, 3}), value("transpose_out", {1, 2, 3}), value("output", {1, 2, 3})};

    NodeDesc transpose;
    transpose.name = "transpose0";
    transpose.op_type = "Transpose";
    transpose.inputs = {"input"};
    transpose.outputs = {"transpose_out"};
    transpose.attributes["perm"] = std::vector<int64_t>{0, 1, 2};

    NodeDesc relu;
    relu.name = "relu0";
    relu.op_type = "ReLU";
    relu.inputs = {"transpose_out"};
    relu.outputs = {"output"};

    model.graph.nodes = {transpose, relu};
    return model;
}

ModelDesc BuildReshapeChainModelDesc(bool keep_inner_output_live = false) {
    ModelDesc model;
    model.name = "reshape_chain_graph";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"input"};
    model.graph.outputs = keep_inner_output_live ? std::vector<std::string>{"output", "reshape0_out"}
                                                 : std::vector<std::string>{"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("input", {2, 3}), value("reshape0_out", {1, 6}), value("output", {3, 2})};

    NodeDesc reshape0;
    reshape0.name = "reshape0";
    reshape0.op_type = "Reshape";
    reshape0.inputs = {"input"};
    reshape0.outputs = {"reshape0_out"};
    reshape0.attributes["shape"] = std::vector<int64_t>{1, 6};

    NodeDesc reshape1;
    reshape1.name = "reshape1";
    reshape1.op_type = "Reshape";
    reshape1.inputs = {"reshape0_out"};
    reshape1.outputs = {"output"};
    reshape1.attributes["shape"] = std::vector<int64_t>{3, 2};

    model.graph.nodes = {reshape0, reshape1};
    return model;
}

ModelDesc BuildYoloDecodePatternModelDesc() {
    ModelDesc model;
    model.name = "yolo_decode_pattern";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"raw"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims, bool constant = false) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        desc.constant = constant;
        return desc;
    };

    model.graph.values = {
        value("raw", {1, 12, 1, 1}),
        value("xy_scale", {1}, true),
        value("grid", {1, 2, 1, 1, 2}, true),
        value("stride", {1}, true),
        value("wh_scale", {1}, true),
        value("anchor", {1, 2, 1, 1, 2}, true),
        value("reshaped", {1, 2, 6, 1, 1}),
        value("transposed", {1, 2, 1, 1, 6}),
        value("sigmoid", {1, 2, 1, 1, 6}),
        value("split_xy", {1, 2, 1, 1, 2}),
        value("split_wh", {1, 2, 1, 1, 2}),
        value("split_rest", {1, 2, 1, 1, 2}),
        value("mul_xy", {1, 2, 1, 1, 2}),
        value("add_xy", {1, 2, 1, 1, 2}),
        value("decoded_xy", {1, 2, 1, 1, 2}),
        value("mul_wh", {1, 2, 1, 1, 2}),
        value("pow_wh", {1, 2, 1, 1, 2}),
        value("decoded_wh", {1, 2, 1, 1, 2}),
        value("concat", {1, 2, 1, 1, 6}),
        value("output", {1, 2, 6}),
    };

    NodeDesc reshape0;
    reshape0.name = "reshape0";
    reshape0.op_type = "Reshape";
    reshape0.inputs = {"raw"};
    reshape0.outputs = {"reshaped"};
    reshape0.attributes["shape"] = std::vector<int64_t>{1, 2, 6, 1, 1};

    NodeDesc transpose;
    transpose.name = "transpose0";
    transpose.op_type = "Transpose";
    transpose.inputs = {"reshaped"};
    transpose.outputs = {"transposed"};
    transpose.attributes["perm"] = std::vector<int64_t>{0, 1, 3, 4, 2};

    NodeDesc sigmoid;
    sigmoid.name = "sigmoid0";
    sigmoid.op_type = "Sigmoid";
    sigmoid.inputs = {"transposed"};
    sigmoid.outputs = {"sigmoid"};

    NodeDesc split;
    split.name = "split0";
    split.op_type = "Split";
    split.inputs = {"sigmoid"};
    split.outputs = {"split_xy", "split_wh", "split_rest"};
    split.attributes["axis"] = static_cast<int64_t>(4);
    split.attributes["split_sizes"] = std::vector<int64_t>{2, 2, 2};

    NodeDesc mul_xy;
    mul_xy.name = "mul_xy";
    mul_xy.op_type = "Mul";
    mul_xy.inputs = {"split_xy", "xy_scale"};
    mul_xy.outputs = {"mul_xy"};

    NodeDesc add_xy;
    add_xy.name = "add_xy";
    add_xy.op_type = "Add";
    add_xy.inputs = {"mul_xy", "grid"};
    add_xy.outputs = {"add_xy"};

    NodeDesc mul_xy_stride;
    mul_xy_stride.name = "mul_xy_stride";
    mul_xy_stride.op_type = "Mul";
    mul_xy_stride.inputs = {"add_xy", "stride"};
    mul_xy_stride.outputs = {"decoded_xy"};

    NodeDesc mul_wh;
    mul_wh.name = "mul_wh";
    mul_wh.op_type = "Mul";
    mul_wh.inputs = {"split_wh", "wh_scale"};
    mul_wh.outputs = {"mul_wh"};

    NodeDesc pow_wh;
    pow_wh.name = "pow_wh";
    pow_wh.op_type = "Pow";
    pow_wh.inputs = {"mul_wh"};
    pow_wh.outputs = {"pow_wh"};
    pow_wh.attributes["exponent"] = 2.0f;

    NodeDesc mul_wh_anchor;
    mul_wh_anchor.name = "mul_wh_anchor";
    mul_wh_anchor.op_type = "Mul";
    mul_wh_anchor.inputs = {"pow_wh", "anchor"};
    mul_wh_anchor.outputs = {"decoded_wh"};

    NodeDesc concat;
    concat.name = "concat0";
    concat.op_type = "Concat";
    concat.inputs = {"decoded_xy", "decoded_wh", "split_rest"};
    concat.outputs = {"concat"};
    concat.attributes["axis"] = static_cast<int64_t>(4);

    NodeDesc reshape1;
    reshape1.name = "reshape1";
    reshape1.op_type = "Reshape";
    reshape1.inputs = {"concat"};
    reshape1.outputs = {"output"};
    reshape1.attributes["shape"] = std::vector<int64_t>{1, 2, 6};

    model.graph.nodes = {reshape0, transpose, sigmoid, split, mul_xy, add_xy, mul_xy_stride,
                         mul_wh,   pow_wh,    mul_wh_anchor, concat, reshape1};
    return model;
}

ModelDesc BuildResizeConcatPatternModelDesc() {
    ModelDesc model;
    model.name = "resize_concat_pattern";
    model.version = 1;
    model.graph.name = "main";
    model.graph.inputs = {"resize_input", "skip_input"};
    model.graph.outputs = {"output"};

    auto value = [](const std::string& name, std::vector<int64_t> dims) {
        ValueDesc desc;
        desc.tensor.name = name;
        desc.tensor.dims = std::move(dims);
        desc.tensor.data_type = DataType::FP32;
        return desc;
    };

    model.graph.values = {value("resize_input", {1, 64, 40, 40}),
                          value("skip_input", {1, 32, 80, 80}),
                          value("resize_out", {1, 64, 80, 80}),
                          value("output", {1, 96, 80, 80})};

    NodeDesc resize;
    resize.name = "resize0";
    resize.op_type = "Resize";
    resize.inputs = {"resize_input"};
    resize.outputs = {"resize_out"};
    resize.attributes["scales"] = std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f};

    NodeDesc concat;
    concat.name = "concat0";
    concat.op_type = "Concat";
    concat.inputs = {"resize_out", "skip_input"};
    concat.outputs = {"output"};
    concat.attributes["axis"] = static_cast<int64_t>(1);

    model.graph.nodes = {resize, concat};
    return model;
}

void BindSiluGraphInputs(StaticGraph* graph) {
    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({-2.0f, -1.0f, 0.0f, 2.0f}, {4});
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindIdentityRelayInputs(StaticGraph* graph) {
    auto input_tensor = std::make_shared<Tensor>();
    input_tensor->Assign<float>({-1.0f, 2.0f, -3.0f, 4.0f}, {2, 2});
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindMatMulAddInputs(StaticGraph* graph) {
    auto a = std::make_shared<Tensor>();
    a->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = std::make_shared<Tensor>();
    b->Assign<float>({1.0f, 0.0f, 2.0f, 1.0f, 0.0f, 1.0f, 3.0f, 2.0f, 1.0f, 0.0f, 1.0f, 4.0f}, {3, 4});
    auto bias = std::make_shared<Tensor>();
    bias->Assign<float>({0.5f, -1.0f, 1.5f, 0.0f}, {4});
    ASSERT_EQ(graph->SetTensor("a", a), 0);
    ASSERT_EQ(graph->SetTensor("b", b), 0);
    ASSERT_EQ(graph->SetTensor("bias", bias), 0);
}

void BindQwenMatMulAddInputs(StaticGraph* graph) {
    auto a = std::make_shared<Tensor>();
    a->Assign<float>({1.0f, 2.0f, 3.0f}, {1, 1, 3});
    auto b = std::make_shared<Tensor>();
    b->Assign<float>({1.0f, 0.0f, 2.0f, 1.0f, 0.0f, 1.0f, 3.0f, 2.0f, 1.0f, 0.0f, 1.0f, 4.0f}, {3, 4});
    auto residual = std::make_shared<Tensor>();
    residual->Assign<float>({0.5f, -1.0f, 1.5f, 0.0f}, {1, 1, 4});
    ASSERT_EQ(graph->SetTensor("a", a), 0);
    ASSERT_EQ(graph->SetTensor("b", b), 0);
    ASSERT_EQ(graph->SetTensor("residual", residual), 0);
}

void BindQwenDepthwiseConvInputs(StaticGraph* graph) {
    constexpr int64_t kChannels = 3;
    std::vector<feather::BFloat16> state_values;
    std::vector<feather::BFloat16> mixed_values;
    std::vector<feather::BFloat16> weight_values;
    state_values.reserve(static_cast<size_t>(kChannels * 3));
    mixed_values.reserve(static_cast<size_t>(kChannels));
    weight_values.reserve(static_cast<size_t>(kChannels * 4));
    for (int64_t channel = 0; channel < kChannels; ++channel) {
        for (int64_t offset = 0; offset < 3; ++offset) {
            state_values.push_back(feather::BFloat16{feather::FloatToBFloat16(
                0.125f * static_cast<float>(channel * 3 + offset - 4))});
        }
        mixed_values.push_back(
            feather::BFloat16{feather::FloatToBFloat16(0.0625f * static_cast<float>(channel * 5 - 3))});
        for (int64_t offset = 0; offset < 4; ++offset) {
            weight_values.push_back(feather::BFloat16{feather::FloatToBFloat16(
                0.03125f * static_cast<float>(channel * 7 + offset * 3 - 8))});
        }
    }

    auto state = std::make_shared<Tensor>();
    state->Assign<feather::BFloat16>(state_values, {1, kChannels, 3});
    auto mixed = std::make_shared<Tensor>();
    mixed->Assign<feather::BFloat16>(mixed_values, {1, kChannels, 1});
    auto weight = std::make_shared<Tensor>();
    weight->Assign<feather::BFloat16>(weight_values, {kChannels, 1, 1, 4});
    auto reshape_shape = std::make_shared<Tensor>();
    reshape_shape->Assign<int64_t>({1, kChannels, 1, 4}, {4});
    ASSERT_EQ(graph->SetTensor("state", state), 0);
    ASSERT_EQ(graph->SetTensor("mixed", mixed), 0);
    ASSERT_EQ(graph->SetTensor("weight", weight), 0);
    ASSERT_EQ(graph->SetTensor("reshape_shape", reshape_shape), 0);
}

void BindQwenGatedDeltaInputs(StaticGraph* graph) {
    auto state = std::make_shared<Tensor>();
    state->Assign<float>({
        0.5f, -0.25f, 0.75f, 1.0f,
        -0.5f, 0.25f, -0.75f, 0.5f,
        0.125f, -1.0f, 0.625f, -0.375f,
        -0.75f, 0.25f, 1.5f, -0.5f,
        0.375f, -0.625f, 0.875f, 0.125f,
        -0.25f, 0.5f, -1.25f, 0.75f,
    }, {1, 2, 3, 4});
    auto k = std::make_shared<Tensor>();
    k->Assign<float>({0.25f, -0.5f, 0.75f, -0.4f, 0.6f, 0.2f}, {1, 2, 3});
    auto v = std::make_shared<Tensor>();
    v->Assign<float>({0.5f, -0.25f, 0.75f, -1.0f, -0.5f, 0.25f, 0.125f, 0.875f}, {1, 2, 4});
    auto beta = std::make_shared<Tensor>();
    beta->Assign<float>({0.2f, 0.35f}, {1, 2});
    auto decay = std::make_shared<Tensor>();
    decay->Assign<float>({0.9f, 0.8f}, {1, 2});
    auto q = std::make_shared<Tensor>();
    q->Assign<float>({0.3f, -0.1f, 0.5f, -0.2f, 0.4f, 0.6f}, {1, 2, 3});
    ASSERT_EQ(graph->SetTensor("state", state), 0);
    ASSERT_EQ(graph->SetTensor("k", k), 0);
    ASSERT_EQ(graph->SetTensor("v", v), 0);
    ASSERT_EQ(graph->SetTensor("beta", beta), 0);
    ASSERT_EQ(graph->SetTensor("decay", decay), 0);
    ASSERT_EQ(graph->SetTensor("q", q), 0);
}

std::pair<std::vector<float>, std::vector<float>> RunQwenGatedDeltaGraph(DeviceType device, bool apply_fusion) {
    StaticGraph graph;
    graph.SetKernelDevice(device);
    if (graph.SetModel(BuildQwenGatedDeltaModelDesc()) != 0) return {};
    BindQwenGatedDeltaInputs(&graph);
    if (graph.Build() != 0) return {};

    auto pass_manager = std::make_shared<PassManager>();
    if (apply_fusion) pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);
    if (graph.ApplyPasses() != 0) return {};

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    if (lowering.Lower(graph, &runtime) != 0 || runtime.Run() != 0) return {};
    const auto next_state = runtime.GetTensor("next_state");
    const auto core = runtime.GetTensor("core");
    if (next_state == nullptr || core == nullptr || next_state->data_type() != DataType::FP32 ||
        core->data_type() != DataType::FP32) {
        return {};
    }
    return {{next_state->data<float>(), next_state->data<float>() + next_state->numel()},
            {core->data<float>(), core->data<float>() + core->numel()}};
}

std::pair<std::vector<uint16_t>, std::vector<uint16_t>> RunQwenDepthwiseConvGraph(DeviceType device,
                                                                                     bool apply_fusion) {
    StaticGraph graph;
    graph.SetKernelDevice(device);
    if (graph.SetModel(BuildQwenDepthwiseConvModelDesc()) != 0) return {};
    BindQwenDepthwiseConvInputs(&graph);
    if (graph.Build() != 0) return {};

    auto pass_manager = std::make_shared<PassManager>();
    if (apply_fusion) pass_manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
    graph.SetPassManager(pass_manager);
    if (graph.ApplyPasses() != 0) return {};

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    if (lowering.Lower(graph, &runtime) != 0 || runtime.Run() != 0) return {};
    const auto conv_out = runtime.GetTensor("conv_out");
    const auto next_state = runtime.GetTensor("next_state");
    if (conv_out == nullptr || next_state == nullptr || conv_out->data_type() != DataType::BF16 ||
        next_state->data_type() != DataType::BF16) {
        return {};
    }

    const auto* conv_data = conv_out->data<feather::BFloat16>();
    const auto* state_data = next_state->data<feather::BFloat16>();
    if (conv_data == nullptr || state_data == nullptr) return {};
    std::vector<uint16_t> conv_values(static_cast<size_t>(conv_out->numel()));
    std::vector<uint16_t> state_values(static_cast<size_t>(next_state->numel()));
    for (int64_t index = 0; index < conv_out->numel(); ++index) {
        conv_values[static_cast<size_t>(index)] = conv_data[index].bits;
    }
    for (int64_t index = 0; index < next_state->numel(); ++index) {
        state_values[static_cast<size_t>(index)] = state_data[index].bits;
    }
    return {std::move(conv_values), std::move(state_values)};
}

void BindNoOpGraphInputs(StaticGraph* graph, const std::vector<int64_t>& dims) {
    auto input_tensor = std::make_shared<Tensor>();
    const int64_t numel = std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<float> values(static_cast<size_t>(numel), 1.0f);
    input_tensor->Assign<float>(values, dims);
    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
}

void BindConvReluGraphInputs(StaticGraph* graph) {
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

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->Assign<float>({
        1, 1,
        1, 1,
    }, {2, 2});

    ASSERT_EQ(graph->SetTensor("input", input_tensor), 0);
    ASSERT_EQ(graph->SetTensor("weight", weight_tensor), 0);
    ASSERT_EQ(graph->SetTensor("bias", bias_tensor), 0);
}

void BindYoloDecodePatternInputs(StaticGraph* graph) {
    auto raw = std::make_shared<Tensor>();
    raw->Assign<float>(
        {
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
            2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f,
        },
        {1, 12, 1, 1});
    auto xy_scale = std::make_shared<Tensor>();
    xy_scale->Assign<float>({2.0f}, {1});
    auto grid = std::make_shared<Tensor>();
    grid->Assign<float>({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2});
    auto stride = std::make_shared<Tensor>();
    stride->Assign<float>({8.0f}, {1});
    auto wh_scale = std::make_shared<Tensor>();
    wh_scale->Assign<float>({2.0f}, {1});
    auto anchor = std::make_shared<Tensor>();
    anchor->Assign<float>({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2});

    ASSERT_EQ(graph->SetTensor("raw", raw), 0);
    ASSERT_EQ(graph->SetTensor("xy_scale", xy_scale), 0);
    ASSERT_EQ(graph->SetTensor("grid", grid), 0);
    ASSERT_EQ(graph->SetTensor("stride", stride), 0);
    ASSERT_EQ(graph->SetTensor("wh_scale", wh_scale), 0);
    ASSERT_EQ(graph->SetTensor("anchor", anchor), 0);
}

void BindResizeConcatPatternInputs(StaticGraph* graph) {
    auto resize_input = std::make_shared<Tensor>();
    resize_input->Assign<float>(std::vector<float>(static_cast<size_t>(1 * 64 * 40 * 40), 1.0f), {1, 64, 40, 40});
    auto skip_input = std::make_shared<Tensor>();
    skip_input->Assign<float>(std::vector<float>(static_cast<size_t>(1 * 32 * 80 * 80), 2.0f), {1, 32, 80, 80});

    ASSERT_EQ(graph->SetTensor("resize_input", resize_input), 0);
    ASSERT_EQ(graph->SetTensor("skip_input", skip_input), 0);
}

class RecordingPass : public GraphPass {
   public:
    explicit RecordingPass(std::vector<std::string>* trace, std::string label)
        : trace_(trace), label_(std::move(label)) {}

    const std::string& name() const override { return label_; }

    int32_t Run(StaticGraph* graph) override {
        if (graph == nullptr || trace_ == nullptr) {
            return -1;
        }
        trace_->push_back(label_);
        return 0;
    }

   private:
    std::vector<std::string>* trace_;
    std::string label_;
};

}  // namespace

TEST(static_graph_pass_test, BuildCapturesNodeMetadataAndUseDef) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);

    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.NodeSize(), 2U);

    const auto* conv_node = graph.GetNode("conv0");
    ASSERT_NE(conv_node, nullptr);
    EXPECT_EQ(conv_node->op_type, "Conv2D");
    EXPECT_EQ(conv_node->outputs, std::vector<std::string>({"conv_out"}));

    const auto* relu_node = graph.GetNode("relu0");
    ASSERT_NE(relu_node, nullptr);
    EXPECT_EQ(relu_node->inputs, std::vector<std::string>({"conv_out"}));

    EXPECT_EQ(graph.GetProducer("conv_out"), "conv0");
    EXPECT_EQ(graph.GetUsers("conv_out"), std::vector<std::string>({"relu0"}));
}

TEST(static_graph_pass_test, BuildCapturesProducerAndUsersForGraphValues) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);

    ASSERT_EQ(graph.Build(), 0);
    EXPECT_EQ(graph.GetProducer("input"), "");
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"conv0"}));
    EXPECT_EQ(graph.GetProducer("relu_out"), "relu0");
    EXPECT_TRUE(graph.GetUsers("relu_out").empty());
}

TEST(static_graph_pass_test, ApplyPassesRunsRegisteredPassesInOrder) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "first"));
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "second"));

    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"first", "second"}));
}

TEST(static_graph_pass_test, RemoveNodeUpdatesUseDefAndOperatorView) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluDeadTailModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_TRUE(graph.RemoveNode("relu0"));
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_TRUE(graph.GetUsers("conv_out").empty());
    EXPECT_EQ(graph.GetProducer("relu_out"), "");
    EXPECT_EQ(graph.GetNode("relu0"), nullptr);
}

TEST(static_graph_pass_test, ApplyPassesUsesInstalledPassManager) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "installed"));
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"installed"}));
}

TEST(static_graph_pass_test, SetModelPreservesPreinstalledPassManager) {
    StaticGraph graph;
    std::vector<std::string> trace;
    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<RecordingPass>(&trace, "preinstalled"));
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"preinstalled"}));
}

TEST(static_graph_pass_test, ReplaceInputValueRebuildsAffectedNode) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_TRUE(graph.ReplaceInputValue("relu0", "conv_out", "input"));
    EXPECT_TRUE(graph.GetUsers("conv_out").empty());
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"conv0", "relu0"}));

    auto output_tensor = graph.GetTensor("relu_out");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({3, 3}));
}

TEST(static_graph_pass_test, DeadNodeEliminationPassRemovesUnusedTailNodes) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildConvReluDeadTailModelDesc()), 0);
    BindConvReluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_EQ(graph.GetNode("relu0"), nullptr);
}

TEST(static_graph_pass_test, SigmoidMulFusionPassRewritesSiluPattern) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildSigmoidMulModelDesc(false)), 0);
    BindSiluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_EQ(graph.GetNode("sigmoid0"), nullptr);

    const auto* fused = graph.GetNode("mul0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "SiLU");
    EXPECT_EQ(fused->inputs, std::vector<std::string>({"input"}));
    EXPECT_EQ(fused->outputs, std::vector<std::string>({"output"}));
    EXPECT_TRUE(graph.GetUsers("sigmoid_out").empty());
    EXPECT_EQ(graph.GetUsers("input"), std::vector<std::string>({"mul0"}));
    EXPECT_EQ(graph.GetProducer("output"), "mul0");
}

TEST(static_graph_pass_test, SigmoidMulFusionPassAcceptsReversedMulInputs) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildSigmoidMulModelDesc(true)), 0);
    BindSiluGraphInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    const auto* fused = graph.GetNode("mul0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "SiLU");
    EXPECT_EQ(fused->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, IdentityEliminationPassRemovesInteriorIdentity) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(false)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<IdentityEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.GetNode("identity0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, IdentityEliminationPassPreservesGraphOutputIdentity) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(true)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<IdentityEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_NE(graph.GetNode("identity0"), nullptr);
}

TEST(static_graph_pass_test, QwenStateOutputAliasPassRemovesOnlyExplicitX86StateRelay) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildQwenStateOutputIdentityModelDesc()), 0);
    auto state = std::make_shared<Tensor>(8);
    state->Resize({1, 4});
    state->set_data_type(DataType::BF16);
    ASSERT_EQ(graph.SetTensor("cache_state", state), 0);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenStateOutputAliasPass>());
    pass_manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_NE(graph.GetNode("state_producer"), nullptr);
    EXPECT_EQ(graph.GetNode("state_output"), nullptr);
    EXPECT_EQ(graph.GetProducer("next_cache_state"), "");
    EXPECT_EQ(graph.GetTensor("next_cache_state"), graph.GetTensor("state_raw"));
}

#ifdef FEATHER_WITH_CUDA
TEST(static_graph_pass_test, QwenStateOutputAliasPassRemovesExplicitCudaStateRelay) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(BuildQwenStateOutputIdentityModelDesc()), 0);
    auto state = std::make_shared<Tensor>(8);
    state->Resize({1, 4});
    state->set_data_type(DataType::BF16);
    ASSERT_EQ(graph.SetTensor("cache_state", state), 0);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenStateOutputAliasPass>());
    pass_manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_NE(graph.GetNode("state_producer"), nullptr);
    EXPECT_EQ(graph.GetNode("state_output"), nullptr);
    EXPECT_EQ(graph.GetProducer("next_cache_state"), "");
    EXPECT_EQ(graph.GetTensor("next_cache_state"), graph.GetTensor("state_raw"));
}
#endif

TEST(static_graph_pass_test, MatMulAddFusionPassRewritesToGemm) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildMatMulAddModelDesc(false)), 0);
    BindMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.GetNode("matmul0"), nullptr);
    const auto* fused = graph.GetNode("add0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "Gemm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"a", "b", "bias"}));
}

TEST(static_graph_pass_test, MatMulAddFusionPassSkipsWhenMatMulOutputIsGraphOutput) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildMatMulAddModelDesc(true)), 0);
    BindMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_NE(graph.GetNode("matmul0"), nullptr);
    const auto* add = graph.GetNode("add0");
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(add->op_type, "Add");
}

TEST(static_graph_pass_test, QwenMatMulAddFusionPassRewritesSingletonResidual) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildQwenMatMulAddModelDesc()), 0);
    BindQwenMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenMatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.GetNode("qwen_matmul0"), nullptr);
    const auto* fused = graph.GetNode("qwen_add0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "Gemm");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"a", "b", "residual"}));
}

TEST(static_graph_pass_test, QwenMatMulAddFusionPassSkipsNonQwenModel) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildMatMulAddModelDesc(false)), 0);
    BindMatMulAddInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenMatMulAddFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_NE(graph.GetNode("matmul0"), nullptr);
    const auto* add = graph.GetNode("add0");
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(add->op_type, "Add");
}

TEST(static_graph_pass_test, QwenDepthwiseConvFusionPassRewritesStateConvOnX86) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildQwenDepthwiseConvModelDesc()), 0);
    BindQwenDepthwiseConvInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    EXPECT_EQ(graph.NodeSize(), 1U);
    const auto* conv = graph.GetNode("state_conv");
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->op_type, "QwenDepthwiseConvState");
    EXPECT_EQ(conv->inputs, (std::vector<std::string>{"state", "mixed", "weight"}));
    EXPECT_EQ(conv->outputs, (std::vector<std::string>{"conv_out", "discarded_prefix", "next_state"}));
    EXPECT_EQ(graph.GetNode("state_concat"), nullptr);
    EXPECT_EQ(graph.GetNode("state_reshape"), nullptr);
    EXPECT_EQ(graph.GetNode("state_split"), nullptr);
}

#ifdef FEATHER_WITH_CUDA
TEST(static_graph_pass_test, QwenDepthwiseConvFusionPassLowersStateConvToCuda) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(BuildQwenDepthwiseConvModelDesc()), 0);
    BindQwenDepthwiseConvInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    ASSERT_EQ(graph.NodeSize(), 1U);
    ASSERT_NE(graph.GetNode("state_conv"), nullptr);
    EXPECT_EQ(graph.GetNode("state_conv")->op_type, "QwenDepthwiseConvState");

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0);
    const auto* node = runtime.GetNode("state_conv");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->kernel_device, DeviceType::CUDA);
}
#endif

TEST(static_graph_pass_test, QwenDepthwiseConvFusionPassPreservesConvAndNextState) {
    const auto reference = RunQwenDepthwiseConvGraph(DeviceType::COMMON, false);
    const auto fused = RunQwenDepthwiseConvGraph(DeviceType::X86, true);
    ASSERT_FALSE(reference.first.empty());
    ASSERT_FALSE(reference.second.empty());
    EXPECT_EQ(fused.first, reference.first);
    EXPECT_EQ(fused.second, reference.second);
}

TEST(static_graph_pass_test, QwenDepthwiseConvFusionPassSkipsCommonDevice) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(BuildQwenDepthwiseConvModelDesc()), 0);
    BindQwenDepthwiseConvInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    EXPECT_EQ(graph.NodeSize(), 4U);
    ASSERT_NE(graph.GetNode("state_conv"), nullptr);
    EXPECT_EQ(graph.GetNode("state_conv")->op_type, "Conv2D");
}

TEST(static_graph_pass_test, QwenDepthwiseConvFusionPassPreservesObservedPrefix) {
    for (const auto prefix_is_graph_output : {false, true}) {
        StaticGraph graph;
        graph.SetKernelDevice(DeviceType::X86);
        ASSERT_EQ(graph.SetModel(BuildQwenDepthwiseConvModelDesc(!prefix_is_graph_output, prefix_is_graph_output)), 0);
        BindQwenDepthwiseConvInputs(&graph);
        ASSERT_EQ(graph.Build(), 0);

        auto pass_manager = std::make_shared<PassManager>();
        pass_manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
        graph.SetPassManager(pass_manager);
        ASSERT_EQ(graph.ApplyPasses(), 0);

        ASSERT_NE(graph.GetNode("state_conv"), nullptr);
        EXPECT_EQ(graph.GetNode("state_conv")->op_type, "Conv2D");
        EXPECT_EQ(graph.NodeSize(), prefix_is_graph_output ? 4U : 5U);
    }
}

TEST(static_graph_pass_test, QwenGatedDeltaFusionPassRewritesLinearStateAndOutputOnX86) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildQwenGatedDeltaModelDesc()), 0);
    BindQwenGatedDeltaInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    const auto* state = graph.GetNode("next_state_update");
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->op_type, "QwenGatedDelta");
    EXPECT_EQ(state->inputs, (std::vector<std::string>{"state", "k", "v", "beta", "decay", "q"}));
    EXPECT_EQ(state->outputs, (std::vector<std::string>{"next_state", "core"}));
    EXPECT_EQ(graph.GetNode("core"), nullptr);
    EXPECT_EQ(graph.GetProducer("core"), "next_state_update");
    EXPECT_EQ(graph.GetNode("state_decay"), nullptr);
    EXPECT_EQ(graph.GetNode("output_full"), nullptr);
}

#ifdef FEATHER_WITH_CUDA
TEST(static_graph_pass_test, QwenGatedDeltaFusionPassLowersStateAndOutputToCuda) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(BuildQwenGatedDeltaModelDesc()), 0);
    BindQwenGatedDeltaInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    ASSERT_EQ(graph.NodeSize(), 1U);
    ASSERT_NE(graph.GetNode("next_state_update"), nullptr);
    EXPECT_EQ(graph.GetNode("next_state_update")->op_type, "QwenGatedDelta");

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0);
    const auto* node = runtime.GetNode("next_state_update");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->kernel_device, DeviceType::CUDA);
}
#endif

TEST(static_graph_pass_test, QwenGatedDeltaFusionPassPreservesStateAndCoreValues) {
    const auto reference = RunQwenGatedDeltaGraph(DeviceType::COMMON, false);
    const auto fused = RunQwenGatedDeltaGraph(DeviceType::X86, true);
    ASSERT_EQ(reference.first.size(), fused.first.size());
    ASSERT_EQ(reference.second.size(), fused.second.size());
    ASSERT_FALSE(reference.first.empty());
    ASSERT_FALSE(reference.second.empty());
    for (size_t index = 0; index < reference.first.size(); ++index) {
        EXPECT_NEAR(fused.first[index], reference.first[index], 1e-5f) << "state index=" << index;
    }
    for (size_t index = 0; index < reference.second.size(); ++index) {
        EXPECT_NEAR(fused.second[index], reference.second[index], 1e-5f) << "core index=" << index;
    }
}

TEST(static_graph_pass_test, QwenGatedDeltaFusionPassSkipsCommonDevice) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(BuildQwenGatedDeltaModelDesc()), 0);
    BindQwenGatedDeltaInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 15U);
    EXPECT_EQ(graph.GetNode("next_state_update")->op_type, "Add");
    EXPECT_EQ(graph.GetNode("core")->op_type, "ReduceSum");
}

TEST(static_graph_pass_test, QwenGatedDeltaFusionPassRequiresForwardSubtractionOrder) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildQwenGatedDeltaReversedSubModelDesc()), 0);
    BindQwenGatedDeltaInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 15U);
    EXPECT_EQ(graph.GetNode("next_state_update")->op_type, "Add");
    EXPECT_EQ(graph.GetNode("core")->op_type, "ReduceSum");
}

TEST(static_graph_pass_test, QwenGatedDeltaFusionPassSkipsGraphOutputIntermediate) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::X86);
    ASSERT_EQ(graph.SetModel(BuildQwenGatedDeltaIntermediateOutputModelDesc()), 0);
    BindQwenGatedDeltaInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 15U);
    EXPECT_EQ(graph.GetNode("next_state_update")->op_type, "Add");
    EXPECT_EQ(graph.GetNode("core")->op_type, "ReduceSum");
}

TEST(static_graph_pass_test, DefaultPassManagerIncludesGeneralFusionPasses) {
    auto pass_manager = feather::CreateDefaultPassManager();
    ASSERT_NE(pass_manager, nullptr);
    EXPECT_EQ(pass_manager->PassCount(), 12U);
}

TEST(static_graph_pass_test, YoloPassManagerAddsDecodeFusionOnTopOfDefaultPasses) {
    auto pass_manager = feather::CreateYoloPassManager();
    ASSERT_NE(pass_manager, nullptr);
    EXPECT_EQ(pass_manager->PassCount(), 15U);
}

TEST(static_graph_pass_test, ReshapeChainEliminationPassRemovesInnerReshape) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildReshapeChainModelDesc(false)), 0);
    BindNoOpGraphInputs(&graph, {2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("reshape0"), nullptr);
    const auto* reshape1 = graph.GetNode("reshape1");
    ASSERT_NE(reshape1, nullptr);
    EXPECT_EQ(reshape1->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, ReshapeChainEliminationPassPreservesGraphOutputIntermediate) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildReshapeChainModelDesc(true)), 0);
    BindNoOpGraphInputs(&graph, {2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_NE(graph.GetNode("reshape0"), nullptr);
    EXPECT_NE(graph.GetNode("reshape1"), nullptr);
}

TEST(static_graph_pass_test, NoOpEliminationPassRemovesSameShapeReshape) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildNoOpReshapeModelDesc()), 0);
    BindNoOpGraphInputs(&graph, {2, 2});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<NoOpEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("reshape0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, NoOpEliminationPassRemovesIdentityTranspose) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityTransposeModelDesc()), 0);
    BindNoOpGraphInputs(&graph, {1, 2, 3});
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<NoOpEliminationPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("transpose0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, ApplyPassesUsesDefaultPassManagerWhenNotOverridden) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildIdentityRelayModelDesc(false)), 0);
    BindIdentityRelayInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.GetNode("identity0"), nullptr);
    const auto* relu = graph.GetNode("relu0");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->inputs, std::vector<std::string>({"input"}));
}

TEST(static_graph_pass_test, YoloDecodeFusionPassRewritesDecodeScalePattern) {
    StaticGraph graph;
    ASSERT_EQ(graph.SetModel(BuildYoloDecodePatternModelDesc()), 0);
    BindYoloDecodePatternInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<YoloDecodeFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);

    const auto* fused = graph.GetNode("reshape1");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "YoloDecode");
    EXPECT_EQ(fused->inputs,
              (std::vector<std::string>{"raw", "xy_scale", "grid", "stride", "wh_scale", "anchor"}));
    EXPECT_EQ(fused->outputs, std::vector<std::string>({"output"}));
    EXPECT_EQ(graph.GetProducer("output"), "reshape1");
    EXPECT_EQ(graph.GetUsers("raw"), std::vector<std::string>({"reshape1"}));
    EXPECT_EQ(graph.GetNode("concat0"), nullptr);
    EXPECT_EQ(graph.GetNode("transpose0"), nullptr);
}

#ifdef FEATHER_WITH_CUDA
TEST(static_graph_pass_test, ResizeConcatFusionPassRewritesResizeConcatPatternOnCuda) {
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(BuildResizeConcatPatternModelDesc()), 0);
    BindResizeConcatPatternInputs(&graph);
    ASSERT_EQ(graph.Build(), 0);

    auto pass_manager = std::make_shared<PassManager>();
    pass_manager->AddPass(std::make_unique<ResizeConcatFusionPass>());
    graph.SetPassManager(pass_manager);

    ASSERT_EQ(graph.ApplyPasses(), 0);
    EXPECT_EQ(graph.NodeSize(), 1U);
    EXPECT_EQ(graph.OperatorSize(), 1U);
    EXPECT_EQ(graph.GetNode("resize0"), nullptr);

    const auto* fused = graph.GetNode("concat0");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "ResizeConcat");
    EXPECT_EQ(fused->inputs, (std::vector<std::string>{"resize_input", "skip_input"}));
    EXPECT_EQ(fused->outputs, std::vector<std::string>({"output"}));
    EXPECT_EQ(graph.GetProducer("output"), "concat0");
    EXPECT_EQ(graph.GetUsers("resize_out"), std::vector<std::string>{});
}
#endif
