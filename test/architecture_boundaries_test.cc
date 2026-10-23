// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include "core/execution_plan.h"
#include "core/graph_lowering.h"

namespace feather {
namespace {
std::shared_ptr<Tensor> Data(float value = 1) {
    auto result = std::make_shared<Tensor>();
    result->Assign<float>({value, value, value, value}, {2, 2});
    return result;
}
std::shared_ptr<Tensor> Shape(int64_t a = 2, int64_t b = 2) {
    auto result = std::make_shared<Tensor>();
    result->Assign<int64_t>({a, b}, {2});
    return result;
}
void BuildReshape(StaticGraph& graph) {
    model::ModelDesc model;
    model.graph.inputs = {"x", "shape"};
    model.graph.outputs = {"out"};
    for (const auto& name : {"x", "shape", "out"}) {
        model::ValueDesc value;
        value.tensor.name = name;
        value.tensor.dims = std::string(name) == "shape" ? std::vector<int64_t>{2} : std::vector<int64_t>{2, 2};
        value.tensor.data_type = std::string(name) == "shape" ? DataType::INT64 : DataType::FP32;
        model.graph.values.push_back(value);
    }
    model::NodeDesc node;
    node.name = "reshape";
    node.op_type = "Reshape";
    node.inputs = {"x", "shape"};
    node.outputs = {"out"};
    model.graph.nodes.push_back(node);
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Data()), 0);
    ASSERT_EQ(graph.SetTensor("shape", Shape()), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
}
TEST(architecture_boundaries_test, LegacyLoweringRejectsControlMutationBeforeExecution) {
    StaticGraph original;
    BuildReshape(original);
    auto model = original.model();
    model::ValueDesc value;
    value.tensor.name = "probe";
    value.tensor.dims = {2, 2};
    value.tensor.data_type = DataType::FP32;
    model.graph.values.push_back(value);
    model.graph.outputs.push_back("probe");
    model::NodeDesc earlier;
    earlier.name = "earlier";
    earlier.op_type = "ReLU";
    earlier.inputs = {"x"};
    earlier.outputs = {"probe"};
    model.graph.nodes.insert(model.graph.nodes.begin(), earlier);
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Data()), 0);
    ASSERT_EQ(graph.SetTensor("shape", Shape()), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    RuntimeGraph runtime;
    GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    ASSERT_TRUE(runtime.RunStatus().ok());
    ASSERT_FLOAT_EQ(runtime.GetTensor("probe")->data<float>()[0], 1);
    graph.GetTensor("x")->Assign<float>({7, 7, 7, 7}, {2, 2});
    graph.GetTensor("shape")->Assign<int64_t>({1, 4}, {2});
    EXPECT_FALSE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(runtime.GetTensor("probe")->data<float>()[0], 1);
}
TEST(architecture_boundaries_test, LegacyLoweringRejectsControlsChangedSinceBuild) {
    StaticGraph graph;
    BuildReshape(graph);
    graph.GetTensor("shape")->Assign<int64_t>({1, 4}, {2});
    RuntimeGraph runtime;
    GraphLowering lowering;
    EXPECT_FALSE(lowering.LowerStatus(graph, &runtime).ok());
}
TEST(architecture_boundaries_test, LegacyLoweringRetainsBindingsAndChecksDescriptors) {
    StaticGraph graph;
    BuildReshape(graph);
    GraphLowering lowering;
    RuntimeGraph first, second;
    ASSERT_TRUE(lowering.LowerStatus(graph, &first).ok());
    ASSERT_TRUE(lowering.LowerStatus(graph, &second).ok());
    EXPECT_EQ(first.GetTensor("x"), graph.GetTensor("x"));
    graph.GetTensor("x")->Assign<float>({3, 3, 3, 3}, {2, 2});
    ASSERT_TRUE(first.RunStatus().ok());
    ASSERT_TRUE(second.RunStatus().ok());
    EXPECT_FLOAT_EQ(second.GetTensor("out")->data<float>()[0], 3);
    graph.GetTensor("x")->Resize(std::vector<int64_t>{4});
    EXPECT_FALSE(first.RunStatus().ok());
    EXPECT_FALSE(second.RunStatus().ok());
}
}  // namespace
}  // namespace feather

#include "core/operator_registry.h"
namespace feather {
namespace {
void RegisterRelay(const std::string& type, const std::string& builtin, OperatorSemantics semantics) {
    RegisterOperatorSemantics(type, std::move(semantics));
    OperatorRegistry::instance().Register(type,
                                          [builtin](const model::NodeDesc& desc, OperatorRegistry::TensorMap& tensors,
                                                    const OperatorRegistry::BuildContext& context) {
                                              auto node = desc;
                                              node.op_type = builtin;
                                              return OperatorRegistry::instance().Create(node, tensors, context);
                                          });
}
TEST(architecture_boundaries_test, CustomShapeSemanticsAreFrozenAndEnforcedByBothEntrypoints) {
    OperatorSemantics semantics;
    semantics.shape_value_inputs = {1};
    RegisterRelay("DeclaredReshape", "Reshape", semantics);
    StaticGraph original;
    BuildReshape(original);
    auto model = original.model();
    model.graph.nodes[0].op_type = "DeclaredReshape";
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    graph.SetTensor("x", Data());
    graph.SetTensor("shape", Shape());
    ASSERT_TRUE(graph.BuildStatus().ok());
    RegisterOperatorSemantics("DeclaredReshape", {});  // Existing nodes retain their declaration.
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session;
    ASSERT_TRUE(compiled->CreateSession(&session).ok());
    EXPECT_FALSE(session->BindInput("shape", Shape(1, 4)).ok());
    RuntimeGraph runtime;
    GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    graph.GetTensor("shape")->Assign<int64_t>({1, 4}, {2});
    EXPECT_FALSE(runtime.RunStatus().ok());
}
TEST(architecture_boundaries_test, CustomEffectsRejectParallelAndInternallyProducedState) {
    OperatorSemantics semantics;
    semantics.serial_execution = true;
    semantics.mutable_inputs = {0};
    semantics.external_inputs = {0};
    semantics.exclusive_inputs = {0};
    RegisterRelay("DeclaredStateful", "ReLU", semantics);
    model::ModelDesc model;
    model.graph.inputs = {"x"};
    model.graph.outputs = {"out"};
    for (auto name : {"x", "mid", "out"}) {
        model::ValueDesc v;
        v.tensor.name = name;
        v.tensor.dims = {2, 2};
        v.tensor.data_type = DataType::FP32;
        model.graph.values.push_back(v);
    }
    model::NodeDesc state;
    state.name = "state";
    state.op_type = "DeclaredStateful";
    state.inputs = {"x"};
    state.outputs = {"out"};
    model.graph.nodes = {state};
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    graph.SetModel(model);
    graph.SetTensor("x", Data());
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session;
    SessionOptions options;
    options.thread_mode = RuntimeThreadMode::kParallelGraph;
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
    RuntimeGraph runtime;
    runtime.SetThreadMode(RuntimeThreadMode::kParallelGraph);
    GraphLowering lowering;
    EXPECT_FALSE(lowering.LowerStatus(graph, &runtime).ok());
    model::NodeDesc producer;
    producer.name = "producer";
    producer.op_type = "ReLU";
    producer.inputs = {"x"};
    producer.outputs = {"mid"};
    state.inputs = {"mid"};
    model.graph.nodes = {producer, state};
    StaticGraph invalid;
    invalid.SetKernelDevice(DeviceType::COMMON);
    invalid.SetModel(model);
    invalid.SetTensor("x", Data());
    ASSERT_TRUE(invalid.BuildStatus().ok());
    EXPECT_FALSE(CompiledGraph::Compile(invalid, &compiled).ok());
    EXPECT_FALSE(lowering.LowerStatus(invalid, &runtime).ok());
}
TEST(architecture_boundaries_test, WholeTensorRelayDeclarationsExcludeViews) {
    EXPECT_EQ(WholeTensorAliasSource(GetOperatorSemantics("Identity"), {"x"}, {"y"}, "y"), "x");
    EXPECT_TRUE(WholeTensorAliasSource(GetOperatorSemantics("Reshape"), {"x", "shape"}, {"y"}, "y").empty());
}
TEST(architecture_boundaries_test, AbsorbedReplacementInstallsItsDeclaredSemantics) {
    OperatorSemantics semantics;
    semantics.serial_execution = true;
    RegisterRelay("DeclaredAbsorbedSplit", "Split", semantics);
    model::ModelDesc model;
    model.graph.inputs = {"x"};
    model.graph.outputs = {"a", "b"};
    for (const auto* name : {"x", "a", "b"}) {
        model::ValueDesc value;
        value.tensor.name = name;
        value.tensor.dims = {2, 2};
        value.tensor.data_type = DataType::FP32;
        model.graph.values.push_back(value);
    }
    model::NodeDesc first;
    first.name = "first";
    first.op_type = "ReLU";
    first.inputs = {"x"};
    first.outputs = {"a"};
    auto second = first;
    second.name = "second";
    second.outputs = {"b"};
    model.graph.nodes = {first, second};
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Data()), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    auto replacement = first;
    replacement.op_type = "DeclaredAbsorbedSplit";
    replacement.outputs = {"a", "b"};
    replacement.attributes["axis"] = int64_t{0};
    replacement.attributes["split_sizes"] = std::vector<int64_t>{1, 1};
    ASSERT_TRUE(graph.ReplaceNodeDescAndAbsorbNode(replacement, "second"));
    ASSERT_NE(graph.GetNode("first"), nullptr);
    EXPECT_TRUE(graph.GetNode("first")->semantics.serial_execution);
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session;
    SessionOptions options;
    options.thread_mode = RuntimeThreadMode::kParallelGraph;
    EXPECT_FALSE(compiled->CreateSession(&session, options).ok());
}
TEST(architecture_boundaries_test, RuntimePreflightRejectsNewExternalStateAliasBeforeAnyNodeRuns) {
    OperatorSemantics semantics;
    semantics.mutable_inputs = {0};
    semantics.external_inputs = {0};
    semantics.exclusive_inputs = {0};
    RegisterRelay("DeclaredLateAliasState", "ReLU", semantics);
    model::ModelDesc model;
    model.graph.inputs = {"x", "state", "other"};
    model.graph.outputs = {"mid", "out"};
    for (const auto* name : {"x", "state", "other", "mid", "out"}) {
        model::ValueDesc value;
        value.tensor.name = name;
        value.tensor.dims = {2, 2};
        value.tensor.data_type = DataType::FP32;
        model.graph.values.push_back(value);
    }
    model::NodeDesc earlier;
    earlier.name = "earlier";
    earlier.op_type = "ReLU";
    earlier.inputs = {"x"};
    earlier.outputs = {"mid"};
    model::NodeDesc stateful;
    stateful.name = "stateful";
    stateful.op_type = "DeclaredLateAliasState";
    stateful.inputs = {"state"};
    stateful.outputs = {"out"};
    model.graph.nodes = {earlier, stateful};
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Data(1)), 0);
    ASSERT_EQ(graph.SetTensor("state", Data(2)), 0);
    ASSERT_EQ(graph.SetTensor("other", Data(3)), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    RuntimeGraph runtime;
    runtime.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    ASSERT_TRUE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(runtime.GetTensor("mid")->data<float>()[0], 1);
    graph.GetTensor("x")->Assign<float>({7, 7, 7, 7}, {2, 2});
    graph.GetTensor("state")->ShareDataWith(*graph.GetTensor("other"));
    EXPECT_FALSE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(runtime.GetTensor("mid")->data<float>()[0], 1);
}
}  // namespace
}  // namespace feather
