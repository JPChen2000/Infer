// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <future>
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

#include "core/execution_plan.h"
#include "core/graph_lowering.h"
#include "core/operator_registry.h"
#include "core/stateful_session.h"
#include "src/operator/params.h"

namespace feather {
namespace {
std::shared_ptr<Tensor> Filled(float value, const std::vector<int64_t>& shape = {128}) {
    size_t count = 0; CheckedElementCount(shape, &count);
    auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>(std::vector<float>(count, value), shape);
    return tensor;
}

model::ModelDesc Chain() {
    model::ModelDesc model;
    model.graph.inputs = {"x"}; model.graph.outputs = {"out"};
    for (const auto* name : {"x", "b", "a0", "a1", "a2", "a3", "out"}) {
        model::ValueDesc value; value.tensor.name = name; value.tensor.dims = {128};
        value.tensor.data_type = DataType::FP32; value.constant = value.tensor.name == "b";
        model.graph.values.push_back(value);
    }
    for (int i = 0; i < 5; ++i) {
        model::NodeDesc node; node.name = "n" + std::to_string(i); node.op_type = i % 2 ? "ReLU" : "Add";
        node.inputs = {i == 0 ? "x" : "a" + std::to_string(i - 1)};
        if (!(i % 2)) node.inputs.push_back("b");
        node.outputs = {i == 4 ? "out" : "a" + std::to_string(i)};
        model.graph.nodes.push_back(node);
    }
    return model;
}

void Build(StaticGraph& graph) {
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(Chain()), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1)), 0); ASSERT_EQ(graph.SetTensor("b", Filled(2)), 0);
    ASSERT_EQ(graph.Build(), 0);
}

std::shared_ptr<Tensor> Integers(const std::vector<int64_t>& values) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<int64_t>(values, {static_cast<int64_t>(values.size())});
    return tensor;
}

model::ModelDesc ControlModel(const std::string& op_type, const std::vector<std::string>& controls) {
    model::ModelDesc model;
    model.graph.inputs = {"x"}; model.graph.outputs = {"out"};
    model::ValueDesc data; data.tensor.name = "x"; data.tensor.dims = {2, 2};
    data.tensor.data_type = DataType::FP32; model.graph.values.push_back(data);
    data.tensor.name = "out"; model.graph.values.push_back(data);
    model::NodeDesc node; node.name = "control"; node.op_type = op_type;
    node.inputs = {"x"}; node.outputs = {"out"};
    for (const auto& name : controls) {
        model::ValueDesc control; control.tensor.name = name;
        control.tensor.dims = {op_type == "Reshape" ? 2 : 1}; control.tensor.data_type = DataType::INT64;
        model.graph.values.push_back(control); model.graph.inputs.push_back(name); node.inputs.push_back(name);
    }
    model.graph.nodes.push_back(node); return model;
}

TEST(execution_plan_test, CompiledReshapeRejectsChangedShapeValuesBeforeCopyOrExecution) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(ControlModel("Reshape", {"shape"})), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1, {2, 2})), 0);
    ASSERT_EQ(graph.SetTensor("shape", Integers({2, 2})), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    ASSERT_TRUE(session->BindInput("shape", Integers({2, 2})).ok());
    EXPECT_FALSE(session->BindInput("shape", Integers({1, 4})).ok());
    EXPECT_EQ(session->runtime().GetTensor("shape")->data<int64_t>()[0], 2);
    ASSERT_TRUE(session->BindInput("x", Filled(7, {2, 2})).ok()); ASSERT_TRUE(session->Run().ok());
    EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], 7);
    session->runtime().GetTensor("shape")->mutable_data<int64_t>()[0] = 1;
    EXPECT_FALSE(session->Run().ok());
    EXPECT_FALSE(session->runtime().RunStatus().ok());
}

TEST(execution_plan_test, CompiledSliceRejectsChangedBoundsAxesAndSteps) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(ControlModel("Slice", {"starts", "ends", "axes", "steps"})), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1, {2, 2})), 0);
    for (const auto& item : std::vector<std::pair<std::string, int64_t>>{{"starts", 0}, {"ends", 1}, {"axes", 0}, {"steps", 1}})
        ASSERT_EQ(graph.SetTensor(item.first, Integers({item.second})), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    for (const auto& item : std::vector<std::pair<std::string, int64_t>>{{"starts", 1}, {"ends", 2}, {"axes", 1}, {"steps", 2}})
        EXPECT_FALSE(session->BindInput(item.first, Integers({item.second})).ok()) << item.first;
    ASSERT_TRUE(session->Run().ok());
    EXPECT_EQ(session->runtime().GetTensor("out")->dims().data(), (std::vector<int64_t>{1, 2}));
}

TEST(execution_plan_test, CompilationRejectsControlsChangedAfterShapeInference) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(ControlModel("Reshape", {"shape"})), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1, {2, 2})), 0);
    ASSERT_EQ(graph.SetTensor("shape", Integers({2, 2})), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    graph.GetTensor("shape")->mutable_data<int64_t>()[0] = 1;
    graph.GetTensor("shape")->mutable_data<int64_t>()[1] = 4;
    std::shared_ptr<const CompiledGraph> compiled;
    EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled).ok()); EXPECT_EQ(compiled, nullptr);
}

TEST(execution_plan_test, NumericalPowExponentCanBeReboundWithoutChangingThePlan) {
    auto model = ControlModel("Pow", {"exponent"});
    for (auto& value : model.graph.values)
        if (value.tensor.name == "exponent") value.tensor.data_type = DataType::FP32;
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(2, {2, 2})), 0);
    ASSERT_EQ(graph.SetTensor("exponent", Filled(2, {1})), 0); ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    for (const float exponent : {3.f, 4.f}) {
        ASSERT_TRUE(session->BindInput("exponent", Filled(exponent, {1})).ok());
        ASSERT_TRUE(session->Run().ok());
        EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], exponent == 3 ? 8 : 16);
    }
}

model::ModelDesc DerivedShapeModel(const std::string& producer) {
    auto model = ControlModel("Reshape", {"shape"}); model.graph.inputs = {"x"};
    model::NodeDesc node; node.name = "shape_producer"; node.op_type = producer; node.outputs = {"shape"};
    if (producer == "Shape") node.inputs = {"x"};
    else {
        auto value = model.graph.values[1]; value.tensor.name = "seed_shape";
        value.tensor.dims = {2}; value.constant = true; model.graph.values.push_back(value);
        node.inputs = {"seed_shape"}; node.attributes["to"] = int64_t{7};
    }
    model.graph.nodes.insert(model.graph.nodes.begin(), node); return model;
}

TEST(execution_plan_test, DerivedShapeRejectsUpstreamValueChangesBeforeFirstExecution) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(DerivedShapeModel("Cast")), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(7, {2, 2})), 0);
    ASSERT_EQ(graph.SetTensor("seed_shape", Filled(2, {2})), 0); ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    auto* seed = session->runtime().GetTensor("seed_shape")->mutable_data<float>(); seed[0] = 1; seed[1] = 4;
    EXPECT_FALSE(session->Run().ok()); EXPECT_FALSE(session->runtime().RunStatus().ok());
    EXPECT_EQ(session->runtime().GetTensor("shape")->data<int64_t>()[0], 2);
    auto* source = graph.GetTensor("seed_shape")->mutable_data<float>(); source[0] = 1; source[1] = 4;
    compiled.reset(); EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled).ok());
}

TEST(execution_plan_test, DerivedShapeDependsOnInputDescriptorWithoutFreezingItsData) {
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(DerivedShapeModel("Shape")), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1, {2, 2})), 0); ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    for (const float value : {3.f, 7.f}) {
        ASSERT_TRUE(session->BindInput("x", Filled(value, {2, 2})).ok()); ASSERT_TRUE(session->Run().ok());
        EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], value);
    }
}

TEST(execution_plan_test, StatefulSessionCommitsCrossedReshapeViewsSimultaneously) {
    auto model = ControlModel("Reshape", {});
    model.graph.inputs = {"x", "b"}; model.graph.outputs = {"out", "next_b"};
    auto value = model.graph.values[0]; value.tensor.name = "b"; model.graph.values.push_back(value);
    value.tensor.name = "next_b"; model.graph.values.push_back(value);
    model.graph.nodes[0].inputs = {"b"}; model.graph.nodes[0].attributes["shape"] = std::vector<int64_t>{2, 2};
    auto node = model.graph.nodes[0]; node.name = "other"; node.inputs = {"x"}; node.outputs = {"next_b"};
    model.graph.nodes.push_back(node);
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1, {2, 2})), 0); ASSERT_EQ(graph.SetTensor("b", Filled(2, {2, 2})), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    for (const bool reuse : {false, true}) {
        SessionOptions options; options.reuse_memory = reuse;
        std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
        StatefulSession state(session->runtime());
        ASSERT_TRUE(state.Bind({"x", "out"}).ok()); ASSERT_TRUE(state.Bind({"b", "next_b"}).ok());
        for (int step = 0; step < 3; ++step) {
            ASSERT_TRUE(session->Run().ok()); ASSERT_TRUE(state.Advance().ok());
            EXPECT_FLOAT_EQ(session->runtime().GetTensor("x")->data<float>()[0], step % 2 ? 1 : 2);
            EXPECT_FLOAT_EQ(session->runtime().GetTensor("b")->data<float>()[0], step % 2 ? 2 : 1);
        }
    }
}

TEST(execution_plan_test, StatefulSessionValidatesAllBindingsBeforeChangingAnyState) {
    RuntimeGraph graph;
    for (const auto& name : {"a", "b"}) ASSERT_EQ(graph.SetTensor(name, Filled(1)), 0);
    for (const auto& name : {"next_a", "next_b"}) ASSERT_EQ(graph.SetTensor(name, Filled(2)), 0);
    StatefulSession state(graph); ASSERT_TRUE(state.Bind({"a", "next_a"}).ok());
    ASSERT_TRUE(state.Bind({"b", "next_b"}).ok());
    graph.GetTensor("next_b")->Resize(std::vector<int64_t>{64});
    EXPECT_FALSE(state.Advance().ok());
    EXPECT_FLOAT_EQ(graph.GetTensor("a")->data<float>()[0], 1);
    EXPECT_FLOAT_EQ(graph.GetTensor("b")->data<float>()[0], 1);
}

TEST(execution_plan_test, StatefulSessionSnapshotsCrossedAppendViewsWithDistinctStorageWrappers) {
    RuntimeGraph graph;
    auto a = std::make_shared<Tensor>(); a->Assign<float>({1, 2, 3}, {3});
    auto b = std::make_shared<Tensor>(); b->Assign<float>({4, 5, 6}, {3});
    const auto view = [](const std::shared_ptr<Tensor>& source, size_t offset) {
        auto tensor = std::make_shared<Tensor>(); tensor->Resize(std::vector<int64_t>{1});
        tensor->set_data_type(DataType::FP32);
        tensor->ResetBuffer(std::make_shared<Buffer>(source->mutable_data<float>() + offset,
            sizeof(float), source->storage()), sizeof(float));
        return tensor;
    };
    ASSERT_EQ(graph.SetTensor("a", a), 0); ASSERT_EQ(graph.SetTensor("b", b), 0);
    ASSERT_EQ(graph.SetTensor("next_a", view(b, 2)), 0); ASSERT_EQ(graph.SetTensor("next_b", view(a, 0)), 0);
    StatefulSession state(graph);
    ASSERT_TRUE(state.Bind({"a", "next_a", 0}).ok()); ASSERT_TRUE(state.Bind({"b", "next_b", 0}).ok());
    ASSERT_TRUE(state.Advance().ok());
    EXPECT_EQ((std::vector<float>(a->data<float>(), a->data<float>() + 3)), (std::vector<float>{2, 3, 6}));
    EXPECT_EQ((std::vector<float>(b->data<float>(), b->data<float>() + 3)), (std::vector<float>{5, 6, 1}));
}

TEST(execution_plan_test, SingleStateAppendSnapshotsOverlappingHostStorageWrappers) {
    for (const bool direct : {false, true}) {
        RuntimeGraph graph; auto state = std::make_shared<Tensor>(); state->Assign<float>({1, 2, 3}, {3});
        auto next = std::make_shared<Tensor>(); next->Resize(std::vector<int64_t>{1}); next->set_data_type(DataType::FP32);
        next->ResetBuffer(std::make_shared<Buffer>(state->mutable_data<float>(), sizeof(float), state->storage()), sizeof(float));
        ASSERT_EQ(graph.SetTensor("state", state), 0); ASSERT_EQ(graph.SetTensor("next", next), 0);
        StatefulSession session(graph); ASSERT_TRUE(session.Bind({"state", "next", 0}).ok());
        ASSERT_TRUE((direct ? ApplyStateBinding(graph, {"state", "next", 0}, DeviceType::COMMON) : session.Advance()).ok());
        EXPECT_EQ((std::vector<float>(state->data<float>(), state->data<float>() + 3)), (std::vector<float>{2, 3, 1}));
    }
}

TEST(execution_plan_test, StateUpdateValidatesResolvedIdentityProducerBeforeMutation) {
    for (const bool direct : {false, true}) {
        auto model = ControlModel("Identity", {});
        auto value = model.graph.values[0]; value.tensor.name = "state";
        model.graph.values.push_back(value); model.graph.inputs.push_back("state");
        StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
        ASSERT_EQ(graph.SetTensor("x", Filled(2, {2, 2})), 0);
        ASSERT_EQ(graph.SetTensor("state", Filled(1, {2, 2})), 0); ASSERT_TRUE(graph.BuildStatus().ok());
        std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
        std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
        StatefulSession state(session->runtime()); ASSERT_TRUE(state.Bind({"state", "out"}).ok());
        session->runtime().GetTensor("x")->set_layout(DataLayout::NHWC);
        EXPECT_FALSE((direct ? ApplyStateBinding(session->runtime(), {"state", "out"}, DeviceType::COMMON) : state.Advance()).ok());
        EXPECT_FLOAT_EQ(session->runtime().GetTensor("state")->data<float>()[0], 1);
        auto source = session->runtime().GetTensor("x"); source->set_layout(DataLayout::ND);
        source->ResetBuffer(std::make_shared<Buffer>(sizeof(float)), sizeof(float));
        EXPECT_FALSE((direct ? ApplyStateBinding(session->runtime(), {"state", "out"}, DeviceType::COMMON) : state.Advance()).ok());
        EXPECT_FLOAT_EQ(session->runtime().GetTensor("state")->data<float>()[0], 1);
    }
}

TEST(execution_plan_test, StaticConstructionDefersActivationAllocation) {
    StaticGraph graph; Build(graph);
    ASSERT_NE(graph.GetTensor("a0")->storage(), nullptr);
    EXPECT_FALSE(graph.GetTensor("a0")->storage()->materialized());
    EXPECT_NE(dynamic_cast<const DeferredKernel*>(graph.GetNode("n0")->op->kernel()), nullptr);
    EXPECT_TRUE(graph.GetTensor("x")->storage()->materialized());
    auto tensor = graph.GetTensor("a0");
    (void)tensor->mutable_data<float>();
    EXPECT_TRUE(tensor->storage()->materialized());
}

TEST(execution_plan_test, LoweringCanBeRepeatedWithoutConsumingStaticKernels) {
    StaticGraph graph; Build(graph);
    RuntimeGraph first, second; GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &first), 0); ASSERT_EQ(graph.Check(), 0);
    ASSERT_EQ(lowering.Lower(graph, &second), 0); ASSERT_EQ(graph.Check(), 0);
    ASSERT_EQ(first.Run(), 0); ASSERT_EQ(second.Run(), 0);
    EXPECT_FLOAT_EQ(second.GetTensor("out")->data<float>()[0], 7);
}

TEST(execution_plan_test, SessionsOutliveSourceGraphAndIsolateInputsWeightsAndOutputs) {
    std::shared_ptr<const CompiledGraph> compiled;
    std::weak_ptr<Tensor> original_input;
    {
        StaticGraph graph; Build(graph); original_input = graph.GetTensor("x");
        const auto status = CompiledGraph::Compile(graph, &compiled); ASSERT_TRUE(status.ok()) << status.message();
        graph.GetTensor("b")->mutable_data<float>()[0] = 100;
    }
    EXPECT_TRUE(original_input.expired());
    std::unique_ptr<ExecutionSession> first, second;
    ASSERT_TRUE(compiled->CreateSession(&first).ok()); ASSERT_TRUE(compiled->CreateSession(&second).ok());
    EXPECT_NE(first->runtime().GetTensor("x")->storage(), second->runtime().GetTensor("x")->storage());
    EXPECT_NE(first->runtime().GetTensor("b")->storage(), second->runtime().GetTensor("b")->storage());
    EXPECT_EQ(first->runtime().GetTensor("b")->storage()->buffer(), second->runtime().GetTensor("b")->storage()->buffer());
    ASSERT_TRUE(first->BindInput("x", Filled(10)).ok()); ASSERT_TRUE(second->BindInput("x", Filled(20)).ok());
    auto run_first = std::async(std::launch::async, [&] { return first->Run(); });
    auto run_second = std::async(std::launch::async, [&] { return second->Run(); });
    ASSERT_TRUE(run_first.get().ok()); ASSERT_TRUE(run_second.get().ok());
    EXPECT_FLOAT_EQ(first->runtime().GetTensor("out")->data<float>()[0], 16);
    EXPECT_FLOAT_EQ(second->runtime().GetTensor("out")->data<float>()[0], 26);
    first->runtime().GetTensor("b")->mutable_data<float>()[0] = 3;
    ASSERT_TRUE(first->Run().ok()); ASSERT_TRUE(second->Run().ok());
    EXPECT_FLOAT_EQ(first->runtime().GetTensor("out")->data<float>()[0], 19);
    EXPECT_FLOAT_EQ(second->runtime().GetTensor("out")->data<float>()[0], 26);
    EXPECT_FALSE(first->BindInput("out", Filled(0)).ok());
    EXPECT_FALSE(first->BindInput("x", Filled(0, {64})).ok());
    std::shared_ptr<Tensor> output;
    ASSERT_TRUE(first->ReadOutput("out", &output).ok()); EXPECT_FALSE(first->ReadOutput("a0", &output).ok());
    first.reset(); ASSERT_TRUE(second->Run().ok());
}

TEST(execution_plan_test, ArenaReusesOnlyDisjointLifetimesAndParallelSessionsKeepSeparateAllocations) {
    StaticGraph graph; Build(graph); std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    EXPECT_LT(compiled->memory_plan().arena_bytes, compiled->memory_plan().unshared_bytes);
    std::unique_ptr<ExecutionSession> planned, plain, parallel;
    SessionOptions no_reuse; no_reuse.reuse_memory = false;
    SessionOptions concurrent; concurrent.thread_mode = RuntimeThreadMode::kParallelGraph; concurrent.thread_count = 2;
    ASSERT_TRUE(compiled->CreateSession(&planned).ok()); ASSERT_TRUE(compiled->CreateSession(&plain, no_reuse).ok());
    ASSERT_TRUE(compiled->CreateSession(&parallel, concurrent).ok());
    EXPECT_THROW(planned->runtime().SetThreadMode(RuntimeThreadMode::kParallelGraph), std::logic_error);
    EXPECT_TRUE(planned->ReusesMemory()); EXPECT_FALSE(plain->ReusesMemory()); EXPECT_FALSE(parallel->ReusesMemory());
    EXPECT_EQ(planned->runtime().GetTensor("a0")->raw_data(), planned->runtime().GetTensor("a2")->raw_data());
    EXPECT_NE(planned->runtime().GetTensor("a0")->raw_data(), planned->runtime().GetTensor("a1")->raw_data());
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (auto* session : {planned.get(), plain.get(), parallel.get()}) {
            ASSERT_TRUE(session->BindInput("x", Filled(static_cast<float>(iteration))).ok());
            ASSERT_TRUE(session->Run().ok());
            EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], iteration + 6);
        }
    }
}

TEST(execution_plan_test, RebindingPreservesViewsAndPublicOutputAliases) {
    auto model = Chain(); model.graph.nodes.resize(3);
    model.graph.nodes[0].op_type = "Reshape"; model.graph.nodes[0].inputs = {"x"};
    model.graph.nodes[0].attributes["shape"] = std::vector<int64_t>{128};
    model.graph.nodes[1].op_type = "Add"; model.graph.nodes[1].inputs = {"a0", "b"};
    model.graph.nodes[2].op_type = "Identity"; model.graph.nodes[2].inputs = {"a1"}; model.graph.nodes[2].outputs = {"out"};
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1)), 0); ASSERT_EQ(graph.SetTensor("b", Filled(2)), 0);
    ASSERT_EQ(graph.Build(), 0); ASSERT_TRUE(graph.RemoveNodeWithOutputAlias("n2"));
    std::shared_ptr<const CompiledGraph> compiled; auto status = CompiledGraph::Compile(graph, &compiled);
    ASSERT_TRUE(status.ok()) << status.message();
    std::unique_ptr<ExecutionSession> session; status = compiled->CreateSession(&session); ASSERT_TRUE(status.ok()) << status.message();
    for (float input : {3.f, 9.f}) {
        ASSERT_TRUE(session->BindInput("x", Filled(input)).ok()); ASSERT_TRUE(session->Run().ok());
        EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], input + 2);
    }
}

TEST(execution_plan_test, BranchedArenaScheduleMatchesLifetimesAcrossRepeatedRuns) {
    auto model = Chain(); model.graph.outputs = {"a2", "out"};
    const std::vector<std::string> inputs = {"x", "a0", "a1", "x", "a3"};
    const std::vector<std::string> outputs = {"a0", "a1", "a2", "a3", "out"};
    for (size_t i = 0; i < model.graph.nodes.size(); ++i) {
        auto& node = model.graph.nodes[i]; node.op_type = "Add";
        node.inputs = {inputs[i], "b"}; node.outputs = {outputs[i]};
    }
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0); ASSERT_EQ(graph.SetTensor("x", Filled(1)), 0);
    ASSERT_EQ(graph.SetTensor("b", Filled(1)), 0); ASSERT_EQ(graph.Build(), 0);
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    for (bool reuse : {false, true}) {
        SessionOptions options; options.reuse_memory = reuse;
        std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
        for (float input : {1.f, 9.f, -2.f}) {
            ASSERT_TRUE(session->BindInput("x", Filled(input)).ok()); ASSERT_TRUE(session->Run().ok());
            EXPECT_FLOAT_EQ(session->runtime().GetTensor("a2")->data<float>()[0], input + 3);
            EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], input + 2);
        }
    }
}

TEST(execution_plan_test, SessionRejectsStructuralMutationAndMismatchedQuantization) {
    StaticGraph graph; Build(graph);
    QuantizationParams quantization; quantization.enabled = true; quantization.scale = .25f;
    graph.GetTensor("x")->set_quantization(quantization);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    auto input = Filled(9); input->set_quantization(quantization);
    ASSERT_TRUE(session->BindInput("x", input).ok());
    for (int field = 0; field < 8; ++field) {
        auto changed = quantization;
        switch (field) {
            case 0: changed.enabled = false; break;
            case 1: changed.scale = .5f; break;
            case 2: changed.zero_point = 1; break;
            case 3: changed.granularity = QuantizationGranularity::kPerChannel; break;
            case 4: changed.axis = 0; break;
            case 5: changed.block_size = 32; break;
            case 6: changed.scales = {.25f, .5f}; break;
            case 7: changed.zero_points = {0, 1}; break;
        }
        input->set_quantization(changed); EXPECT_FALSE(session->BindInput("x", input).ok());
    }
    EXPECT_EQ(session->runtime().SetTensor("x", Filled(0)), -1);
    EXPECT_THROW(session->runtime().AddNode(RuntimeNode{}), std::logic_error);
    EXPECT_THROW(session->runtime().Clear(), std::logic_error);
    EXPECT_THROW(session->runtime().SetOutputNames({"x"}), std::logic_error);
    EXPECT_THROW(session->runtime().GetOrCreateValueId("new"), std::logic_error);
    EXPECT_THROW(session->runtime().SetThreadCount(2), std::logic_error);
    EXPECT_FALSE(session->runtime().FinalizeStatus().ok());
    ASSERT_TRUE(session->Run().ok());
}

TEST(execution_plan_test, RuntimeRequiresSuccessfulFinalizeAfterMutation) {
    StaticGraph graph; Build(graph); RuntimeGraph runtime; GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0); ASSERT_TRUE(runtime.RunStatus().ok());
    runtime.AddNode(RuntimeNode{});
    EXPECT_FALSE(runtime.RunStatus().ok()); EXPECT_FALSE(runtime.FinalizeStatus().ok());
    EXPECT_FALSE(runtime.RunStatus().ok());
}

TEST(execution_plan_test, OmittedOptionalInputHasNoRuntimeDependency) {
    StaticGraph graph; Build(graph);
    RuntimeGraph runtime;
    for (const auto& name : {"x", "b", "a0"}) ASSERT_EQ(runtime.SetTensor(name, graph.GetTensor(name)), 0);
    RuntimeNode node; node.name = "optional_input"; node.op_type = "Add";
    node.inputs = {"x", "", "b"}; node.outputs = {"a0"};
    node.kernel = graph.GetNode("n0")->op->kernel()->Clone(); node.kernel_device = node.kernel->device();
    runtime.AddNode(std::move(node)); ASSERT_TRUE(runtime.FinalizeStatus().ok()); ASSERT_TRUE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(runtime.GetTensor("a0")->data<float>()[0], 3);
}

TEST(execution_plan_test, SessionRejectsDescriptorChangesThroughMutableTensorHandles) {
    StaticGraph graph; Build(graph); std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    auto tensor = session->runtime().GetTensor("x");
    tensor->Resize(std::vector<int64_t>{64}); EXPECT_FALSE(session->Run().ok());
    EXPECT_FALSE(session->runtime().RunStatus().ok());
    tensor->Resize(std::vector<int64_t>{128}); ASSERT_TRUE(session->Run().ok());
    tensor->set_layout(DataLayout::NHWC); EXPECT_FALSE(session->Run().ok());
    tensor->set_layout(DataLayout::ND); ASSERT_TRUE(session->Run().ok());
    auto output = session->runtime().GetTensor("out");
    output->set_data_type(DataType::INT8); EXPECT_FALSE(session->Run().ok());
    output->set_data_type(DataType::FP32); ASSERT_TRUE(session->Run().ok());
}

TEST(execution_plan_test, HostMemoryBudgetIsCheckedBeforeSessionAllocation) {
    StaticGraph graph; Build(graph); std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> planned, plain, limited;
    ASSERT_TRUE(compiled->CreateSession(&planned).ok());
    SessionOptions options; options.reuse_memory = false;
    ASSERT_TRUE(compiled->CreateSession(&plain, options).ok());
    EXPECT_LT(planned->memory_usage().mutable_host_bytes, plain->memory_usage().mutable_host_bytes);
    EXPECT_EQ(planned->memory_usage().shared_weight_bytes, 128 * sizeof(float));
    options.host_memory_limit = planned->memory_usage().mutable_host_bytes;
    EXPECT_FALSE(compiled->CreateSession(&limited, options).ok()); EXPECT_EQ(limited, nullptr);
    options.reuse_memory = true;
    EXPECT_TRUE(compiled->CreateSession(&limited, options).ok());
}

TEST(execution_plan_test, ConstantViewsShareInitializedReadOnlyStorage) {
    auto model = Chain(); model.graph.nodes.resize(2);
    model.graph.nodes[0].op_type = "Reshape"; model.graph.nodes[0].inputs = {"b"};
    model.graph.nodes[0].attributes["shape"] = std::vector<int64_t>{128};
    model.graph.nodes[1].op_type = "Add"; model.graph.nodes[1].inputs = {"x", "a0"}; model.graph.nodes[1].outputs = {"out"};
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("x", Filled(1)), 0); ASSERT_EQ(graph.SetTensor("b", Filled(2)), 0);
    ASSERT_EQ(graph.Build(), 0);
    std::shared_ptr<const CompiledGraph> compiled; auto status = CompiledGraph::Compile(graph, &compiled);
    ASSERT_TRUE(status.ok()) << status.message();
    std::unique_ptr<ExecutionSession> session; status = compiled->CreateSession(&session); ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(session->runtime().GetTensor("b")->storage(), session->runtime().GetTensor("a0")->storage());
    ASSERT_TRUE(session->Run().ok()); EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], 3);
    std::unique_ptr<ExecutionSession> second; ASSERT_TRUE(compiled->CreateSession(&second).ok());
    session->runtime().GetTensor("b")->mutable_data<float>()[0] = 7;
    EXPECT_EQ(session->runtime().GetTensor("b")->storage(), session->runtime().GetTensor("a0")->storage());
    EXPECT_FLOAT_EQ(session->runtime().GetTensor("a0")->data<float>()[0], 7);
    ASSERT_TRUE(session->Run().ok()); ASSERT_TRUE(second->Run().ok());
    EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], 8);
    EXPECT_FLOAT_EQ(second->runtime().GetTensor("out")->data<float>()[0], 3);
}

class TestBackend final : public BackendContext {
 public:
    explicit TestBackend(bool fail) : fail_(fail) {}
    Status ReadHost(Tensor&) override {
        ++reads;
        return fail_ ? Status::Error(StatusCode::kExecutionFailed, "test transfer unavailable") : Status::Ok();
    }
    void HostWritten(Tensor&) override { ++writes; }
    int reads{0}, writes{0};
 private:
    bool fail_;
};

class PolicyBackend final : public BackendContext {
 public:
    BackendCapabilities Capabilities(DeviceType device) const override {
        if (device == DeviceType::ASCEND) return {true, false, false, false};
        return BackendContext::Capabilities(device);
    }
    Status ReadHost(Tensor&) override { ++reads; return Status::Ok(); }
    void HostWritten(Tensor&) override { ++writes; }
    Status SwapState(Tensor&, Tensor&) override { ++swaps; return Status::Ok(); }
    Status AppendState(Tensor&, const Tensor&, int) override { ++appends; return Status::Ok(); }
    std::shared_ptr<const Storage> DeviceStorage(const Tensor& tensor) const override {
        const auto found = allocations.find(&tensor);
        return found == allocations.end() ? nullptr : found->second;
    }
    std::unordered_map<const Tensor*, std::shared_ptr<Storage>> allocations;
    int reads{0}, writes{0}, swaps{0}, appends{0};
};

TEST(execution_plan_test, SingleDeviceAppendRejectsAliasedAllocationBeforeMutation) {
    for (const bool direct : {false, true}) {
        auto backend = std::make_unique<PolicyBackend>(); auto* policy = backend.get();
        RuntimeGraph graph(std::move(backend));
        ASSERT_EQ(graph.SetTensor("state", Filled(1, {3})), 0); ASSERT_EQ(graph.SetTensor("next", Filled(2, {1})), 0);
        auto allocation = std::make_shared<Storage>(3 * sizeof(float));
        policy->allocations[graph.GetTensor("state").get()] = allocation;
        policy->allocations[graph.GetTensor("next").get()] = allocation;
        StatefulSession session(graph, DeviceType::ASCEND); ASSERT_TRUE(session.Bind({"state", "next", 0}).ok());
        EXPECT_FALSE((direct ? ApplyStateBinding(graph, {"state", "next", 0}, DeviceType::ASCEND) : session.Advance()).ok());
        EXPECT_EQ(policy->appends, 0);
    }
}

TEST(execution_plan_test, DeviceStateBatchRejectsCrossAliasesBeforeBackendMutation) {
    auto backend = std::make_unique<PolicyBackend>(); auto* policy = backend.get();
    RuntimeGraph graph(std::move(backend));
    auto a = Filled(1), b = Filled(2);
    ASSERT_EQ(graph.SetTensor("a", a), 0); ASSERT_EQ(graph.SetTensor("b", b), 0);
    ASSERT_EQ(graph.SetTensor("next_a", b), 0); ASSERT_EQ(graph.SetTensor("next_b", a), 0);
    StatefulSession state(graph, DeviceType::ASCEND);
    ASSERT_TRUE(state.Bind({"a", "next_a"}).ok()); ASSERT_TRUE(state.Bind({"b", "next_b"}).ok());
    EXPECT_FALSE(state.Advance().ok()); EXPECT_EQ(policy->swaps, 0);
    EXPECT_EQ(policy->reads, 0); EXPECT_EQ(policy->writes, 0);
}

TEST(execution_plan_test, DeviceStateBatchRejectsAliasedDeviceAllocationsWithSeparateHostBuffers) {
    auto backend = std::make_unique<PolicyBackend>(); auto* policy = backend.get();
    RuntimeGraph graph(std::move(backend));
    for (const auto& name : {"a", "b", "next_a", "next_b"}) ASSERT_EQ(graph.SetTensor(name, Filled(1)), 0);
    auto first = std::make_shared<Storage>(128 * sizeof(float));
    auto second = std::make_shared<Storage>(128 * sizeof(float));
    policy->allocations[graph.GetTensor("a").get()] = first;
    policy->allocations[graph.GetTensor("b").get()] = second;
    policy->allocations[graph.GetTensor("next_a").get()] = second;
    policy->allocations[graph.GetTensor("next_b").get()] = first;
    StatefulSession state(graph, DeviceType::ASCEND);
    ASSERT_TRUE(state.Bind({"a", "next_a"}).ok()); ASSERT_TRUE(state.Bind({"b", "next_b"}).ok());
    EXPECT_FALSE(state.Advance().ok()); EXPECT_EQ(policy->swaps, 0);
}

TEST(execution_plan_test, DeviceStatePolicyDoesNotTreatOtherAcceleratorsAsHostMemory) {
    RuntimeGraph unsupported;
    EXPECT_FALSE(unsupported.Backend().Capabilities(DeviceType::ASCEND).supported);
    auto backend = std::make_unique<PolicyBackend>(); auto* policy = backend.get();
    RuntimeGraph graph(std::move(backend));
    ASSERT_EQ(graph.SetTensor("state", Filled(0)), 0); ASSERT_EQ(graph.SetTensor("next", Filled(3)), 0);
    ASSERT_TRUE(ApplyStateBinding(graph, {"state", "next", -1}, DeviceType::ASCEND).ok());
    EXPECT_EQ(policy->swaps, 1); EXPECT_EQ(policy->reads, 0); EXPECT_EQ(policy->writes, 0);
    EXPECT_FLOAT_EQ(graph.GetTensor("state")->data<float>()[0], 0);
}

class ThrowingBackend final : public BackendContext {
 public:
    Status ReadHost(Tensor&) override { throw std::runtime_error("transfer exception"); }
};

TEST(execution_plan_test, ParallelTransferExceptionsReturnStatusWithoutStrandingWorkers) {
    StaticGraph graph; Build(graph); std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    SessionOptions options; options.thread_mode = RuntimeThreadMode::kParallelGraph; options.thread_count = 2;
    options.backend_factory = [] { return std::make_unique<ThrowingBackend>(); };
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    const auto status = session->Run(); EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("transfer exception"), std::string::npos);
    EXPECT_NE(status.message().find("node=n0"), std::string::npos);
}
TEST(execution_plan_test, BackendCanBeInjectedWithoutVendorDependenciesAndPreservesTransferErrors) {
    StaticGraph graph; Build(graph);
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    SessionOptions options; TestBackend* backend = nullptr;
    options.backend_factory = [&] { auto context = std::make_unique<TestBackend>(false); backend = context.get(); return context; };
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    ASSERT_TRUE(session->Run().ok()); EXPECT_GT(backend->reads, 0); EXPECT_GT(backend->writes, 0);
    options.backend_factory = [] { return std::make_unique<TestBackend>(true); };
    ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
    const auto status = session->Run(); EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("prepare node=n0 input=x"), std::string::npos);
    EXPECT_NE(status.message().find("test transfer unavailable"), std::string::npos);
}

class TaggedKernel : public KernelBase { public: explicit TaggedKernel(int tag) : tag(tag) {} int32_t compute() override { return 0; } int tag; };
TEST(execution_plan_test, KernelSelectionChecksConstraintsAndExplicitFallbackPolicy) {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "TestConstrained", [] { return std::make_unique<TaggedKernel>(1); });
    dispatcher.RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND, "TestConstrained", "wide",
        [] { return std::make_unique<TaggedKernel>(2); }, [](const KernelRequest& request) {
            return request.tensors.size() && request.tensors[0]->numel() >= 128 && request.workspace_limit >= 64
                ? Status::Ok() : Status::Error(StatusCode::kUnsupported, "shape/workspace constraint");
        }, 10);
    KernelRequest request{DeviceType::COMMON, DataType::FP32, DataLayout::ND, "TestConstrained", {Filled(1)}};
    auto wide = dispatcher.create(request); ASSERT_NE(wide, nullptr); EXPECT_EQ(static_cast<TaggedKernel*>(wide.get())->tag, 2);
    request.workspace_limit = 0;
    auto small = dispatcher.create(request); ASSERT_NE(small, nullptr); EXPECT_EQ(static_cast<TaggedKernel*>(small.get())->tag, 1);
    {
        KernelSelectionScope scope(KernelFallbackPolicy::kAllowCommon, nullptr, false, 0);
        auto limited = CreateKernelForTensor(DeviceType::COMMON, "TestConstrained", {Filled(1)});
        ASSERT_NE(limited, nullptr); EXPECT_EQ(static_cast<TaggedKernel*>(limited.get())->tag, 1);
    }
    request.device = DeviceType::CUDA;
    EXPECT_EQ(dispatcher.create(request, KernelFallbackPolicy::kRequireRequested), nullptr);
    EXPECT_NE(dispatcher.create(request, KernelFallbackPolicy::kAllowCommon), nullptr);
    auto pinned_factory = wide->InstanceFactory();
    dispatcher.RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND, "TestConstrained", "wide",
        [] { return std::make_unique<TaggedKernel>(3); }, {}, 10);
    EXPECT_EQ(static_cast<TaggedKernel*>(pinned_factory().get())->tag, 2);
}

TEST(execution_plan_test, WrongOwnedParameterTypeProducesNodeContextInsteadOfInvalidCast) {
    RuntimeGraph graph;
    RuntimeNode node; node.name = "broken_add"; node.op_type = "Add";
    node.kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP32, "Add");
    node.kernel->SetParamOwner(std::make_shared<operators::UnaryParam>());
    graph.AddNode(std::move(node)); ASSERT_EQ(graph.Finalize(), 0);
    auto status = graph.RunStatus(); EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("broken_add"), std::string::npos);
    EXPECT_NE(status.message().find("parameter type mismatch"), std::string::npos);
}

TEST(execution_plan_test, StatefulSessionAdvancesResetsAndRejectsIncompatibleAppend) {
    RuntimeGraph graph;
    ASSERT_EQ(graph.SetTensor("state", Filled(0, {1, 3})), 0);
    ASSERT_EQ(graph.SetTensor("next", Filled(5, {1, 3})), 0);
    ASSERT_EQ(graph.SetTensor("token", Filled(9, {1, 1})), 0);
    StatefulSession session(graph);
    ASSERT_TRUE(session.Bind({"state", "next", -1}).ok());
    ASSERT_TRUE(session.Advance().ok()); EXPECT_FLOAT_EQ(graph.GetTensor("state")->data<float>()[0], 5);
    ASSERT_TRUE(session.Reset().ok()); EXPECT_FLOAT_EQ(graph.GetTensor("state")->data<float>()[0], 0);
    ASSERT_TRUE(ApplyStateBinding(graph, {"state", "token", 1}, DeviceType::COMMON).ok());
    EXPECT_FLOAT_EQ(graph.GetTensor("state")->data<float>()[2], 9);
    EXPECT_FALSE(ApplyStateBinding(graph, {"state", "token", 0}, DeviceType::COMMON).ok());
}

TEST(execution_plan_test, StateBindingsAcceptEquivalentNdAndNchwLayouts) {
    for (const auto input_layout : {DataLayout::ND, DataLayout::NCHW}) {
        const auto output_layout = input_layout == DataLayout::ND ? DataLayout::NCHW : DataLayout::ND;
        RuntimeGraph graph;
        auto state = Filled(0, {1, 2, 3, 4});
        auto next = Filled(5, {1, 2, 3, 4});
        auto token = Filled(9, {1, 2, 1, 4});
        state->set_layout(input_layout);
        next->set_layout(output_layout);
        token->set_layout(output_layout);
        ASSERT_EQ(graph.SetTensor("state", state), 0);
        ASSERT_EQ(graph.SetTensor("next", next), 0);
        ASSERT_EQ(graph.SetTensor("token", token), 0);
        StatefulSession session(graph);
        const auto status = session.Bind({"state", "next", -1});
        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_TRUE(session.Advance().ok());
        for (int i = 0; i < 24; ++i) EXPECT_FLOAT_EQ(state->data<float>()[i], 5);
        ASSERT_TRUE(ApplyStateBinding(graph, {"state", "token", 2}, DeviceType::COMMON).ok());
        for (int head = 0; head < 2; ++head) {
            for (int i = 0; i < 12; ++i)
                EXPECT_FLOAT_EQ(state->data<float>()[head * 12 + i], i < 8 ? 5 : 9);
        }
        EXPECT_EQ(state->layout(), input_layout);
        EXPECT_EQ(next->layout(), output_layout);
    }
}

TEST(execution_plan_test, StateBindingsRejectChannelLastLayoutMismatch) {
    for (const auto layout : {DataLayout::ND, DataLayout::NCHW}) {
        for (const bool channel_last_input : {false, true}) {
            RuntimeGraph graph;
            auto state = Filled(0, {1, 2, 3, 4});
            auto next = Filled(5, {1, 2, 3, 4});
            auto token = Filled(9, {1, 2, 1, 4});
            state->set_layout(channel_last_input ? DataLayout::NHWC : layout);
            next->set_layout(channel_last_input ? layout : DataLayout::NHWC);
            token->set_layout(next->layout());
            ASSERT_EQ(graph.SetTensor("state", state), 0);
            ASSERT_EQ(graph.SetTensor("next", next), 0);
            ASSERT_EQ(graph.SetTensor("token", token), 0);
            StatefulSession session(graph);
            EXPECT_FALSE(session.Bind({"state", "next", -1}).ok());
            EXPECT_FALSE(ApplyStateBinding(graph, {"state", "token", 2}, DeviceType::COMMON).ok());
        }
    }
}

class DependentPass : public GraphPass {
 public:
    const std::string& name() const override { static const std::string name = "dependent"; return name; }
    std::vector<std::string> Requires() const override { return {"missing"}; }
    int32_t Run(StaticGraph*) override { return 0; }
};
class CyclicPass : public GraphPass {
 public:
    const std::string& name() const override { static const std::string name = "cyclic"; return name; }
    int32_t Run(StaticGraph* graph) override { return graph->ReplaceInputValue("n0", "x", "a0") ? 0 : -1; }
};
TEST(execution_plan_test, PassPipelineChecksPrerequisitesAndValidatesResultingDag) {
    StaticGraph graph; Build(graph);
    auto passes = std::make_shared<PassManager>(); passes->AddPass(std::make_unique<DependentPass>());
    graph.SetPassManager(passes); auto status = graph.ApplyPassesStatus(); EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("requires prior pass"), std::string::npos);
    passes = std::make_shared<PassManager>(); passes->AddPass(std::make_unique<CyclicPass>());
    graph.SetPassManager(passes); status = graph.ApplyPassesStatus(); EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("after pass=cyclic"), std::string::npos);
    std::shared_ptr<const CompiledGraph> compiled; EXPECT_FALSE(CompiledGraph::Compile(graph, &compiled).ok());
}

TEST(execution_plan_test, GeneralAndModelSpecificPassPipelinesAreSeparate) {
    EXPECT_EQ(CreateDefaultPassManager()->PassCount(), 6);
    EXPECT_EQ(CreateQwenPassManager()->PassCount(), 12);
    EXPECT_EQ(CreateYoloPassManager()->PassCount(), 9);
}
#ifdef FEATHER_WITH_CUDA
TEST(execution_plan_test, WorkspaceBudgetRejectsBeforeDeviceAllocationAndTracksRoundedCapacity) {
    RuntimeGraph graph; kernel::cuda_detail::ExecutionContextScope scope(graph.CudaContext());
    graph.Backend().SetWorkspaceLimit(0);
    void* memory = nullptr;
    EXPECT_EQ(kernel::cuda_detail::AcquireTemporaryDeviceBuffer(1, &memory), -1);
    EXPECT_EQ(memory, nullptr);
    EXPECT_NE(kernel::cuda_detail::CudaLastErrorMessage().find("workspace budget"), std::string::npos);
    graph.Backend().SetWorkspaceLimit(255);
    EXPECT_EQ(kernel::cuda_detail::AcquireTemporaryDeviceBuffer(1, &memory), -1);
    EXPECT_EQ(memory, nullptr);
}

TEST(execution_plan_test, DeviceStorageOwnsAllocationBeyondCacheLifetime) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    RuntimeGraph graph;
    kernel::cuda_detail::ExecutionContextScope scope(graph.CudaContext());
    auto tensor = Filled(4);
    void* device = nullptr;
    ASSERT_EQ(kernel::cuda_detail::AcquireTensorDevice(tensor.get(), tensor->memory_size(), tensor->raw_data(), &device), 0);
    auto storage = graph.Backend().DeviceStorage(*tensor);
    ASSERT_NE(storage, nullptr); EXPECT_EQ(storage->device(), DeviceType::CUDA);
    ASSERT_TRUE(graph.Backend().Synchronize().ok());
    graph.Backend().Clear();
    float value = 0;
    ASSERT_EQ(cudaMemcpy(&value, storage->data(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(value, 4);
}

TEST(execution_plan_test, CudaCompiledSessionsMatchExpectedResultsAndIsolateConcurrentInputs) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        (void)cudaGetLastError(); GTEST_SKIP() << "CUDA device unavailable";
    }
    StaticGraph graph; graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(Chain()), 0); ASSERT_EQ(graph.SetTensor("x", Filled(1)), 0);
    ASSERT_EQ(graph.SetTensor("b", Filled(2)), 0); ASSERT_EQ(graph.Build(), 0);
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> first, second;
    ASSERT_TRUE(compiled->CreateSession(&first).ok()); ASSERT_TRUE(compiled->CreateSession(&second).ok());
    ASSERT_FALSE(first->ReusesMemory()); ASSERT_NE(first->runtime().CudaContext(), second->runtime().CudaContext());
    for (float input : {1.f, 7.f}) {
        ASSERT_TRUE(first->BindInput("x", Filled(input)).ok()); ASSERT_TRUE(second->BindInput("x", Filled(input + 10)).ok());
        auto run = std::async(std::launch::async, [&] { return first->Run(); });
        const auto other = second->Run(); const auto status = run.get();
        ASSERT_TRUE(status.ok()) << status.message(); ASSERT_TRUE(other.ok()) << other.message();
        std::shared_ptr<Tensor> a, b;
        ASSERT_TRUE(first->ReadOutput("out", &a).ok()); ASSERT_TRUE(second->ReadOutput("out", &b).ok());
        for (size_t i = 0; i < 128; ++i) {
            EXPECT_FLOAT_EQ(a->data<float>()[i], input + 6); EXPECT_FLOAT_EQ(b->data<float>()[i], input + 16);
        }
    }
}
#endif
}  // namespace
}  // namespace feather
