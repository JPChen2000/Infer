// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "core/execution_plan.h"
#include "core/graph_lowering.h"
#include "core/stateful_session.h"

namespace feather { namespace {
std::shared_ptr<Tensor> Values(std::vector<float> values, std::vector<int64_t> dims = {2, 2}) {
    auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>(values, dims); return tensor;
}
void BuildMatMul(StaticGraph& graph) {
    model::ModelDesc model; model.graph.inputs = {"a", "b"}; model.graph.outputs = {"out"};
    for (const auto* name : {"a", "b", "out"}) {
        model::ValueDesc value; value.tensor.name = name; value.tensor.dims = {2, 2};
        value.tensor.data_type = DataType::FP32; model.graph.values.push_back(value);
    }
    model::NodeDesc node; node.name = "mm"; node.op_type = "MatMul";
    node.inputs = {"a", "b"}; node.outputs = {"out"}; model.graph.nodes = {node};
    graph.SetKernelDevice(DeviceType::COMMON); ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("a", Values({1, 2, 3, 4})), 0);
    ASSERT_EQ(graph.SetTensor("b", Values({5, 6, 7, 8})), 0);
    ASSERT_TRUE(graph.BuildStatus().ok());
}

TEST(RuntimeBindingContract, SchedulingChangesPreservePreviouslyInstalledTensorHandles) {
    for (bool change_mode : {false, true}) {
        StaticGraph graph; BuildMatMul(graph); RuntimeGraph runtime;
        runtime.SetThreadMode(RuntimeThreadMode::kSerialGraph);
        GraphLowering lowering; ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
        auto original = runtime.GetTensor("a");
        if (change_mode) runtime.SetThreadMode(RuntimeThreadMode::kParallelGraph);
        else runtime.SetThreadCount(runtime.ThreadCount() + 1);
        ASSERT_EQ(runtime.SetTensor("a", Values({2, 2, 2, 2})), 0);
        EXPECT_EQ(runtime.GetTensor("a"), original);
        ASSERT_TRUE(runtime.FinalizeStatus().ok());
        ASSERT_TRUE(runtime.RunStatus().ok());
        EXPECT_FLOAT_EQ(runtime.GetTensor("out")->data<float>()[0], 24);
    }
}

TEST(RuntimeBindingContract, CompiledOutputCannotAcquireUnsupportedInputAlias) {
    StaticGraph graph; BuildMatMul(graph);
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    auto input = session->runtime().GetTensor("a");
    session->runtime().GetTensor("out")->ShareDataWith(*input);
    EXPECT_FALSE(session->runtime().CheckStatus().ok());
    EXPECT_FALSE(session->Run().ok());
    EXPECT_EQ((std::vector<float>(input->data<float>(), input->data<float>() + 4)),
              (std::vector<float>{1, 2, 3, 4}));
}

TEST(RuntimeBindingContract, LegacyRefinalizationCannotBlessChangedStorageView) {
    StaticGraph graph; BuildMatMul(graph); RuntimeGraph runtime; GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    auto input = runtime.GetTensor("a");
    runtime.GetTensor("out")->ResetStorage(input->storage(), input->memory_size(), input->offset());
    runtime.SetThreadCount(runtime.ThreadCount() + 1);
    EXPECT_FALSE(runtime.FinalizeStatus().ok());
    EXPECT_FALSE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(input->data<float>()[0], 1);
}

TEST(RuntimeBindingContract, AddingNodeCannotReplaceAnExistingParameterBinding) {
    StaticGraph graph; BuildMatMul(graph); RuntimeGraph runtime; GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    auto original = runtime.GetTensor("a");
    RuntimeNode node; node.name = "schedule_extension";
    runtime.AddNode(std::move(node));
    ASSERT_EQ(runtime.SetTensor("a", Values({2, 2, 2, 2})), 0);
    EXPECT_EQ(runtime.GetTensor("a"), original);
    EXPECT_FLOAT_EQ(original->data<float>()[0], 2);
    EXPECT_EQ(runtime.SetTensor("out", Values({9, 9, 9, 9})), -1);
}

TEST(RuntimeBindingContract, ValidatedStateSwapsUpdateOnlyTheirOwnViewContracts) {
    for (bool compiled_session : {false, true}) {
        StaticGraph graph; BuildMatMul(graph);
        RuntimeGraph legacy;
        std::unique_ptr<ExecutionSession> session;
        if (compiled_session) {
            std::shared_ptr<const CompiledGraph> compiled;
            ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
            SessionOptions options; options.reuse_memory = false;
            ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
        } else {
            GraphLowering lowering; ASSERT_TRUE(lowering.LowerStatus(graph, &legacy).ok());
        }
        auto& runtime = compiled_session ? session->runtime() : legacy;
        StatefulSession state(runtime);
        ASSERT_TRUE(state.Bind({"a", "out"}).ok());
        ASSERT_TRUE(runtime.RunStatus().ok());
        const auto old_storage = runtime.GetTensor("a")->storage();
        ASSERT_TRUE(state.Advance().ok());
        EXPECT_NE(runtime.GetTensor("a")->storage(), old_storage);
        ASSERT_TRUE(runtime.CheckStatus().ok());
        ASSERT_TRUE(runtime.RunStatus().ok());
        EXPECT_FLOAT_EQ(runtime.GetTensor("out")->data<float>()[0], 249);
        ASSERT_TRUE(state.Advance().ok());
        EXPECT_TRUE(runtime.CheckStatus().ok());
    }
}

TEST(RuntimeBindingContract, StateAdvanceCannotBlessUntrackedStorageReplacement) {
    StaticGraph graph; BuildMatMul(graph); RuntimeGraph runtime; GraphLowering lowering;
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    StatefulSession state(runtime); ASSERT_TRUE(state.Bind({"a", "out"}).ok());
    ASSERT_TRUE(runtime.RunStatus().ok());
    auto original_input = runtime.GetTensor("a")->storage();
    runtime.GetTensor("out")->ShareDataWith(*runtime.GetTensor("b"));
    EXPECT_FALSE(state.Advance().ok());
    EXPECT_EQ(runtime.GetTensor("a")->storage(), original_input);
    EXPECT_FLOAT_EQ(runtime.GetTensor("a")->data<float>()[0], 1);
}

TEST(RuntimeBindingContract, InputBindingAndStateResetRejectChangedViewBeforeWriting) {
    StaticGraph graph; BuildMatMul(graph);
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    auto input = session->runtime().GetTensor("a");
    session->runtime().GetTensor("out")->ShareDataWith(*input);
    EXPECT_FALSE(session->BindInput("a", Values({7, 7, 7, 7})).ok());
    EXPECT_FLOAT_EQ(input->data<float>()[0], 1);
    StatefulSession state(session->runtime());
    ASSERT_TRUE(state.Bind({"a", "out"}).ok());
    EXPECT_FALSE(state.Reset().ok());
    EXPECT_FLOAT_EQ(input->data<float>()[0], 1);
}

TEST(RuntimeBindingContract, ImmutableStateSourcesPreserveConstantsAndStorageViews) {
    for (bool compiled_session : {false, true}) {
        for (bool resolve_identity : {false, true}) {
            SCOPED_TRACE(testing::Message() << "compiled=" << compiled_session << " identity=" << resolve_identity);
            model::ModelDesc model; model.graph.inputs = {"state"}; model.graph.outputs = {"out"};
            for (const auto* name : {"state", "constant", "out"}) {
                model::ValueDesc value; value.tensor.name = name; value.tensor.dims = {2, 2};
                value.tensor.data_type = DataType::FP32; value.constant = value.tensor.name == "constant";
                model.graph.values.push_back(value);
            }
            model::NodeDesc identity; identity.name = "alias"; identity.op_type = "Identity";
            identity.inputs = {"constant"}; identity.outputs = {"out"}; model.graph.nodes = {identity};
            StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
            ASSERT_EQ(graph.SetModel(model), 0);
            ASSERT_EQ(graph.SetTensor("state", Values({1, 2, 3, 4})), 0);
            ASSERT_EQ(graph.SetTensor("constant", Values({5, 6, 7, 8})), 0);
            ASSERT_TRUE(graph.BuildStatus().ok());
            RuntimeGraph legacy; std::unique_ptr<ExecutionSession> session;
            if (compiled_session) {
                std::shared_ptr<const CompiledGraph> compiled;
                ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
                SessionOptions options; options.reuse_memory = false;
                ASSERT_TRUE(compiled->CreateSession(&session, options).ok());
            } else {
                GraphLowering lowering; ASSERT_TRUE(lowering.LowerStatus(graph, &legacy).ok());
            }
            auto& runtime = compiled_session ? session->runtime() : legacy;
            auto input = runtime.GetTensor("state"), constant = runtime.GetTensor("constant");
            auto output = runtime.GetTensor("out");
            ASSERT_TRUE(constant->is_immutable());
            const auto input_storage = input->storage(), constant_storage = constant->storage();
            const auto output_storage = output->storage();
            StatefulSession state(runtime);
            ASSERT_TRUE(state.Bind({"state", resolve_identity ? "out" : "constant"}).ok());
            for (int step = 0; step < 2; ++step) {
                ASSERT_TRUE(runtime.RunStatus().ok());
                const auto constant_version = constant->storage()->version();
                ASSERT_TRUE(state.Advance().ok());
                EXPECT_EQ(input->storage(), input_storage);
                EXPECT_EQ(constant->storage(), constant_storage);
                EXPECT_EQ(output->storage(), output_storage);
                EXPECT_EQ(constant->storage()->version(), constant_version);
                EXPECT_EQ((std::vector<float>(input->data<float>(), input->data<float>() + 4)),
                          (std::vector<float>{5, 6, 7, 8}));
                EXPECT_EQ((std::vector<float>(constant->data<float>(), constant->data<float>() + 4)),
                          (std::vector<float>{5, 6, 7, 8}));
                EXPECT_TRUE(runtime.CheckStatus().ok());
            }
        }
    }
}

TEST(RuntimeBindingContract, UnchangedStorageIdentityAllowsTrackedReadOnlyCopyOnWrite) {
    StaticGraph graph; BuildMatMul(graph); RuntimeGraph runtime; GraphLowering lowering;
    auto tensor = graph.GetTensor("a");
    auto owner = Values({1, 2, 3, 4});
    tensor->ResetBuffer(std::make_shared<Buffer>(owner->raw_data(), 16, owner, true), 16);
    ASSERT_TRUE(lowering.LowerStatus(graph, &runtime).ok());
    const auto id = tensor->storage()->id(); const auto* pointer = tensor->raw_data();
    tensor->mutable_data<float>()[0] = 2;
    EXPECT_EQ(tensor->storage()->id(), id); EXPECT_NE(tensor->raw_data(), pointer);
    ASSERT_TRUE(runtime.RunStatus().ok());
    EXPECT_FLOAT_EQ(runtime.GetTensor("out")->data<float>()[0], 24);
}
} }
