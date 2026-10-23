// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "core/execution_plan.h"
#include "src/operator/params.h"

namespace feather {
namespace {
model::ModelDesc DescriptorModel(DataType output = DataType::FP32) {
    model::ModelDesc m; m.graph.inputs = {"x", "w"}; m.graph.outputs = {"out"};
    for (auto name : {"x", "w", "out"}) {
        model::ValueDesc v; v.tensor.name = name; v.tensor.dims = {1};
        v.tensor.data_type = std::string(name) == "out" ? output : DataType::FP32;
        m.graph.values.push_back(v);
    }
    model::NodeDesc n; n.name = "add"; n.op_type = "Add"; n.inputs = {"x", "w"}; n.outputs = {"out"};
    m.graph.nodes.push_back(n); return m;
}
void DescriptorInputs(StaticGraph& g, DataType output = DataType::FP32) {
    g.SetKernelDevice(DeviceType::COMMON); g.SetModel(DescriptorModel(output));
    for (auto name : {"x", "w"}) { auto t = std::make_shared<Tensor>(); t->Assign<float>({1}, {1}); g.SetTensor(name, t); }
}
TEST(build_descriptor_contract_test, RejectsDeclaredDtypeThatConflictsWithInferredKernelOutput) {
    StaticGraph g; DescriptorInputs(g, DataType::INT64);
    EXPECT_FALSE(g.BuildStatus().ok());
}
TEST(build_descriptor_contract_test, MatMulRejectsDeclaredOutputDtypeItsCandidateCannotProduce) {
    auto model = DescriptorModel(DataType::INT64); model.graph.nodes[0].op_type = "MatMul";
    for (auto& value : model.graph.values) value.tensor.dims = {1, 1};
    StaticGraph g; g.SetKernelDevice(DeviceType::COMMON); g.SetModel(model);
    for (auto name : {"x", "w"}) { auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>({1}, {1, 1}); g.SetTensor(name, tensor); }
    const auto status = g.BuildStatus();
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("MatMul output dtype"), std::string::npos);
}
class Int64OutputMatMul final : public KernelBase {
 public:
    Status Supports() const override {
        const auto* param = ParamAs<operators::MatMulParam>();
        return param && param->out && param->out->data_type() == DataType::INT64 ? Status::Ok()
            : Status::Error(StatusCode::kUnsupported, "requires INT64 output");
    }
    int32_t compute() override {
        auto* param = ParamAs<operators::MatMulParam>();
        param->out->mutable_data<int64_t>()[0] = static_cast<int64_t>(param->a->data<float>()[0] * param->b->data<float>()[0]);
        return 0;
    }
};
TEST(build_descriptor_contract_test, HeterogeneousOutputCandidateCanExplicitlySupportDeclaredDtype) {
    auto& dispatcher = KernelDispatcher::instance();
    auto factory = [] { return std::make_unique<Int64OutputMatMul>(); };
    dispatcher.RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND, "MatMul",
        "test_int64_output", factory, {}, 100);
    auto model = DescriptorModel(DataType::INT64); model.graph.nodes[0].op_type = "MatMul";
    for (auto& value : model.graph.values) value.tensor.dims = {1, 1};
    StaticGraph g; g.SetKernelDevice(DeviceType::COMMON); g.SetModel(model);
    for (auto name : {"x", "w"}) { auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>({2}, {1, 1}); g.SetTensor(name, tensor); }
    const auto build = g.BuildStatus();
    dispatcher.RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND, "MatMul",
        "test_int64_output", factory, [](const KernelRequest&) { return Status::Error(StatusCode::kUnsupported, "test candidate retired"); }, 100);
    ASSERT_TRUE(build.ok()) << build.message();
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(g, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    ASSERT_TRUE(session->Run().ok()); EXPECT_EQ(session->runtime().GetTensor("out")->data<int64_t>()[0], 4);
}
TEST(build_descriptor_contract_test, RejectsDescriptorChangesBetweenBuildAndCompile) {
    for (int change = 0; change < 4; ++change) {
        StaticGraph g; DescriptorInputs(g); ASSERT_TRUE(g.BuildStatus().ok());
        auto x = g.GetTensor("x");
        if (change == 0) x->Resize(std::vector<int64_t>{1, 1});
        if (change == 1) x->set_data_type(DataType::INT32);
        if (change == 2) x->set_layout(DataLayout::NHWC);
        if (change == 3) x->set_quantization({true, 0.25f});
        std::shared_ptr<const CompiledGraph> compiled;
        EXPECT_FALSE(CompiledGraph::Compile(g, &compiled).ok()) << change;
    }
}
TEST(build_descriptor_contract_test, PayloadAndStorageChangesPreserveBuiltDescriptor) {
    StaticGraph g; DescriptorInputs(g); ASSERT_TRUE(g.BuildStatus().ok());
    auto x = g.GetTensor("x"); x->Assign<float>({5}, {1});
    auto allocation = std::make_shared<Storage>(sizeof(float));
    x->ResetStorage(allocation, sizeof(float)); x->mutable_data<float>()[0] = 7;
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(g, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    ASSERT_TRUE(session->Run().ok()); EXPECT_FLOAT_EQ(session->runtime().GetTensor("out")->data<float>()[0], 8);
}
TEST(build_descriptor_contract_test, ReplacingBuiltTensorDoesNotRebindFrozenKernelParameters) {
    StaticGraph g; DescriptorInputs(g); ASSERT_TRUE(g.BuildStatus().ok());
    auto replacement = std::make_shared<Tensor>(); replacement->Assign<float>({9}, {1});
    ASSERT_EQ(g.SetTensor("x", replacement), 0);
    std::shared_ptr<const CompiledGraph> compiled;
    EXPECT_FALSE(CompiledGraph::Compile(g, &compiled).ok());
}
TEST(build_descriptor_contract_test, PassEntryRejectsExternalDescriptorChangesInsteadOfResnapshotting) {
    StaticGraph g; DescriptorInputs(g); ASSERT_TRUE(g.BuildStatus().ok());
    g.GetTensor("x")->set_data_type(DataType::INT32);
    EXPECT_FALSE(g.ApplyPassesStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled;
    EXPECT_FALSE(CompiledGraph::Compile(g, &compiled).ok());
}
TEST(build_descriptor_contract_test, ValidOutputAliasRewriteDoesNotBlessUnrelatedInputMutation) {
    auto model = DescriptorModel();
    auto value = model.graph.values.back(); value.tensor.name = "public"; model.graph.values.push_back(value);
    model::NodeDesc relay; relay.name = "relay"; relay.op_type = "Identity";
    relay.inputs = {"out"}; relay.outputs = {"public"}; model.graph.nodes.push_back(relay); model.graph.outputs = {"public"};
    StaticGraph g; g.SetKernelDevice(DeviceType::COMMON); g.SetModel(model);
    for (auto name : {"x", "w"}) { auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>({1}, {1}); g.SetTensor(name, tensor); }
    ASSERT_TRUE(g.BuildStatus().ok());
    g.GetTensor("x")->set_data_type(DataType::INT32);
    ASSERT_TRUE(g.RemoveNodeWithOutputAlias("relay"));
    std::shared_ptr<const CompiledGraph> compiled;
    const auto status = CompiledGraph::Compile(g, &compiled);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("descriptor changed since build: x"), std::string::npos);
}
class ReplaceWithInt64Cast final : public GraphPass {
 public:
    const std::string& name() const override { static const std::string value = "replace_with_int64_cast"; return value; }
    int32_t Run(StaticGraph* graph) override {
        if (!graph->SetValueDataType("out", DataType::INT64)) return -1;
        model::NodeDesc node; node.name = "add"; node.op_type = "Cast";
        node.inputs = {"x"}; node.outputs = {"out"}; node.attributes["to"] = int64_t{7};
        return graph->ReplaceNodeDesc(node) ? 0 : -1;
    }
};
TEST(build_descriptor_contract_test, SuccessfulPassCanExplicitlyChangeDtypeAndRebuildKernel) {
    StaticGraph g; DescriptorInputs(g); ASSERT_TRUE(g.BuildStatus().ok());
    auto passes = std::make_shared<PassManager>(); passes->AddPass(std::make_unique<ReplaceWithInt64Cast>());
    g.SetPassManager(passes); ASSERT_TRUE(g.ApplyPassesStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled; ASSERT_TRUE(CompiledGraph::Compile(g, &compiled).ok());
    std::unique_ptr<ExecutionSession> session; ASSERT_TRUE(compiled->CreateSession(&session).ok());
    ASSERT_TRUE(session->Run().ok()); EXPECT_EQ(session->runtime().GetTensor("out")->data<int64_t>()[0], 1);
}
}  // namespace
}  // namespace feather
