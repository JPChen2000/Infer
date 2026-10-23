// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "core/operator_registry.h"

namespace feather {
namespace {
struct SelectionParam {
    std::shared_ptr<Tensor> input, output;
    template <typename Visitor> void VisitTensors(Visitor visitor) { visitor(input); visitor(output); }
};

class SelectionKernel final : public KernelBase {
 public:
    SelectionKernel(int tag, StatusCode rejection = StatusCode::kOk, bool fail_compute = false)
        : tag_(tag), rejection_(rejection), fail_compute_(fail_compute) {}
    Status Supports() const override {
        const auto* p = ParamAs<SelectionParam>();
        if (!p || !p->input || !p->output)
            return Status::Error(StatusCode::kInvalidArgument, "parameters must be bound");
        if (p->input->numel() != 1)
            return Status::Error(StatusCode::kInvalidArgument, "expected scalar input");
        return rejection_ == StatusCode::kOk ? Status::Ok()
            : Status::Error(rejection_, "bound shape rejected by candidate " + std::to_string(tag_));
    }
    int32_t compute() override {
        if (!Supports().ok()) return -1;
        auto* p = ParamAs<SelectionParam>();
        p->output->mutable_data<float>()[0] = p->input->data<float>()[0] + tag_;
        return fail_compute_ ? -1 : 0;
    }
 private:
    int tag_;
    StatusCode rejection_;
    bool fail_compute_;
};

class SelectionOp final : public OpBase {
 public:
    explicit SelectionOp(SelectionParam param) : param_(std::move(param)) {
        SetInputs({param_.input}); SetOutputs({param_.output});
    }
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override {
        kernel_ = std::move(kernel);
        if (kernel_) kernel_->SetParamOwner(std::make_shared<SelectionParam>(param_));
    }
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override { return kernel_ ? kernel_->compute() : -1; }
 private:
    SelectionParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

OperatorRegistry::TensorMap SelectionTensors() {
    auto input = std::make_shared<Tensor>(); input->Assign<float>({2}, {1});
    auto output = std::make_shared<Tensor>(); output->Assign<float>({0}, {1});
    return {{"input", input}, {"output", output}};
}

void RegisterSelectionOp(const std::string& name) {
    OperatorRegistry::instance().Register(name,
        [](const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
            auto op = std::make_shared<SelectionOp>(SelectionParam{tensors.at("input"), tensors.at("output")});
            auto kernel = CreateKernelForTensor(context.device, node.op_type, {tensors.at("input"), tensors.at("output")});
            if (!kernel) return std::shared_ptr<OpBase>{};
            op->AttachKernel(std::move(kernel));
            return std::static_pointer_cast<OpBase>(op);
        });
}

void Candidate(const std::string& name, DeviceType device, std::string id, int tag,
               StatusCode rejection = StatusCode::kOk, int priority = 0) {
    KernelDispatcher::instance().RegisterCandidate(device, DataType::FP32, DataLayout::ND, name, std::move(id),
        [tag, rejection] { return std::make_unique<SelectionKernel>(tag, rejection); }, {}, priority);
}

model::NodeDesc SelectionNode(const std::string& name) {
    model::NodeDesc node; node.name = "selection"; node.op_type = name;
    node.inputs = {"input"}; node.outputs = {"output"}; return node;
}

TEST(kernel_selection_contract_test, BoundUnsupportedCandidateContinuesBeforeDirectOperatorExecution) {
    const std::string name = "BoundCandidateDirect";
    RegisterSelectionOp(name);
    Candidate(name, DeviceType::COMMON, "reject", 90, StatusCode::kUnsupported, 10);
    Candidate(name, DeviceType::COMMON, "accept", 3);
    auto tensors = SelectionTensors();
    auto op = OperatorRegistry::instance().Create(SelectionNode(name), tensors,
        OperatorRegistry::BuildContext{DeviceType::COMMON});
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->Run(), 0);
    EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 5);
}

TEST(kernel_selection_contract_test, BoundUnsupportedCandidatesUseCommonOnlyWhenPermitted) {
    const std::string name = "BoundCandidateCommonFallback";
    RegisterSelectionOp(name);
    Candidate(name, DeviceType::CUDA, "reject", 90, StatusCode::kUnsupported, 10);
    Candidate(name, DeviceType::COMMON, "accept", 4);
    auto tensors = SelectionTensors();
    auto op = OperatorRegistry::instance().Create(SelectionNode(name), tensors,
        OperatorRegistry::BuildContext{DeviceType::CUDA});
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->kernel()->device(), DeviceType::COMMON);
    ASSERT_EQ(op->Run(), 0);
    EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 6);
    EXPECT_EQ(OperatorRegistry::instance().Create(SelectionNode(name), tensors,
        OperatorRegistry::BuildContext{DeviceType::CUDA, KernelFallbackPolicy::kRequireRequested}), nullptr);
    EXPECT_EQ(OperatorRegistry::instance().LastCreateStatus().code(), StatusCode::kUnsupported);
    EXPECT_NE(OperatorRegistry::instance().LastCreateStatus().message().find("candidate=reject"), std::string::npos);
    EXPECT_NE(OperatorRegistry::instance().LastCreateStatus().message().find("bound shape rejected"), std::string::npos);
}

TEST(kernel_selection_contract_test, InvalidParametersAndExecutionErrorsNeverTryLaterCandidates) {
    for (const auto rejection : {StatusCode::kInvalidArgument, StatusCode::kExecutionFailed}) {
        const auto name = "BoundCandidateFatal" + std::to_string(static_cast<int>(rejection));
        RegisterSelectionOp(name);
        Candidate(name, DeviceType::CUDA, "fatal", 90, rejection, 10);
        Candidate(name, DeviceType::CUDA, "next", 3);
        Candidate(name, DeviceType::COMMON, "common", 4);
        auto tensors = SelectionTensors();
        EXPECT_EQ(OperatorRegistry::instance().Create(SelectionNode(name), tensors,
            OperatorRegistry::BuildContext{DeviceType::CUDA}), nullptr);
        EXPECT_EQ(OperatorRegistry::instance().LastCreateStatus().code(), rejection);
        EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 0);
    }
}

TEST(kernel_selection_contract_test, DeferredRecipePinsAcceptedFactoryAndParameters) {
    const std::string name = "BoundCandidateSnapshot";
    RegisterSelectionOp(name);
    Candidate(name, DeviceType::COMMON, "reject", 90, StatusCode::kUnsupported, 10);
    Candidate(name, DeviceType::COMMON, "accept", 3);
    auto tensors = SelectionTensors();
    OperatorRegistry::BuildContext context{DeviceType::COMMON}; context.defer_kernels = true;
    auto op = OperatorRegistry::instance().Create(SelectionNode(name), tensors, context);
    ASSERT_NE(op, nullptr);
    ASSERT_NE(dynamic_cast<const DeferredKernel*>(op->kernel()), nullptr);
    auto factory = op->kernel()->InstanceFactory();
    auto parameters = op->kernel()->SnapshotParameters();
    Candidate(name, DeviceType::COMMON, "accept", 70);
    Candidate(name, DeviceType::COMMON, "new_high_priority", 80, StatusCode::kOk, 100);
    auto kernel = factory(); ASSERT_NE(kernel, nullptr);
    KernelBase::TensorBindings bindings;
    bindings[tensors.at("input").get()] = tensors.at("input");
    bindings[tensors.at("output").get()] = tensors.at("output");
    ASSERT_TRUE(parameters(*kernel, bindings).ok());
    ASSERT_TRUE(kernel->Supports().ok());
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 5);
}

TEST(kernel_selection_contract_test, RequestPredicateFatalErrorsStopBeforeFactoryCreation) {
    const std::string name = "BoundCandidatePredicateFatal";
    RegisterSelectionOp(name);
    KernelDispatcher::instance().RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND,
        name, "fatal_predicate", [] { return std::make_unique<SelectionKernel>(90); },
        [](const KernelRequest&) { return Status::Error(StatusCode::kInvalidArgument, "invalid request metadata"); }, 10);
    Candidate(name, DeviceType::COMMON, "accept", 3);
    auto tensors = SelectionTensors();
    EXPECT_EQ(OperatorRegistry::instance().Create(SelectionNode(name), tensors,
        OperatorRegistry::BuildContext{DeviceType::COMMON}), nullptr);
    EXPECT_EQ(OperatorRegistry::instance().LastCreateStatus().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(OperatorRegistry::instance().LastCreateStatus().message().find("fatal_predicate"), std::string::npos);
    EXPECT_NE(OperatorRegistry::instance().LastCreateStatus().message().find("invalid request metadata"), std::string::npos);
}

TEST(kernel_selection_contract_test, CudaInt8DoesNotFallbackAfterBoundCandidateRejection) {
    const std::string name = "BoundCandidateInt8Strict";
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::INT8, DataLayout::ND, name, "reject",
        [] { return std::make_unique<SelectionKernel>(90, StatusCode::kUnsupported); });
    dispatcher.RegisterCandidate(DeviceType::COMMON, DataType::INT8, DataLayout::ND, name, "accept",
        [] { return std::make_unique<SelectionKernel>(3); });
    auto tensors = SelectionTensors();
    auto provisional = dispatcher.create(KernelRequest{DeviceType::CUDA, DataType::INT8, DataLayout::ND, name, {}});
    ASSERT_NE(provisional, nullptr);
    provisional->SetParamOwner(std::make_shared<SelectionParam>(SelectionParam{tensors.at("input"), tensors.at("output")}));
    std::unique_ptr<KernelBase> selected;
    const auto status = provisional->SelectBoundCandidate(&selected);
    EXPECT_EQ(status.code(), StatusCode::kUnsupported);
    EXPECT_EQ(selected, nullptr);
    EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 0);
}

TEST(kernel_selection_contract_test, ExecutionFailureDoesNotRetryAfterOutputMutation) {
    const std::string name = "BoundCandidateExecutionFailure";
    RegisterSelectionOp(name);
    KernelDispatcher::instance().RegisterCandidate(DeviceType::COMMON, DataType::FP32, DataLayout::ND,
        name, "execution_failure", [] { return std::make_unique<SelectionKernel>(90, StatusCode::kOk, true); }, {}, 10);
    Candidate(name, DeviceType::COMMON, "next", 3);
    auto tensors = SelectionTensors();
    auto op = OperatorRegistry::instance().Create(SelectionNode(name), tensors,
        OperatorRegistry::BuildContext{DeviceType::COMMON});
    ASSERT_NE(op, nullptr);
    EXPECT_FALSE(op->kernel()->HasSelectionRecipe());
    EXPECT_EQ(op->Run(), -1);
    EXPECT_FLOAT_EQ(tensors.at("output")->data<float>()[0], 92);
}

TEST(kernel_selection_contract_test, EarlyBuilderFailureDoesNotReusePreviousSelectionError) {
    const std::string missing = "BoundCandidateMissingBeforeEarlyFailure";
    RegisterSelectionOp(missing);
    auto tensors = SelectionTensors();
    EXPECT_EQ(OperatorRegistry::instance().Create(SelectionNode(missing), tensors,
        OperatorRegistry::BuildContext{DeviceType::COMMON}), nullptr);
    ASSERT_EQ(OperatorRegistry::instance().LastCreateStatus().code(), StatusCode::kUnsupported);
    const std::string early = "BoundCandidateEarlyBuilderFailure";
    OperatorRegistry::instance().Register(early,
        [](const model::NodeDesc&, OperatorRegistry::TensorMap&, const OperatorRegistry::BuildContext&) {
            return std::shared_ptr<OpBase>{};
        });
    EXPECT_EQ(OperatorRegistry::instance().Create(SelectionNode(early), tensors,
        OperatorRegistry::BuildContext{DeviceType::COMMON}), nullptr);
    EXPECT_EQ(OperatorRegistry::instance().LastCreateStatus().code(), StatusCode::kBuildFailed);
    EXPECT_NE(OperatorRegistry::instance().LastCreateStatus().message().find(early), std::string::npos);
}
}  // namespace
}  // namespace feather
