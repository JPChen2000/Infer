#include "src/operator/global_average_pool_op.h"

#include <cstdint>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateGlobalAveragePoolKernel() {
    kernel::EnsureGlobalAveragePoolKernelsRegistered();
    return CreateHostKernelForTensor("GlobalAveragePool", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildGlobalAveragePoolOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    GlobalAveragePoolParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    if (param.input == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<GlobalAveragePoolOp>(node.name.empty() ? "global_average_pool" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("GlobalAveragePool", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_global_average_pool_op_registered = []() {
    OperatorRegistry::instance().Register("GlobalAveragePool", BuildGlobalAveragePoolOp);
    return true;
}();

}  // namespace

void EnsureGlobalAveragePoolOperatorRegistered() { (void)g_global_average_pool_op_registered; }

GlobalAveragePoolOp::GlobalAveragePoolOp() : OpBase("global_average_pool", "GlobalAveragePool") {}

GlobalAveragePoolOp::GlobalAveragePoolOp(const GlobalAveragePoolParam& param)
    : GlobalAveragePoolOp("global_average_pool", param) {}

GlobalAveragePoolOp::GlobalAveragePoolOp(std::string name, const GlobalAveragePoolParam& param)
    : OpBase(std::move(name), "GlobalAveragePool"), param_(param) {
    SyncIO();
}

void GlobalAveragePoolOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t GlobalAveragePoolOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr || param_.input->dims().size() != 4) {
        return -1;
    }
    return 0;
}

int32_t GlobalAveragePoolOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const std::vector<int64_t> out_shape = {param_.input->dims()[0], param_.input->dims()[1], 1, 1};
    const int64_t out_numel = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    const size_t required_bytes =
        static_cast<size_t>(out_numel) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

void GlobalAveragePoolOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t GlobalAveragePoolOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
