#include "src/operator/add_op.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {

namespace {

bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                         std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
        const int64_t lhs_dim = i < out_rank - lhs_dims.size() ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = i < out_rank - rhs_dims.size() ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

bool CheckBinaryShape(const BinaryParam& param) {
    if (param.lhs == nullptr || param.rhs == nullptr || param.out == nullptr) {
        return false;
    }
    std::vector<int64_t> out_dims;
    return InferBroadcastShape(param.lhs->dims().data(), param.rhs->dims().data(), &out_dims);
}

std::shared_ptr<OpBase> BuildAddOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    BinaryParam param{};
    param.lhs = tensors[node.inputs[0]];
    param.rhs = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<AddOp>(node.name.empty() ? "add" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Add", {param.lhs, param.rhs, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_add_op_registered = []() {
    OperatorRegistry::instance().Register("Add", BuildAddOp);
    return true;
}();

}  // namespace

void EnsureAddOperatorRegistered() { (void)g_add_op_registered; }

AddOp::AddOp() : OpBase("add", "Add") {}

AddOp::AddOp(const BinaryParam& param) : AddOp("add", param) {}

AddOp::AddOp(std::string name, const BinaryParam& param) : OpBase(std::move(name), "Add"), param_(param) { SyncIO(); }

void AddOp::SyncIO() {
    SetInputs({param_.lhs, param_.rhs});
    SetOutputs({param_.out});
}

int32_t AddOp::CheckShape() const { return CheckBinaryShape(param_) ? 0 : -1; }

int32_t AddOp::InferOutputShapes() {
    const int32_t status = elementwise_detail::InferBinaryOutput(&param_);
    if (status == 0) SyncIO();
    return status;
}

void AddOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<BinaryParam>(param_));
    }
}

int32_t AddOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    RefreshKernelParams();
    return kernel_->compute();
}

void AddOp::RefreshKernelParams() {
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<BinaryParam>(param_));
}

}  // namespace operators
}  // namespace feather
