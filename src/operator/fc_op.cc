#include "src/operator/fc_op.h"

#include <cstdint>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

using feather::operators::FcOp;
namespace feather {
namespace operators {

namespace {

std::unique_ptr<KernelBase> CreateFcKernel(const OperatorRegistry::BuildContext& context) {
    kernel::EnsureFcKernelsRegistered();
    return CreateKernelForTensor(context.device, "FC", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildFcOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    FcParam param;
    param.input = tensors[node.inputs[0]];
    param.w = tensors[node.inputs[1]];
    param.bias = node.inputs.size() > 2 ? tensors[node.inputs[2]] : nullptr;
    param.out = tensors[node.outputs[0]];
    if (param.input == nullptr || param.w == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<FcOp>(node.name.empty() ? "fc" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "FC", {param.input, param.w, param.bias, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_fc_op_registered = []() {
    OperatorRegistry::instance().Register("FC", BuildFcOp);
    return true;
}();

}  // namespace

void EnsureFcOperatorRegistered() { (void)g_fc_op_registered; }

FcOp::FcOp() : OpBase("fc", "FC") {}

FcOp::FcOp(const FcParam& param) : FcOp("fc", param) {}

FcOp::FcOp(std::string name, const FcParam& param) : OpBase(std::move(name), "FC"), param_(param) {
    SyncIO();
}

void FcOp::SyncIO() {
    SetInputs({param_.input, param_.w, param_.bias});
    SetOutputs({param_.out});
}

int32_t FcOp::CheckShape() const {
    if (param_.input == nullptr || param_.w == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.input->dims().size() != 2 || param_.w->dims().size() != 2) {
        return -1;
    }
    if (param_.input->dims()[1] != param_.w->dims()[0]) {
        return -1;
    }
    if (param_.bias != nullptr && param_.bias->IsInitialized()) {
        if (param_.bias->dims().size() == 1) {
            if (param_.bias->dims()[0] != param_.w->dims()[1]) {
                return -1;
            }
        } else if (param_.bias->dims().size() == 2) {
            if (param_.bias->dims()[0] != param_.input->dims()[0] || param_.bias->dims()[1] != param_.w->dims()[1]) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

int32_t FcOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const std::vector<int64_t> out_shape = {param_.input->dims()[0], param_.w->dims()[1]};
    const size_t required_bytes =
        static_cast<size_t>(out_shape[0] * out_shape[1]) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.w, param_.bias, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    return 0;
}

int32_t FcOp::Run() {
    if (InferOutputShapes() != 0) {
        return -1;
    }
    if (kernel_ == nullptr) {
        return -1;
    }

    auto status = kernel_->compute();
    if (status != 0) {
        return status;
    }
    return 0;
}


}  // namespace operators
}  // namespace feather
