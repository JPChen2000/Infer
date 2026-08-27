#include "src/operator/qwen_gemm_argmax_op.h"

#include <cmath>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {
namespace {

std::shared_ptr<OpBase> BuildQwenGemmArgmaxOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }
    QwenGemmArgmaxParam param;
    param.a = tensors[node.inputs[0]];
    param.b = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];
    param.output_scale = 1.0f;
    const auto scale_it = node.attributes.find("output_scale");
    if (scale_it != node.attributes.end()) {
        const auto* scale = std::get_if<float>(&scale_it->second);
        if (scale == nullptr) {
            return nullptr;
        }
        param.output_scale = *scale;
    }
    if (param.a == nullptr || param.b == nullptr || param.out == nullptr) {
        return nullptr;
    }
    auto op = std::make_shared<QwenGemmArgmaxOp>(node.name.empty() ? "qwen_gemm_argmax" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureQwenGemmArgmaxKernelsRegistered();
    const DataType execution_dtype = param.a->data_type();
    auto kernel = CreateKernelForTensor(context.device, "QwenGemmArgmax", {param.a, param.b}, execution_dtype);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_registered = []() {
    OperatorRegistry::instance().Register("QwenGemmArgmax", BuildQwenGemmArgmaxOp);
    return true;
}();

}  // namespace

void EnsureQwenGemmArgmaxOperatorRegistered() { (void)g_registered; }

QwenGemmArgmaxOp::QwenGemmArgmaxOp(std::string name, const QwenGemmArgmaxParam& param)
    : OpBase(std::move(name), "QwenGemmArgmax"), param_(param) {
    SyncIO();
}

void QwenGemmArgmaxOp::SyncIO() { SetInputs({param_.a, param_.b}); SetOutputs({param_.out}); }

int32_t QwenGemmArgmaxOp::CheckShape() const {
    const auto dtype = param_.a == nullptr ? DataType::UNKNOWN : param_.a->data_type();
    const bool supported_dtype = dtype == DataType::BF16 || dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2;
    if (param_.a == nullptr || param_.b == nullptr || param_.out == nullptr || !param_.a->IsInitialized() ||
        !param_.b->IsInitialized() || !supported_dtype || param_.b->data_type() != dtype ||
        !std::isfinite(param_.output_scale) || param_.output_scale <= 0.0f || param_.a->dims().size() < 2 ||
        param_.b->dims().size() != 2 ||
        param_.a->dims()[param_.a->dims().size() - 1] != param_.b->dims()[1] ||
        param_.a->numel() != param_.a->dims()[param_.a->dims().size() - 1] ||
        param_.b->dims()[0] <= 0) {
        return -1;
    }
    return 0;
}

int32_t QwenGemmArgmaxOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    if (!param_.out->IsInitialized() || param_.out->memory_size() < sizeof(int64_t)) {
        param_.out = std::make_shared<Tensor>(sizeof(int64_t));
    }
    param_.out->Resize({1});
    param_.out->set_data_type(DataType::INT64);
    SyncIO();
    return 0;
}

void QwenGemmArgmaxOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t QwenGemmArgmaxOp::Run() { return InferOutputShapes() == 0 && kernel_ != nullptr ? kernel_->compute() : -1; }

}  // namespace operators
}  // namespace feather
