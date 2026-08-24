#include "src/operator/reshape_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/control_tensor.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

std::vector<int64_t> GetShapeAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                                       const std::string& key) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return {};
    }
    if (auto value = std::get_if<std::vector<int64_t>>(&it->second); value != nullptr) {
        return *value;
    }
    return {};
}

std::unique_ptr<KernelBase> CreateReshapeKernel(const OperatorRegistry::BuildContext& context) {
    kernel::EnsureReshapeKernelsRegistered();
    return CreateKernelForTensor(context.device, "Reshape", {});
}

bool ResolveTargetShape(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& shape_tensor,
                        const std::vector<int64_t>& shape_attribute, std::vector<int64_t>* target_shape) {
    if (input == nullptr || target_shape == nullptr) {
        return false;
    }

    std::vector<int64_t> raw_shape = shape_attribute;
    if (shape_tensor != nullptr && !ReadIntegerTensor(shape_tensor, &raw_shape)) {
        return false;
    }
    if (raw_shape.empty()) {
        return false;
    }

    target_shape->clear();
    target_shape->reserve(raw_shape.size());
    int64_t known_numel = 1;
    int64_t infer_axis = -1;
    const auto input_dims = input->dims().data();
    for (size_t axis = 0; axis < raw_shape.size(); ++axis) {
        int64_t dim = raw_shape[axis];
        if (dim == 0) {
            if (axis >= input_dims.size() || input_dims[axis] <= 0) {
                return false;
            }
            dim = input_dims[axis];
        } else if (dim == -1) {
            if (infer_axis >= 0) {
                return false;
            }
            infer_axis = static_cast<int64_t>(axis);
            target_shape->push_back(-1);
            continue;
        } else if (dim <= 0) {
            return false;
        }
        if (known_numel > input->numel() / dim) {
            return false;
        }
        known_numel *= dim;
        target_shape->push_back(dim);
    }

    if (infer_axis >= 0) {
        if (known_numel == 0 || input->numel() % known_numel != 0) {
            return false;
        }
        (*target_shape)[static_cast<size_t>(infer_axis)] = input->numel() / known_numel;
    }

    int64_t target_numel = 1;
    for (const auto dim : *target_shape) {
        if (dim <= 0 || target_numel > input->numel() / dim) {
            return false;
        }
        target_numel *= dim;
    }
    return target_numel == input->numel();
}

std::shared_ptr<OpBase> BuildReshapeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 1 && node.inputs.size() != 2) || node.outputs.size() != 1) {
        return nullptr;
    }

    ReshapeParam param;
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() == 2) {
        param.shape = tensors[node.inputs[1]];
    }
    param.out = tensors[node.outputs[0]];
    param.target_shape = GetShapeAttribute(node.attributes, "shape");
    if (param.input == nullptr || param.out == nullptr || (param.shape == nullptr && param.target_shape.empty())) {
        return nullptr;
    }

    auto op = std::make_shared<ReshapeOp>(node.name.empty() ? "reshape" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Reshape", {param.input, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_reshape_op_registered = []() {
    OperatorRegistry::instance().Register("Reshape", BuildReshapeOp);
    return true;
}();

}  // namespace

void EnsureReshapeOperatorRegistered() { (void)g_reshape_op_registered; }

ReshapeOp::ReshapeOp() : OpBase("reshape", "Reshape") {}

ReshapeOp::ReshapeOp(const ReshapeParam& param) : ReshapeOp("reshape", param) {}

ReshapeOp::ReshapeOp(std::string name, const ReshapeParam& param) : OpBase(std::move(name), "Reshape"), param_(param) {
    SyncIO();
}

void ReshapeOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.shape != nullptr) {
        inputs.push_back(param_.shape);
    }
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t ReshapeOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    std::vector<int64_t> target_shape;
    return ResolveTargetShape(param_.input, param_.shape, param_.target_shape, &target_shape) ? 0 : -1;
}

int32_t ReshapeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> target_shape;
    if (!ResolveTargetShape(param_.input, param_.shape, param_.target_shape, &target_shape)) {
        return -1;
    }

    // A host reshape is a metadata-only view. Keeping the input storage avoids
    // copying every Qwen activation through shape-only nodes during decode.
    // CUDA keeps separate host storage but shares the device allocation in its
    // view kernel, so output lifetime remains independent of host metadata.
    if (ActiveKernelDevice() != DeviceType::CUDA) {
        param_.out->ShareDataWith(*param_.input);
        param_.out->Resize(target_shape);
        param_.out->set_data_type(param_.input->data_type());
        param_.out->set_layout(param_.input->layout());
        SyncIO();
        return 0;
    }

    const size_t required_bytes =
        static_cast<size_t>(param_.input->numel()) *
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(target_shape);
    } else {
        param_.out->Resize(target_shape);
    }
    param_.out->set_data_type(param_.input->data_type());
    param_.out->set_layout(param_.input->layout());
    SyncIO();
    return 0;
}

int32_t ReshapeOp::Run() {
    if (InferOutputShapes() != 0) {
        return -1;
    }
    if (kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
