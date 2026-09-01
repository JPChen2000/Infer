#include "src/operator/reshape_op.h"

#include <limits>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"
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

    const auto input_dims = input->dims().data();
    bool input_shape_known = true;
    int64_t input_numel = 1;
    for (const auto input_dim : input_dims) {
        if (input_dim <= 0) {
            input_shape_known = false;
            break;
        }
        if (input_numel > std::numeric_limits<int64_t>::max() / input_dim) {
            return false;
        }
        input_numel *= input_dim;
    }

    target_shape->clear();
    target_shape->reserve(raw_shape.size());
    int64_t known_numel = 1;
    int64_t infer_axis = -1;
    for (size_t axis = 0; axis < raw_shape.size(); ++axis) {
        int64_t dim = raw_shape[axis];
        if (dim == 0) {
            if (axis >= input_dims.size() || input_dims[axis] < 0) {
                return false;
            }
            dim = input_dims[axis];
            if (dim == 0) {
                target_shape->push_back(0);
                continue;
            }
        } else if (dim == -1) {
            if (infer_axis >= 0) {
                return false;
            }
            infer_axis = static_cast<int64_t>(axis);
            // The inferred dimension is not knowable until the input shape is
            // refreshed at runtime when the input contains unknown dimensions.
            target_shape->push_back(0);
            continue;
        } else if (dim < 0) {
            return false;
        }
        if (dim == 0) {
            target_shape->push_back(0);
            continue;
        }
        if (known_numel > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        known_numel *= dim;
        target_shape->push_back(dim);
    }

    if (infer_axis >= 0) {
        if (input_shape_known) {
            if (known_numel == 0 || input_numel % known_numel != 0) {
                return false;
            }
            const int64_t inferred_dim = input_numel / known_numel;
            if (inferred_dim <= 0) {
                return false;
            }
            (*target_shape)[static_cast<size_t>(infer_axis)] = inferred_dim;
        }
    }

    // Zero dimensions in the model are placeholders for runtime dimensions,
    // not real zero-element tensors. Defer the exact element-count check until
    // the input has been populated with its runtime shape.
    if (!input_shape_known) {
        return true;
    }

    int64_t target_numel = 1;
    for (const auto dim : *target_shape) {
        if (dim <= 0 || target_numel > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        target_numel *= dim;
    }
    return target_numel == input_numel;
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
    op->SetExecutionDevice(context.device);
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
    const auto device = execution_device_explicit_ ? execution_device_ : ActiveKernelDevice();
    if (device != DeviceType::CUDA && param_.input->data_type() != DataType::INT8) {
        param_.out->ShareDataWith(*param_.input);
        param_.out->Resize(target_shape);
        param_.out->set_data_type(param_.input->data_type());
        param_.out->set_layout(param_.input->layout());
        SyncIO();
        return 0;
    }

    if (tensor_op_detail::InferSameTypeOutput(param_.input, &param_.out, target_shape) != 0) return -1;
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
    RefreshKernelParams();
    return kernel_->compute();
}

void ReshapeOp::RefreshKernelParams() {
    if (kernel_ != nullptr) kernel_->SetParamOwner(std::make_shared<ReshapeParam>(param_));
}

}  // namespace operators
}  // namespace feather
