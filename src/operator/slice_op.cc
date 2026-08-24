#include "src/operator/slice_op.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <cstdint>
#include <utility>

#include "core/operator_registry.h"
#include "src/operator/control_tensor.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

int32_t GetIntAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes, const std::string& key,
                        int32_t default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return static_cast<int32_t>(*value);
    }
    return default_value;
}

std::unique_ptr<KernelBase> CreateSliceKernel(const OperatorRegistry::BuildContext& context) {
    kernel::EnsureSliceKernelsRegistered();
    return CreateKernelForTensor(context.device, "Slice", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildSliceOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if ((node.inputs.size() != 1 && (node.inputs.size() < 4 || node.inputs.size() > 5)) || node.outputs.size() != 1) {
        return nullptr;
    }

    SliceParam param{};
    param.input = tensors[node.inputs[0]];
    if (node.inputs.size() >= 4) {
        param.starts = tensors[node.inputs[1]];
        param.ends = tensors[node.inputs[2]];
        param.axes = tensors[node.inputs[3]];
        if (node.inputs.size() == 5) {
            param.steps = tensors[node.inputs[4]];
        }
    }
    param.out = tensors[node.outputs[0]];
    param.axis = GetIntAttribute(node.attributes, "axis", 1);
    param.start = GetIntAttribute(node.attributes, "start", 0);
    param.end = GetIntAttribute(node.attributes, "end", 0);
    if (param.input == nullptr || param.out == nullptr ||
        ((param.starts == nullptr || param.ends == nullptr || param.axes == nullptr) && node.inputs.size() != 1)) {
        return nullptr;
    }

    auto op = std::make_shared<SliceOp>(node.name.empty() ? "slice" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Slice", {param.input, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_slice_op_registered = []() {
    OperatorRegistry::instance().Register("Slice", BuildSliceOp);
    return true;
}();

}  // namespace

void EnsureSliceOperatorRegistered() { (void)g_slice_op_registered; }

SliceOp::SliceOp() : OpBase("slice", "Slice") {}

SliceOp::SliceOp(const SliceParam& param) : SliceOp("slice", param) {}

SliceOp::SliceOp(std::string name, const SliceParam& param) : OpBase(std::move(name), "Slice"), param_(param) {
    SyncIO();
}

void SliceOp::SyncIO() {
    std::vector<std::shared_ptr<Tensor>> inputs = {param_.input};
    if (param_.starts != nullptr) {
        inputs.push_back(param_.starts);
        inputs.push_back(param_.ends);
        inputs.push_back(param_.axes);
        if (param_.steps != nullptr) {
            inputs.push_back(param_.steps);
        }
    }
    SetInputs(std::move(inputs));
    SetOutputs({param_.out});
}

int32_t SliceOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    int32_t axis_attr = param_.axis;
    int32_t start_attr = param_.start;
    int32_t end_attr = param_.end;
    if (param_.starts != nullptr) {
        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        std::vector<int64_t> axes;
        std::vector<int64_t> steps;
        if (!ReadIntegerTensor(param_.starts, &starts) || !ReadIntegerTensor(param_.ends, &ends) ||
            !ReadIntegerTensor(param_.axes, &axes) || starts.size() != 1 || ends.size() != 1 || axes.size() != 1 ||
            (param_.steps != nullptr && (!ReadIntegerTensor(param_.steps, &steps) || steps.size() != 1 || steps[0] != 1))) {
            return -1;
        }
        axis_attr = static_cast<int32_t>(axes[0]);
        start_attr = static_cast<int32_t>(starts[0]);
        end_attr = static_cast<int32_t>(ends[0]);
    }
    const auto rank = static_cast<int32_t>(param_.input->dims().size());
    if (rank <= 0) {
        return -1;
    }
    const int32_t axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
    if (axis < 0 || axis >= rank) {
        return -1;
    }
    const int64_t dim = param_.input->dims()[axis];
    int64_t start = start_attr < 0 ? start_attr + dim : start_attr;
    int64_t end = end_attr < 0 ? end_attr + dim : end_attr;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);
    return start < end ? 0 : -1;
}

int32_t SliceOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }

    if (param_.starts != nullptr) {
        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        std::vector<int64_t> axes;
        if (!ReadIntegerTensor(param_.starts, &starts) || !ReadIntegerTensor(param_.ends, &ends) ||
            !ReadIntegerTensor(param_.axes, &axes)) {
            return -1;
        }
        param_.start = static_cast<int32_t>(starts[0]);
        param_.end = static_cast<int32_t>(ends[0]);
        param_.axis = static_cast<int32_t>(axes[0]);
    }
    std::vector<int64_t> out_shape = param_.input->dims().data();
    const int32_t rank = static_cast<int32_t>(param_.input->dims().size());
    const int32_t axis = param_.axis < 0 ? param_.axis + rank : param_.axis;
    const int64_t dim = param_.input->dims()[axis];
    int64_t start = param_.start < 0 ? param_.start + dim : param_.start;
    int64_t end = param_.end < 0 ? param_.end + dim : param_.end;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);
    out_shape[axis] = end - start;
    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1},
                                                                 std::multiplies<int64_t>()))) *
        DataTypeBytes(ResolveExecutionDataType({param_.input}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(required_bytes);
        param_.out->Resize(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_data_type(param_.input->data_type());
    SyncIO();
    return 0;
}

void SliceOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParam((void*)&param_);
    }
}

int32_t SliceOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
