#include "src/operator/slice_op.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <cstdint>
#include <utility>

#include "core/operator_registry.h"
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

std::unique_ptr<KernelBase> CreateSliceKernel() {
    kernel::EnsureSliceKernelsRegistered();
    return CreateHostKernelForTensor("Slice", {}, DataType::FP32);
}

std::shared_ptr<OpBase> BuildSliceOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }

    SliceParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];
    param.axis = GetIntAttribute(node.attributes, "axis", 1);
    param.start = GetIntAttribute(node.attributes, "start", 0);
    param.end = GetIntAttribute(node.attributes, "end", 0);

    auto op = std::make_shared<SliceOp>(node.name.empty() ? "slice" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor("Slice", {param.input, param.out}, DataType::FP32);
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
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t SliceOp::CheckShape() const {
    if (param_.input == nullptr || param_.out == nullptr) {
        return -1;
    }
    const auto rank = static_cast<int32_t>(param_.input->dims().size());
    if (rank <= 0) {
        return -1;
    }
    const int32_t axis = param_.axis < 0 ? param_.axis + rank : param_.axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }
    const int64_t dim = param_.input->dims()[axis];
    int64_t start = param_.start < 0 ? param_.start + dim : param_.start;
    int64_t end = param_.end < 0 ? param_.end + dim : param_.end;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);
    return start < end ? 0 : -1;
}

int32_t SliceOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
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
        DataTypeBytes(ResolveExecutionDataType({param_.input, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
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
