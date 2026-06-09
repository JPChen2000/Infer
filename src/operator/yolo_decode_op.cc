#include "src/operator/yolo_decode_op.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

int64_t ComputeNumel(const std::vector<int64_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
}

bool IsScalarTensor(const std::shared_ptr<Tensor>& tensor) {
    return tensor != nullptr && tensor->IsInitialized() && tensor->numel() == 1;
}

std::shared_ptr<OpBase> BuildYoloDecodeOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors) {
    if (node.inputs.size() != 6 || node.outputs.size() != 1) {
        return nullptr;
    }

    YoloDecodeParam param{};
    param.input = tensors[node.inputs[0]];
    param.xy_scale = tensors[node.inputs[1]];
    param.grid = tensors[node.inputs[2]];
    param.stride = tensors[node.inputs[3]];
    param.wh_scale = tensors[node.inputs[4]];
    param.anchor_grid = tensors[node.inputs[5]];
    param.out = tensors[node.outputs[0]];
    if (param.input == nullptr || param.xy_scale == nullptr || param.grid == nullptr ||
        param.stride == nullptr || param.wh_scale == nullptr || param.anchor_grid == nullptr ||
        param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<YoloDecodeOp>(node.name.empty() ? "yolo_decode" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateHostKernelForTensor(
        "YoloDecode",
        {param.input, param.xy_scale, param.grid, param.stride, param.wh_scale, param.anchor_grid, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_yolo_decode_op_registered = []() {
    OperatorRegistry::instance().Register("YoloDecode", BuildYoloDecodeOp);
    return true;
}();

}  // namespace

void EnsureYoloDecodeOperatorRegistered() { (void)g_yolo_decode_op_registered; }

YoloDecodeOp::YoloDecodeOp() : OpBase("yolo_decode", "YoloDecode") {}

YoloDecodeOp::YoloDecodeOp(const YoloDecodeParam& param) : YoloDecodeOp("yolo_decode", param) {}

YoloDecodeOp::YoloDecodeOp(std::string name, const YoloDecodeParam& param)
    : OpBase(std::move(name), "YoloDecode"), param_(param) {
    SyncIO();
}

void YoloDecodeOp::SyncIO() {
    SetInputs({param_.input, param_.xy_scale, param_.grid, param_.stride, param_.wh_scale, param_.anchor_grid});
    SetOutputs({param_.out});
}

int32_t YoloDecodeOp::CheckShape() const {
    if (param_.input == nullptr || param_.xy_scale == nullptr || param_.grid == nullptr ||
        param_.stride == nullptr || param_.wh_scale == nullptr || param_.anchor_grid == nullptr ||
        param_.out == nullptr) {
        return -1;
    }
    if (!IsScalarTensor(param_.xy_scale) || !IsScalarTensor(param_.stride) || !IsScalarTensor(param_.wh_scale)) {
        return -1;
    }
    const auto& in_dims = param_.input->dims().data();
    const auto& grid_dims = param_.grid->dims().data();
    const auto& anchor_dims = param_.anchor_grid->dims().data();
    if (in_dims.size() != 4 || grid_dims.size() != 5 || anchor_dims.size() != 5) {
        return -1;
    }
    if (grid_dims != anchor_dims || grid_dims[4] != 2) {
        return -1;
    }
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(in_dims, param_.input->layout(), &input_shape)) {
        return -1;
    }
    const int64_t batch = input_shape.n;
    const int64_t channels = input_shape.c;
    const int64_t height = input_shape.h;
    const int64_t width = input_shape.w;
    const int64_t anchors = grid_dims[1];
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || anchors <= 0) {
        return -1;
    }
    if (grid_dims[0] != 1 && grid_dims[0] != batch) {
        return -1;
    }
    if (grid_dims[2] != height || grid_dims[3] != width || channels % anchors != 0) {
        return -1;
    }
    const int64_t attrs = channels / anchors;
    if (attrs < 5) {
        return -1;
    }
    return 0;
}

int32_t YoloDecodeOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    const auto& in_dims = param_.input->dims().data();
    const auto& grid_dims = param_.grid->dims().data();
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(in_dims, param_.input->layout(), &input_shape)) {
        return -1;
    }
    const int64_t batch = input_shape.n;
    const int64_t anchors = grid_dims[1];
    const int64_t height = input_shape.h;
    const int64_t width = input_shape.w;
    const int64_t attrs = input_shape.c / anchors;
    const std::vector<int64_t> out_shape = {batch, anchors * height * width, attrs};
    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, ComputeNumel(out_shape))) *
        DataTypeBytes(ResolveExecutionDataType(
            {param_.input, param_.xy_scale, param_.grid, param_.stride, param_.wh_scale, param_.anchor_grid,
             param_.out},
            DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    param_.out->set_layout(DataLayout::ND);
    SyncIO();
    return 0;
}

int32_t YoloDecodeOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
