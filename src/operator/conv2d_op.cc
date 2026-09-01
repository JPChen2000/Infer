#include "src/operator/conv2d_op.h"

#include <algorithm>
#include <cstdint>
#include <functional>
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

std::shared_ptr<OpBase> BuildConv2DOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    Conv2dParam param{};
    param.input = tensors[node.inputs[0]];
    param.w = tensors[node.inputs[1]];
    param.bias = node.inputs.size() > 2 ? tensors[node.inputs[2]] : nullptr;
    param.out = tensors[node.outputs[0]];
    param.stride_h = GetIntAttribute(node.attributes, "stride_h", 1);
    param.stride_w = GetIntAttribute(node.attributes, "stride_w", 1);
    param.pad_h = GetIntAttribute(node.attributes, "pad_h", 0);
    param.pad_w = GetIntAttribute(node.attributes, "pad_w", 0);
    param.dilation_h = GetIntAttribute(node.attributes, "dilation_h", 1);
    param.dilation_w = GetIntAttribute(node.attributes, "dilation_w", 1);
    param.group = GetIntAttribute(node.attributes, "group", 1);
    if (param.input == nullptr || param.w == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<Conv2dOp>(node.name.empty() ? "conv2d" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    auto kernel = CreateKernelForTensor(context.device, "Conv2D", {param.input, param.w, param.bias, param.out});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_conv2d_op_registered = []() {
    OperatorRegistry::instance().Register("Conv2D", BuildConv2DOp);
    OperatorRegistry::instance().Register("Conv", BuildConv2DOp);
    return true;
}();

}  // namespace

void EnsureConv2DOperatorRegistered() { (void)g_conv2d_op_registered; }

Conv2dOp::Conv2dOp() : OpBase("conv2d", "Conv2D") {}

Conv2dOp::Conv2dOp(const Conv2dParam& param) : Conv2dOp("conv2d", param) {}

Conv2dOp::Conv2dOp(std::string name, const Conv2dParam& param) : OpBase(std::move(name), "Conv2D"), param_(param) {
    SyncIO();
}

void Conv2dOp::SyncIO() {
    SetInputs({param_.input, param_.w, param_.bias});
    SetOutputs({param_.out});
}

int32_t Conv2dOp::CheckShape() const {
    if (param_.input == nullptr || param_.w == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.stride_h <= 0 || param_.stride_w <= 0 || param_.pad_h < 0 || param_.pad_w < 0 ||
        param_.dilation_h <= 0 || param_.dilation_w <= 0 || param_.group <= 0) {
        return -1;
    }
    if (param_.input->dims().size() == 2 && param_.w->dims().size() == 2) {
        const int64_t in_h = param_.input->dims()[0];
        const int64_t in_w = param_.input->dims()[1];
        const int64_t kernel_h = param_.w->dims()[0];
        const int64_t kernel_w = param_.w->dims()[1];
        const int64_t out_h = (in_h + 2 * param_.pad_h - param_.dilation_h * (kernel_h - 1) - 1) / param_.stride_h + 1;
        const int64_t out_w = (in_w + 2 * param_.pad_w - param_.dilation_w * (kernel_w - 1) - 1) / param_.stride_w + 1;
        if (out_h <= 0 || out_w <= 0) {
            return -1;
        }
        if (param_.bias != nullptr && param_.bias->IsInitialized()) {
            if (param_.bias->dims().size() != 2 || param_.bias->dims()[0] != out_h || param_.bias->dims()[1] != out_w) {
                return -1;
            }
        }
        return 0;
    }

    if (param_.input->dims().size() != 4 || param_.w->dims().size() != 4) {
        return -1;
    }
    ImageShape4D input_shape;
    if (!DecodeImageShape4D(param_.input->dims().data(), param_.input->layout(), &input_shape)) {
        return -1;
    }
    const int64_t in_c = input_shape.c;
    const int64_t in_h = input_shape.h;
    const int64_t in_w = input_shape.w;
    const int64_t out_c = param_.w->dims()[0];
    const int64_t weight_c = param_.w->dims()[1];
    const int64_t kernel_h = param_.w->dims()[2];
    const int64_t kernel_w = param_.w->dims()[3];
    if (in_c % param_.group != 0 || weight_c * param_.group != in_c) {
        return -1;
    }
    const int64_t out_h = (in_h + 2 * param_.pad_h - param_.dilation_h * (kernel_h - 1) - 1) / param_.stride_h + 1;
    const int64_t out_w = (in_w + 2 * param_.pad_w - param_.dilation_w * (kernel_w - 1) - 1) / param_.stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        return -1;
    }
    if (param_.bias != nullptr && param_.bias->IsInitialized()) {
        if (param_.bias->numel() != out_c) {
            return -1;
        }
    }
    return 0;
}

int32_t Conv2dOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    // Shape inference may need to grow the output buffer at runtime. Keep the
    // existing Tensor object (the graph's value map owns that handle) and
    // preserve its execution and quantization contract while replacing only
    // the backing storage. Constructing a new Tensor here would default it to
    // FP32 with disabled quantization and disconnect the operator from the
    // graph's original output value.
    const DataType declared_output_data_type = param_.out->data_type();
    const DataLayout output_layout = param_.out->layout();
    std::vector<int64_t> out_shape;
    if (param_.input->dims().size() == 2) {
        const int64_t out_h =
            (param_.input->dims()[0] + 2 * param_.pad_h - param_.dilation_h * (param_.w->dims()[0] - 1) - 1) / param_.stride_h + 1;
        const int64_t out_w =
            (param_.input->dims()[1] + 2 * param_.pad_w - param_.dilation_w * (param_.w->dims()[1] - 1) - 1) / param_.stride_w + 1;
        out_shape = {out_h, out_w};
    } else {
        const int64_t out_h =
            (param_.input->dims()[HeightAxisForLayout(param_.input->layout())] + 2 * param_.pad_h -
             param_.dilation_h * (param_.w->dims()[2] - 1) - 1) / param_.stride_h + 1;
        const int64_t out_w =
            (param_.input->dims()[WidthAxisForLayout(param_.input->layout())] + 2 * param_.pad_w -
             param_.dilation_w * (param_.w->dims()[3] - 1) - 1) / param_.stride_w + 1;
        out_shape = EncodeImageShape4D(
            ImageShape4D{param_.input->dims()[0], param_.w->dims()[0], out_h, out_w},
            NormalizeDataLayout(param_.input->layout()));
    }
    const DataType execution_data_type =
        declared_output_data_type != DataType::UNKNOWN
            ? declared_output_data_type
            : ResolveExecutionDataType({param_.input, param_.w, param_.bias, param_.out}, DataType::FP32);
    const size_t required_bytes =
        static_cast<size_t>(std::max<int64_t>(1, ComputeNumel(out_shape))) * DataTypeBytes(execution_data_type);
    if (!param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out->ResetBuffer(std::make_shared<Buffer>(required_bytes), required_bytes);
    }
    param_.out->Resize(out_shape);
    if (declared_output_data_type == DataType::UNKNOWN) {
        param_.out->set_data_type(execution_data_type);
    }
    param_.out->set_layout(param_.input->dims().size() == 4 ? NormalizeDataLayout(output_layout == DataLayout::ND
                                                                                       ? param_.input->layout()
                                                                                       : output_layout)
                                                               : DataLayout::ND);
    SyncIO();
    return 0;
}

int32_t Conv2dOp::Run() {
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
