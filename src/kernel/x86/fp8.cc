#include "src/kernel/batch_normalization.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/cos.h"
#include "src/kernel/erf.h"
#include "src/kernel/exp.h"
#include "src/kernel/expand.h"
#include "src/kernel/flatten.h"
#include "src/kernel/global_average_pool.h"
#include "src/kernel/gather.h"
#include "src/kernel/identity.h"
#include "src/kernel/neg.h"
#include "src/kernel/pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/reduce_mean.h"
#include "src/kernel/reduce_sum.h"
#include "src/kernel/relu.h"
#include "src/kernel/reshape.h"
#include "src/kernel/resize.h"
#include "src/kernel/resize_concat.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/silu.h"
#include "src/kernel/sin.h"
#include "src/kernel/slice.h"
#include "src/kernel/softmax.h"
#include "src/kernel/softplus.h"
#include "src/kernel/split.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/tanh.h"
#include "src/kernel/transpose.h"
#include "src/kernel/unsqueeze.h"
#include "src/kernel/squeeze.h"
#include "src/kernel/where.h"
#include "src/kernel/equal.h"
#include "src/kernel/fp8_host.h"

#include <memory>
#include <algorithm>
#include <array>
#include <cmath>

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_x86_fp8_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
#define REGISTER_X86_FP8(op, klass)                                                                       \
    dispatcher.registerKernel(DeviceType::X86, DataType::FP8E4M3, op, []() {                           \
        return std::make_unique<klass<DeviceType::X86, DataType::FP8E4M3>>();                           \
    });                                                                                                   \
    dispatcher.registerKernel(DeviceType::X86, DataType::FP8E5M2, op, []() {                           \
        return std::make_unique<klass<DeviceType::X86, DataType::FP8E5M2>>();                           \
    });

    REGISTER_X86_FP8("BatchNormalization", BatchNormalizationKernel)
    REGISTER_X86_FP8("Concat", ConcatKernel)
    REGISTER_X86_FP8("Conv2D", Conv2DKernel)
    REGISTER_X86_FP8("Cos", CosKernel)
    REGISTER_X86_FP8("Erf", ErfKernel)
    REGISTER_X86_FP8("Exp", ExpKernel)
    REGISTER_X86_FP8("Expand", ExpandKernel)
    REGISTER_X86_FP8("Flatten", FlattenKernel)
    REGISTER_X86_FP8("GlobalAveragePool", GlobalAveragePoolKernel)
    REGISTER_X86_FP8("Gather", GatherKernel)
    REGISTER_X86_FP8("Identity", IdentityKernel)
    REGISTER_X86_FP8("AvgPool", AvgPoolKernel)
    REGISTER_X86_FP8("MaxPool", MaxPoolKernel)
    REGISTER_X86_FP8("Neg", NegKernel)
    REGISTER_X86_FP8("Pow", PowKernel)
    REGISTER_X86_FP8("ReduceMean", ReduceMeanKernel)
    REGISTER_X86_FP8("ReduceSum", ReduceSumKernel)
    REGISTER_X86_FP8("ReLU", ReluKernel)
    REGISTER_X86_FP8("Reshape", ReshapeKernel)
    REGISTER_X86_FP8("Resize", ResizeKernel)
    REGISTER_X86_FP8("ResizeConcat", ResizeConcatKernel)
    REGISTER_X86_FP8("Sigmoid", SigmoidKernel)
    REGISTER_X86_FP8("SiLU", SiluKernel)
    REGISTER_X86_FP8("Sin", SinKernel)
    REGISTER_X86_FP8("Slice", SliceKernel)
    REGISTER_X86_FP8("Softmax", SoftmaxKernel)
    REGISTER_X86_FP8("Softplus", SoftplusKernel)
    REGISTER_X86_FP8("Split", SplitKernel)
    REGISTER_X86_FP8("Sqrt", SqrtKernel)
    REGISTER_X86_FP8("Tanh", TanhKernel)
    REGISTER_X86_FP8("Transpose", TransposeKernel)
    REGISTER_X86_FP8("Unsqueeze", UnsqueezeKernel)
    REGISTER_X86_FP8("Squeeze", SqueezeKernel)
    REGISTER_X86_FP8("Where", WhereKernel)
    REGISTER_X86_FP8("Equal", EqualKernel)
#undef REGISTER_X86_FP8
    return true;
}();

template <DataType dtype>
int32_t Unary(operators::UnaryParam* param, float (*fn)(float)) {
    return fp8_host::Unary<dtype>(param, fn);
}

template <DataType dtype>
int32_t Pow(operators::PowParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) return -1;
    operators::UnaryParam unary{};
    unary.input = param->input;
    unary.out = param->out;
    return fp8_host::Unary<dtype>(&unary, [exponent](float value) { return std::pow(value, exponent); });
}

template <DataType dtype>
const std::array<float, 256>& Fp8DecodeTable() {
    static const std::array<float, 256> table = []() {
        std::array<float, 256> values{};
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = dtype == DataType::FP8E4M3
                                ? Fp8E4M3ToFloat(static_cast<uint8_t>(index))
                                : Fp8E5M2ToFloat(static_cast<uint8_t>(index));
        }
        return values;
    }();
    return table;
}

template <DataType dtype>
bool IsQwenFp8DepthwiseConvShape(const operators::Conv2dParam* param, int64_t* channels) {
    if (param == nullptr || channels == nullptr || !fp8_host::IsTensor<dtype>(param->input.get()) ||
        !fp8_host::IsTensor<dtype>(param->w.get()) || param->out == nullptr ||
        param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4 ||
        NormalizeDataLayout(param->input->layout()) != DataLayout::NCHW ||
        NormalizeDataLayout(param->out->layout()) != DataLayout::NCHW || param->stride_h != 1 ||
        param->stride_w != 1 || param->pad_h != 0 || param->pad_w != 0 || param->dilation_h != 1 ||
        param->dilation_w != 1) {
        return false;
    }
    const auto& input_dims = param->input->dims();
    const auto& weight_dims = param->w->dims();
    const auto& output_dims = param->out->dims();
    const int64_t c = input_dims[1];
    if (input_dims[0] != 1 || c <= 0 || input_dims[2] != 1 || input_dims[3] != 4 || param->group != c ||
        weight_dims[0] != c || weight_dims[1] != 1 || weight_dims[2] != 1 || weight_dims[3] != 4 ||
        output_dims[0] != 1 || output_dims[1] != c || output_dims[2] != 1 || output_dims[3] != 1 ||
        !fp8_host::HasValidFp8Quantization(param->out.get())) {
        return false;
    }
    if (param->bias != nullptr &&
        (!fp8_host::IsTensor<dtype>(param->bias.get()) || param->bias->dims().size() != 1 ||
         param->bias->numel() != c)) {
        return false;
    }
    *channels = c;
    return true;
}

template <DataType dtype>
int32_t ComputeQwenFp8DepthwiseConv2D(operators::Conv2dParam* param) {
    int64_t channels = 0;
    if (!IsQwenFp8DepthwiseConvShape<dtype>(param, &channels)) {
        return fp8_host::Conv2D<dtype>(param);
    }

    const std::vector<int64_t> expected_dims{1, channels, 1, 1};
    if (!fp8_host::PrepareOutput<dtype>(param->out.get(), &expected_dims)) {
        return -1;
    }
    const auto& table = Fp8DecodeTable<dtype>();
    const auto* input = static_cast<const uint8_t*>(param->input->raw_data());
    const auto* weight = static_cast<const uint8_t*>(param->w->raw_data());
    const auto* bias = param->bias == nullptr ? nullptr : static_cast<const uint8_t*>(param->bias->raw_data());
    auto* output = static_cast<uint8_t*>(param->out->raw_data());
    const float input_scale = param->input->quantization_scale();
    const float weight_scale = param->w->quantization_scale();
    const float bias_scale = bias == nullptr ? 1.0f : param->bias->quantization_scale();
    const float output_scale = param->out->quantization_scale();

    for (int64_t channel = 0; channel < channels; ++channel) {
        const int64_t offset = channel * 4;
        float value = 0.0f;
        for (int64_t tap = 0; tap < 4; ++tap) {
            const float input_value = table[input[offset + tap]] * input_scale;
            const float weight_value = table[weight[offset + tap]] * weight_scale;
            value += input_value * weight_value;
        }
        if (bias != nullptr) {
            value += table[bias[channel]] * bias_scale;
        }
        if constexpr (dtype == DataType::FP8E4M3) {
            output[channel] = FloatToFp8E4M3(value / output_scale);
        } else {
            output[channel] = FloatToFp8E5M2(value / output_scale);
        }
    }
    param->out->set_data_type(dtype);
    return 0;
}

template <DataType dtype>
int32_t Reshape(operators::ReshapeParam* param) {
    if (param == nullptr) return -1;
    operators::AxesParam copy{};
    copy.input = param->input;
    copy.out = param->out;
    return fp8_host::Copy<dtype>(&copy);
}

template <DataType dtype>
int32_t Flatten(operators::FlattenParam* param) { return fp8_host::Flatten<dtype>(param); }

template <DataType dtype>
int32_t Identity(operators::UnaryParam* param) {
    if (param == nullptr) return -1;
    operators::AxesParam copy{};
    copy.input = param->input;
    copy.out = param->out;
    return fp8_host::Copy<dtype>(&copy);
}

}  // namespace

#define DEFINE_X86_UNARY(Class, Function, Label)                                                         \
    template <>                                                                                            \
    int32_t Class<DeviceType::X86, DataType::FP8E4M3>::compute() {                                      \
        AutoTimer timer("X86::" Label "::FP8E4M3");                                                  \
        return Function<DataType::FP8E4M3>(static_cast<operators::UnaryParam*>(param_));                 \
    }                                                                                                     \
    template <>                                                                                            \
    int32_t Class<DeviceType::X86, DataType::FP8E5M2>::compute() {                                      \
        AutoTimer timer("X86::" Label "::FP8E5M2");                                                  \
        return Function<DataType::FP8E5M2>(static_cast<operators::UnaryParam*>(param_));                 \
    }

template <DataType dtype>
int32_t Relu(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::max(0.0f, x); }); }
template <DataType dtype>
int32_t Sigmoid(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return 1.0f / (1.0f + std::exp(-x)); }); }
template <DataType dtype>
int32_t Silu(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return x / (1.0f + std::exp(-x)); }); }
template <DataType dtype>
int32_t Tanh(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::tanh(x); }); }
template <DataType dtype>
int32_t Exp(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::exp(x); }); }
template <DataType dtype>
int32_t Sin(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::sin(x); }); }
template <DataType dtype>
int32_t Cos(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::cos(x); }); }
template <DataType dtype>
int32_t Erf(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::erf(x); }); }
template <DataType dtype>
int32_t Sqrt(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::sqrt(x); }); }
template <DataType dtype>
int32_t Softplus(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return std::max(0.0f, x) + std::log1p(std::exp(-std::fabs(x))); }); }
template <DataType dtype>
int32_t Neg(operators::UnaryParam* p) { return Unary<dtype>(p, [](float x) { return -x; }); }

DEFINE_X86_UNARY(ReluKernel, Relu, "ReLU")
DEFINE_X86_UNARY(SigmoidKernel, Sigmoid, "Sigmoid")
DEFINE_X86_UNARY(SiluKernel, Silu, "SiLU")
DEFINE_X86_UNARY(TanhKernel, Tanh, "Tanh")
DEFINE_X86_UNARY(ExpKernel, Exp, "Exp")
DEFINE_X86_UNARY(SinKernel, Sin, "Sin")
DEFINE_X86_UNARY(CosKernel, Cos, "Cos")
DEFINE_X86_UNARY(ErfKernel, Erf, "Erf")
DEFINE_X86_UNARY(SqrtKernel, Sqrt, "Sqrt")
DEFINE_X86_UNARY(SoftplusKernel, Softplus, "Softplus")
DEFINE_X86_UNARY(NegKernel, Neg, "Neg")

template <>
int32_t PowKernel<DeviceType::X86, DataType::FP8E4M3>::compute() { return Pow<DataType::FP8E4M3>(static_cast<operators::PowParam*>(param_)); }
template <>
int32_t PowKernel<DeviceType::X86, DataType::FP8E5M2>::compute() { return Pow<DataType::FP8E5M2>(static_cast<operators::PowParam*>(param_)); }
template <>
int32_t IdentityKernel<DeviceType::X86, DataType::FP8E4M3>::compute() { return Identity<DataType::FP8E4M3>(static_cast<operators::UnaryParam*>(param_)); }
template <>
int32_t IdentityKernel<DeviceType::X86, DataType::FP8E5M2>::compute() { return Identity<DataType::FP8E5M2>(static_cast<operators::UnaryParam*>(param_)); }
template <>
int32_t ReshapeKernel<DeviceType::X86, DataType::FP8E4M3>::compute() { return Reshape<DataType::FP8E4M3>(static_cast<operators::ReshapeParam*>(param_)); }
template <>
int32_t ReshapeKernel<DeviceType::X86, DataType::FP8E5M2>::compute() { return Reshape<DataType::FP8E5M2>(static_cast<operators::ReshapeParam*>(param_)); }
template <>
int32_t FlattenKernel<DeviceType::X86, DataType::FP8E4M3>::compute() { return Flatten<DataType::FP8E4M3>(static_cast<operators::FlattenParam*>(param_)); }
template <>
int32_t FlattenKernel<DeviceType::X86, DataType::FP8E5M2>::compute() { return Flatten<DataType::FP8E5M2>(static_cast<operators::FlattenParam*>(param_)); }
template <>
int32_t GlobalAveragePoolKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    return fp8_host::GlobalAveragePool<DataType::FP8E4M3>(static_cast<operators::GlobalAveragePoolParam*>(param_));
}
template <>
int32_t GlobalAveragePoolKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    return fp8_host::GlobalAveragePool<DataType::FP8E5M2>(static_cast<operators::GlobalAveragePoolParam*>(param_));
}
template <>
int32_t UnsqueezeKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    return fp8_host::Copy<DataType::FP8E4M3>(static_cast<operators::AxesParam*>(param_));
}
template <>
int32_t UnsqueezeKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    return fp8_host::Copy<DataType::FP8E5M2>(static_cast<operators::AxesParam*>(param_));
}
template <>
int32_t SqueezeKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    return fp8_host::Copy<DataType::FP8E4M3>(static_cast<operators::AxesParam*>(param_));
}
template <>
int32_t SqueezeKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    return fp8_host::Copy<DataType::FP8E5M2>(static_cast<operators::AxesParam*>(param_));
}

#define DEFINE_X86_PARAM(Class, ParamType, Function, Label)                                               \
    template <>                                                                                            \
    int32_t Class<DeviceType::X86, DataType::FP8E4M3>::compute() {                                      \
        AutoTimer timer("X86::" Label "::FP8E4M3");                                                  \
        return Function<DataType::FP8E4M3>(static_cast<ParamType*>(param_));                             \
    }                                                                                                     \
    template <>                                                                                            \
    int32_t Class<DeviceType::X86, DataType::FP8E5M2>::compute() {                                      \
        AutoTimer timer("X86::" Label "::FP8E5M2");                                                  \
        return Function<DataType::FP8E5M2>(static_cast<ParamType*>(param_));                             \
    }

DEFINE_X86_PARAM(TransposeKernel, operators::TransposeParam, fp8_host::Transpose, "Transpose")
DEFINE_X86_PARAM(ConcatKernel, operators::ConcatParam, fp8_host::Concat, "Concat")
DEFINE_X86_PARAM(SplitKernel, operators::SplitParam, fp8_host::Split, "Split")
DEFINE_X86_PARAM(SliceKernel, operators::SliceParam, fp8_host::Slice, "Slice")
DEFINE_X86_PARAM(ReduceMeanKernel, operators::ReduceMeanParam, fp8_host::ReduceMean, "ReduceMean")
DEFINE_X86_PARAM(ReduceSumKernel, operators::ReduceSumParam, fp8_host::ReduceSum, "ReduceSum")
DEFINE_X86_PARAM(SoftmaxKernel, operators::SoftmaxParam, fp8_host::Softmax, "Softmax")
DEFINE_X86_PARAM(GatherKernel, operators::GatherParam, fp8_host::Gather, "Gather")
DEFINE_X86_PARAM(ExpandKernel, operators::ExpandParam, fp8_host::Expand, "Expand")
DEFINE_X86_PARAM(WhereKernel, operators::WhereParam, fp8_host::Where, "Where")
DEFINE_X86_PARAM(EqualKernel, operators::EqualParam, fp8_host::Equal, "Equal")
DEFINE_X86_PARAM(BatchNormalizationKernel, operators::BatchNormParam, fp8_host::BatchNormalization, "BatchNormalization")
DEFINE_X86_PARAM(ResizeKernel, operators::ResizeParam, fp8_host::Resize, "Resize")
DEFINE_X86_PARAM(ResizeConcatKernel, operators::ResizeConcatParam, fp8_host::ResizeConcat, "ResizeConcat")

template <DataType dtype>
int32_t AvgPool(operators::PoolParam* p) { return fp8_host::Pool<dtype, false>(p); }
template <DataType dtype>
int32_t MaxPool(operators::PoolParam* p) { return fp8_host::Pool<dtype, true>(p); }
DEFINE_X86_PARAM(AvgPoolKernel, operators::PoolParam, AvgPool, "AvgPool")
DEFINE_X86_PARAM(MaxPoolKernel, operators::PoolParam, MaxPool, "MaxPool")

#undef DEFINE_X86_PARAM
#undef DEFINE_X86_UNARY

template <>
int32_t Conv2DKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Conv2D::FP8E4M3");
    return ComputeQwenFp8DepthwiseConv2D<DataType::FP8E4M3>(
        static_cast<operators::Conv2dParam*>(param_));
}

template <>
int32_t Conv2DKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Conv2D::FP8E5M2");
    return ComputeQwenFp8DepthwiseConv2D<DataType::FP8E5M2>(
        static_cast<operators::Conv2dParam*>(param_));
}

void EnsureX86Fp8KernelsRegistered() { (void)g_x86_fp8_kernels_registered; }

}  // namespace kernel
}  // namespace feather
