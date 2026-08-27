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

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_fp8_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
#define REGISTER_COMMON_FP8(op, klass)                                                                    \
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP8E4M3, op, []() {                       \
        return std::make_unique<klass<DeviceType::COMMON, DataType::FP8E4M3>>();                       \
    });                                                                                                    \
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP8E5M2, op, []() {                       \
        return std::make_unique<klass<DeviceType::COMMON, DataType::FP8E5M2>>();                       \
    });

    REGISTER_COMMON_FP8("BatchNormalization", BatchNormalizationKernel)
    REGISTER_COMMON_FP8("Concat", ConcatKernel)
    REGISTER_COMMON_FP8("Conv2D", Conv2DKernel)
    REGISTER_COMMON_FP8("Cos", CosKernel)
    REGISTER_COMMON_FP8("Erf", ErfKernel)
    REGISTER_COMMON_FP8("Exp", ExpKernel)
    REGISTER_COMMON_FP8("Expand", ExpandKernel)
    REGISTER_COMMON_FP8("Flatten", FlattenKernel)
    REGISTER_COMMON_FP8("GlobalAveragePool", GlobalAveragePoolKernel)
    REGISTER_COMMON_FP8("Gather", GatherKernel)
    REGISTER_COMMON_FP8("Identity", IdentityKernel)
    REGISTER_COMMON_FP8("AvgPool", AvgPoolKernel)
    REGISTER_COMMON_FP8("MaxPool", MaxPoolKernel)
    REGISTER_COMMON_FP8("Neg", NegKernel)
    REGISTER_COMMON_FP8("Pow", PowKernel)
    REGISTER_COMMON_FP8("ReduceMean", ReduceMeanKernel)
    REGISTER_COMMON_FP8("ReduceSum", ReduceSumKernel)
    REGISTER_COMMON_FP8("ReLU", ReluKernel)
    REGISTER_COMMON_FP8("Reshape", ReshapeKernel)
    REGISTER_COMMON_FP8("Resize", ResizeKernel)
    REGISTER_COMMON_FP8("ResizeConcat", ResizeConcatKernel)
    REGISTER_COMMON_FP8("Sigmoid", SigmoidKernel)
    REGISTER_COMMON_FP8("SiLU", SiluKernel)
    REGISTER_COMMON_FP8("Sin", SinKernel)
    REGISTER_COMMON_FP8("Slice", SliceKernel)
    REGISTER_COMMON_FP8("Softmax", SoftmaxKernel)
    REGISTER_COMMON_FP8("Softplus", SoftplusKernel)
    REGISTER_COMMON_FP8("Split", SplitKernel)
    REGISTER_COMMON_FP8("Sqrt", SqrtKernel)
    REGISTER_COMMON_FP8("Tanh", TanhKernel)
    REGISTER_COMMON_FP8("Transpose", TransposeKernel)
    REGISTER_COMMON_FP8("Unsqueeze", UnsqueezeKernel)
    REGISTER_COMMON_FP8("Squeeze", SqueezeKernel)
    REGISTER_COMMON_FP8("Where", WhereKernel)
    REGISTER_COMMON_FP8("Equal", EqualKernel)
#undef REGISTER_COMMON_FP8
    return true;
}();

template <DataType dtype>
int32_t ComputeRelu(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::max(0.0f, value); });
}

template <DataType dtype>
int32_t ComputeSigmoid(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
}

template <DataType dtype>
int32_t ComputeSilu(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return value / (1.0f + std::exp(-value)); });
}

template <DataType dtype>
int32_t ComputeTanh(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::tanh(value); });
}

template <DataType dtype>
int32_t ComputeExp(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::exp(value); });
}

template <DataType dtype>
int32_t ComputeSin(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::sin(value); });
}

template <DataType dtype>
int32_t ComputeCos(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::cos(value); });
}

template <DataType dtype>
int32_t ComputeErf(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::erf(value); });
}

template <DataType dtype>
int32_t ComputeSqrt(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return std::sqrt(value); });
}

template <DataType dtype>
int32_t ComputeSoftplus(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) {
        return std::max(0.0f, value) + std::log1p(std::exp(-std::fabs(value)));
    });
}

template <DataType dtype>
int32_t ComputeNeg(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return -value; });
}

template <DataType dtype>
int32_t ComputePow(operators::PowParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr &&
        !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) return -1;
    operators::UnaryParam unary{};
    unary.input = param->input;
    unary.out = param->out;
    return fp8_host::Unary<dtype>(&unary, [exponent](float value) { return std::pow(value, exponent); });
}

template <DataType dtype>
int32_t ComputeIdentity(operators::UnaryParam* param) {
    return fp8_host::Unary<dtype>(param, [](float value) { return value; });
}

template <DataType dtype>
int32_t ComputeReshape(operators::ReshapeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    operators::AxesParam axes{};
    axes.input = param->input;
    axes.out = param->out;
    return fp8_host::Copy<dtype>(&axes);
}

template <DataType dtype>
int32_t ComputeUnsqueeze(operators::AxesParam* param) { return fp8_host::Copy<dtype>(param); }

template <DataType dtype>
int32_t ComputeSqueeze(operators::AxesParam* param) { return fp8_host::Copy<dtype>(param); }

template <DataType dtype>
int32_t ComputeFlatten(operators::FlattenParam* param) { return fp8_host::Flatten<dtype>(param); }

template <typename Kernel, DataType dtype, typename Param, int32_t (*Fn)(Param*)>
int32_t DispatchUnary(Param* param) { return Fn(param); }

}  // namespace

#define DEFINE_FP8_UNARY_KERNELS(Class, ParamType, Function, Label)                                      \
    template <>                                                                                            \
    int32_t Class<DeviceType::COMMON, DataType::FP8E4M3>::compute() {                                   \
        AutoTimer timer("Common::" Label "::FP8E4M3");                                                \
        return Function<DataType::FP8E4M3>(static_cast<ParamType*>(param_));                             \
    }                                                                                                     \
    template <>                                                                                            \
    int32_t Class<DeviceType::COMMON, DataType::FP8E5M2>::compute() {                                   \
        AutoTimer timer("Common::" Label "::FP8E5M2");                                                \
        return Function<DataType::FP8E5M2>(static_cast<ParamType*>(param_));                             \
    }

DEFINE_FP8_UNARY_KERNELS(ReluKernel, operators::UnaryParam, ComputeRelu, "ReLU")
DEFINE_FP8_UNARY_KERNELS(SigmoidKernel, operators::UnaryParam, ComputeSigmoid, "Sigmoid")
DEFINE_FP8_UNARY_KERNELS(SiluKernel, operators::UnaryParam, ComputeSilu, "SiLU")
DEFINE_FP8_UNARY_KERNELS(TanhKernel, operators::UnaryParam, ComputeTanh, "Tanh")
DEFINE_FP8_UNARY_KERNELS(ExpKernel, operators::UnaryParam, ComputeExp, "Exp")
DEFINE_FP8_UNARY_KERNELS(SinKernel, operators::UnaryParam, ComputeSin, "Sin")
DEFINE_FP8_UNARY_KERNELS(CosKernel, operators::UnaryParam, ComputeCos, "Cos")
DEFINE_FP8_UNARY_KERNELS(ErfKernel, operators::UnaryParam, ComputeErf, "Erf")
DEFINE_FP8_UNARY_KERNELS(SqrtKernel, operators::UnaryParam, ComputeSqrt, "Sqrt")
DEFINE_FP8_UNARY_KERNELS(SoftplusKernel, operators::UnaryParam, ComputeSoftplus, "Softplus")
DEFINE_FP8_UNARY_KERNELS(NegKernel, operators::UnaryParam, ComputeNeg, "Neg")

DEFINE_FP8_UNARY_KERNELS(PowKernel, operators::PowParam, ComputePow, "Pow")

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::Identity::FP8E4M3");
    return ComputeIdentity<DataType::FP8E4M3>(static_cast<operators::UnaryParam*>(param_));
}
template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::Identity::FP8E5M2");
    return ComputeIdentity<DataType::FP8E5M2>(static_cast<operators::UnaryParam*>(param_));
}

template <>
int32_t ReshapeKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() { return ComputeReshape<DataType::FP8E4M3>(static_cast<operators::ReshapeParam*>(param_)); }
template <>
int32_t ReshapeKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() { return ComputeReshape<DataType::FP8E5M2>(static_cast<operators::ReshapeParam*>(param_)); }
template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() { return ComputeUnsqueeze<DataType::FP8E4M3>(static_cast<operators::AxesParam*>(param_)); }
template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() { return ComputeUnsqueeze<DataType::FP8E5M2>(static_cast<operators::AxesParam*>(param_)); }
template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() { return ComputeSqueeze<DataType::FP8E4M3>(static_cast<operators::AxesParam*>(param_)); }
template <>
int32_t SqueezeKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() { return ComputeSqueeze<DataType::FP8E5M2>(static_cast<operators::AxesParam*>(param_)); }
template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() { return ComputeFlatten<DataType::FP8E4M3>(static_cast<operators::FlattenParam*>(param_)); }
template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() { return ComputeFlatten<DataType::FP8E5M2>(static_cast<operators::FlattenParam*>(param_)); }
template <>
int32_t GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    return fp8_host::GlobalAveragePool<DataType::FP8E4M3>(static_cast<operators::GlobalAveragePoolParam*>(param_));
}
template <>
int32_t GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    return fp8_host::GlobalAveragePool<DataType::FP8E5M2>(static_cast<operators::GlobalAveragePoolParam*>(param_));
}

#define DEFINE_FP8_PARAM_KERNELS(Class, ParamType, Function, Label)                                     \
    template <>                                                                                            \
    int32_t Class<DeviceType::COMMON, DataType::FP8E4M3>::compute() {                                   \
        AutoTimer timer("Common::" Label "::FP8E4M3");                                                \
        return Function<DataType::FP8E4M3>(static_cast<ParamType*>(param_));                             \
    }                                                                                                     \
    template <>                                                                                            \
    int32_t Class<DeviceType::COMMON, DataType::FP8E5M2>::compute() {                                   \
        AutoTimer timer("Common::" Label "::FP8E5M2");                                                \
        return Function<DataType::FP8E5M2>(static_cast<ParamType*>(param_));                             \
    }

DEFINE_FP8_PARAM_KERNELS(TransposeKernel, operators::TransposeParam, fp8_host::Transpose, "Transpose")
DEFINE_FP8_PARAM_KERNELS(ConcatKernel, operators::ConcatParam, fp8_host::Concat, "Concat")
DEFINE_FP8_PARAM_KERNELS(SplitKernel, operators::SplitParam, fp8_host::Split, "Split")
DEFINE_FP8_PARAM_KERNELS(SliceKernel, operators::SliceParam, fp8_host::Slice, "Slice")
DEFINE_FP8_PARAM_KERNELS(ReduceMeanKernel, operators::ReduceMeanParam, fp8_host::ReduceMean, "ReduceMean")
DEFINE_FP8_PARAM_KERNELS(ReduceSumKernel, operators::ReduceSumParam, fp8_host::ReduceSum, "ReduceSum")
DEFINE_FP8_PARAM_KERNELS(SoftmaxKernel, operators::SoftmaxParam, fp8_host::Softmax, "Softmax")
DEFINE_FP8_PARAM_KERNELS(GatherKernel, operators::GatherParam, fp8_host::Gather, "Gather")
DEFINE_FP8_PARAM_KERNELS(ExpandKernel, operators::ExpandParam, fp8_host::Expand, "Expand")
DEFINE_FP8_PARAM_KERNELS(WhereKernel, operators::WhereParam, fp8_host::Where, "Where")
DEFINE_FP8_PARAM_KERNELS(EqualKernel, operators::EqualParam, fp8_host::Equal, "Equal")
DEFINE_FP8_PARAM_KERNELS(BatchNormalizationKernel, operators::BatchNormParam, fp8_host::BatchNormalization, "BatchNormalization")
DEFINE_FP8_PARAM_KERNELS(Conv2DKernel, operators::Conv2dParam, fp8_host::Conv2D, "Conv2D")
DEFINE_FP8_PARAM_KERNELS(ResizeKernel, operators::ResizeParam, fp8_host::Resize, "Resize")
DEFINE_FP8_PARAM_KERNELS(ResizeConcatKernel, operators::ResizeConcatParam, fp8_host::ResizeConcat, "ResizeConcat")

template <DataType dtype>
int32_t ComputeAvgPool(operators::PoolParam* param) { return fp8_host::Pool<dtype, false>(param); }
template <DataType dtype>
int32_t ComputeMaxPool(operators::PoolParam* param) { return fp8_host::Pool<dtype, true>(param); }
DEFINE_FP8_PARAM_KERNELS(AvgPoolKernel, operators::PoolParam, ComputeAvgPool, "AvgPool")
DEFINE_FP8_PARAM_KERNELS(MaxPoolKernel, operators::PoolParam, ComputeMaxPool, "MaxPool")

#undef DEFINE_FP8_PARAM_KERNELS
#undef DEFINE_FP8_UNARY_KERNELS

void EnsureCommonFp8KernelsRegistered() { (void)g_common_fp8_kernels_registered; }

}  // namespace kernel
}  // namespace feather
