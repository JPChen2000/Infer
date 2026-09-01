#include "src/kernel/add.h"
#include "src/kernel/batch_normalization.h"
#include "src/kernel/cast.h"
#include "src/kernel/concat.h"
#include "src/kernel/constant_of_shape.h"
#include "src/kernel/cos.h"
#include "src/kernel/div.h"
#include "src/kernel/equal.h"
#include "src/kernel/erf.h"
#include "src/kernel/exp.h"
#include "src/kernel/expand.h"
#include "src/kernel/flatten.h"
#include "src/kernel/gather.h"
#include "src/kernel/global_average_pool.h"
#include "src/kernel/identity.h"
#include "src/kernel/mul.h"
#include "src/kernel/neg.h"
#include "src/kernel/pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/reduce_mean.h"
#include "src/kernel/reduce_sum.h"
#include "src/kernel/relu.h"
#include "src/kernel/reshape.h"
#include "src/kernel/resize.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/silu.h"
#include "src/kernel/sin.h"
#include "src/kernel/slice.h"
#include "src/kernel/softmax.h"
#include "src/kernel/softplus.h"
#include "src/kernel/split.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/sub.h"
#include "src/kernel/tanh.h"
#include "src/kernel/transpose.h"
#include "src/kernel/unsqueeze.h"
#include "src/kernel/squeeze.h"
#include "src/kernel/where.h"
#include "src/kernel/common/int8_standard_utils.h"
#include "src/operator/params.h"
#include "quant/quantization.h"
#include <array>
#include <cmath>
#include <cstdint>
#include "util/bf16.h"
#include "util/fp16.h"
#include "util/timer.h"
#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {
namespace {

using namespace int8_standard_detail;

enum class UnaryKind { kRelu, kNeg, kSigmoid, kSilu, kExp, kSqrt, kTanh, kErf, kSin, kCos, kSoftplus };

float ApplyUnary(UnaryKind kind, float value) {
    switch (kind) {
        case UnaryKind::kRelu: return std::max(value, 0.0f);
        case UnaryKind::kNeg: return -value;
        case UnaryKind::kSigmoid: return 1.0f / (1.0f + std::exp(-value));
        case UnaryKind::kSilu: return value / (1.0f + std::exp(-value));
        case UnaryKind::kExp: return std::exp(value);
        case UnaryKind::kSqrt: return std::sqrt(value);
        case UnaryKind::kTanh: return std::tanh(value);
        case UnaryKind::kErf: return std::erf(value);
        case UnaryKind::kSin: return std::sin(value);
        case UnaryKind::kCos: return std::cos(value);
        case UnaryKind::kSoftplus: return std::max(value, 0.0f) + std::log1p(std::exp(-std::fabs(value)));
    }
    return value;
}

float ApplyBinary(char kind, float lhs, float rhs) {
    switch (kind) {
        case '+': return lhs + rhs;
        case '-': return lhs - rhs;
        case '*': return lhs * rhs;
        case '/': return lhs / rhs;
        default: return std::numeric_limits<float>::quiet_NaN();
    }
}

bool CanUsePerTensorInt8Lookup(const Tensor* tensor, float* scale, int32_t* zero_point) {
    if (tensor == nullptr || scale == nullptr || zero_point == nullptr || !tensor->IsInitialized() ||
        tensor->data_type() != DataType::INT8) return false;
    const auto& quantization = tensor->quantization();
    if (!quantization.enabled || quantization.granularity != QuantizationGranularity::kPerTensor ||
        quantization.scales.size() > 1 || quantization.zero_points.size() > 1) return false;
    *scale = quantization.scale_at(0);
    *zero_point = quantization.zero_point_at(0);
    return std::isfinite(*scale) && *scale > 0.0f && *zero_point >= -128 && *zero_point <= 127;
}

bool BuildUnaryInt8Lookup(UnaryKind kind, float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          std::array<int8_t, 256>* lookup) {
    if (lookup == nullptr) return false;
    for (int32_t quantized = -128; quantized <= 127; ++quantized) {
        const float real_value =
            (static_cast<float>(quantized - input_zero_point)) * input_scale;
        const float transformed = ApplyUnary(kind, real_value);
        if (!std::isfinite(transformed)) return false;
        (*lookup)[static_cast<size_t>(quantized + 128)] =
            QuantizeInt8Value(transformed, output_scale, output_zero_point);
    }
    return true;
}

bool BuildBinaryInt8Lookup(char kind, float lhs_scale, int32_t lhs_zero_point,
                           float rhs_scale, int32_t rhs_zero_point,
                           float output_scale, int32_t output_zero_point,
                           std::array<int8_t, 256 * 256>* lookup) {
    if (lookup == nullptr) return false;
    for (int32_t lhs_quantized = -128; lhs_quantized <= 127; ++lhs_quantized) {
        const float lhs_value =
            (static_cast<float>(lhs_quantized - lhs_zero_point)) * lhs_scale;
        for (int32_t rhs_quantized = -128; rhs_quantized <= 127; ++rhs_quantized) {
            const float rhs_value =
                (static_cast<float>(rhs_quantized - rhs_zero_point)) * rhs_scale;
            const float transformed = ApplyBinary(kind, lhs_value, rhs_value);
            if (!std::isfinite(transformed)) return false;
            (*lookup)[static_cast<size_t>((lhs_quantized + 128) * 256 + rhs_quantized + 128)] =
                QuantizeInt8Value(transformed, output_scale, output_zero_point);
        }
    }
    return true;
}

int32_t RunUnaryInt8Lookup(operators::UnaryParam* param, UnaryKind kind) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->dims().data() != param->out->dims().data()) return -1;
    float input_scale = 1.0f;
    float output_scale = 1.0f;
    int32_t input_zero_point = 0;
    int32_t output_zero_point = 0;
    if (!CanUsePerTensorInt8Lookup(param->input.get(), &input_scale, &input_zero_point) ||
        !CanUsePerTensorInt8Lookup(param->out.get(), &output_scale, &output_zero_point)) return 1;
    std::array<int8_t, 256> lookup{};
    if (!BuildUnaryInt8Lookup(kind, input_scale, input_zero_point, output_scale, output_zero_point, &lookup)) return -1;
    const int8_t* input = param->input->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    const int64_t count = param->out->numel();
#if defined(FEATHER_WITH_OPENMP)
    const int workers = std::max(1, omp_get_max_threads());
    if (count >= 16384 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t index = 0; index < count; ++index) {
            output[index] = lookup[static_cast<size_t>(static_cast<int32_t>(input[index]) + 128)];
        }
        return 0;
    }
#endif
    for (int64_t index = 0; index < count; ++index) {
        output[index] = lookup[static_cast<size_t>(static_cast<int32_t>(input[index]) + 128)];
    }
    return 0;
}

int32_t RunBinaryInt8Lookup(operators::BinaryParam* param, char kind) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) return -1;
    float lhs_scale = 1.0f;
    float rhs_scale = 1.0f;
    float output_scale = 1.0f;
    int32_t lhs_zero_point = 0;
    int32_t rhs_zero_point = 0;
    int32_t output_zero_point = 0;
    if (!CanUsePerTensorInt8Lookup(param->lhs.get(), &lhs_scale, &lhs_zero_point) ||
        !CanUsePerTensorInt8Lookup(param->rhs.get(), &rhs_scale, &rhs_zero_point) ||
        !CanUsePerTensorInt8Lookup(param->out.get(), &output_scale, &output_zero_point)) return 1;
    const auto& output_dims = param->out->dims().data();
    const bool same_shape = param->lhs->dims().data() == output_dims && param->rhs->dims().data() == output_dims;
    const bool lhs_scalar = param->lhs->numel() == 1 && param->rhs->dims().data() == output_dims;
    const bool rhs_scalar = param->rhs->numel() == 1 && param->lhs->dims().data() == output_dims;
    if (!same_shape && !lhs_scalar && !rhs_scalar) return 1;
    std::array<int8_t, 256 * 256> lookup{};
    if (!BuildBinaryInt8Lookup(kind, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point,
                               output_scale, output_zero_point, &lookup)) return -1;
    const int8_t* lhs = param->lhs->data<int8_t>();
    const int8_t* rhs = param->rhs->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    const int64_t count = param->out->numel();
    const int32_t lhs_value = static_cast<int32_t>(lhs[0]) + 128;
    const int32_t rhs_value = static_cast<int32_t>(rhs[0]) + 128;
    auto write = [&](int64_t index) {
        const int32_t lhs_index = lhs_scalar ? lhs_value : static_cast<int32_t>(lhs[index]) + 128;
        const int32_t rhs_index = rhs_scalar ? rhs_value : static_cast<int32_t>(rhs[index]) + 128;
        output[index] = lookup[static_cast<size_t>(lhs_index * 256 + rhs_index)];
    };
#if defined(FEATHER_WITH_OPENMP)
    const int workers = std::max(1, omp_get_max_threads());
    if (count >= 16384 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t index = 0; index < count; ++index) write(index);
        return 0;
    }
#endif
    for (int64_t index = 0; index < count; ++index) write(index);
    return 0;
}

int32_t RunUnary(operators::UnaryParam* param, UnaryKind kind) {
    const int32_t fast_path = RunUnaryInt8Lookup(param, kind);
    if (fast_path != 1) return fast_path;
    return int8_standard_detail::Unary(param, [kind](float value) { return ApplyUnary(kind, value); });
}

int32_t RunBinary(operators::BinaryParam* param, char kind) {
    const int32_t fast_path = RunBinaryInt8Lookup(param, kind);
    if (fast_path != 1) return fast_path;
    return int8_standard_detail::Binary(param, [kind](float lhs, float rhs) {
        return ApplyBinary(kind, lhs, rhs);
    });
}

int32_t RunInt8MaxPool(operators::PoolParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::INT8 || param->out->data_type() != DataType::INT8 ||
        param->input->dims().size() != 4 || param->out->dims().size() != 4 || param->kernel_h <= 0 ||
        param->kernel_w <= 0 || param->stride_h <= 0 || param->stride_w <= 0 || param->input->dims()[0] != param->out->dims()[0] ||
        param->input->dims()[1] != param->out->dims()[1] ||
        NormalizeDataLayout(param->input->layout()) != DataLayout::NCHW ||
        NormalizeDataLayout(param->out->layout()) != DataLayout::NCHW) {
        return 1;
    }

    float input_scale = 1.0f;
    float output_scale = 1.0f;
    int32_t input_zero_point = 0;
    int32_t output_zero_point = 0;
    if (!CanUsePerTensorInt8Lookup(param->input.get(), &input_scale, &input_zero_point) ||
        !CanUsePerTensorInt8Lookup(param->out.get(), &output_scale, &output_zero_point)) {
        return 1;
    }

    const int64_t batches = param->input->dims()[0];
    const int64_t channels = param->input->dims()[1];
    const int64_t input_height = param->input->dims()[2];
    const int64_t input_width = param->input->dims()[3];
    const int64_t output_height = param->out->dims()[2];
    const int64_t output_width = param->out->dims()[3];
    const int8_t* input = param->input->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();

    const int64_t total_work_items = batches * channels * output_height * output_width;
    std::atomic<bool> failed{false};
    auto compute_output = [&](int64_t work_index) {
        int64_t remaining = work_index;
        const int64_t output_x = remaining % output_width;
        remaining /= output_width;
        const int64_t output_y = remaining % output_height;
        remaining /= output_height;
        const int64_t channel = remaining % channels;
        const int64_t batch = remaining / channels;

        int32_t max_quantized = -128;
        bool has_value = false;
        for (int64_t kernel_y = 0; kernel_y < param->kernel_h; ++kernel_y) {
            const int64_t input_y = output_y * param->stride_h + kernel_y - param->pad_h;
            if (input_y < 0 || input_y >= input_height) continue;
            for (int64_t kernel_x = 0; kernel_x < param->kernel_w; ++kernel_x) {
                const int64_t input_x = output_x * param->stride_w + kernel_x - param->pad_w;
                if (input_x < 0 || input_x >= input_width) continue;
                const int64_t input_offset =
                    ((batch * channels + channel) * input_height + input_y) * input_width + input_x;
                max_quantized = std::max(max_quantized, static_cast<int32_t>(input[input_offset]));
                has_value = true;
            }
        }
        if (!has_value) {
            failed.store(true, std::memory_order_relaxed);
            return;
        }
        const double real_value =
            (static_cast<double>(max_quantized) - static_cast<double>(input_zero_point)) * input_scale;
        const double quantized = real_value / static_cast<double>(output_scale) + output_zero_point;
        const double rounded = std::round(quantized);
        const int32_t clamped = static_cast<int32_t>(std::max(-128.0, std::min(127.0, rounded)));
        const int64_t output_offset =
            ((batch * channels + channel) * output_height + output_y) * output_width + output_x;
        output[output_offset] = static_cast<int8_t>(clamped);
    };

#if defined(FEATHER_WITH_OPENMP)
    const int workers = std::max(1, omp_get_max_threads());
    if (total_work_items >= 16384 && workers > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t work_index = 0; work_index < total_work_items; ++work_index) {
            compute_output(work_index);
        }
    } else
#endif
    {
        for (int64_t work_index = 0; work_index < total_work_items; ++work_index) {
            compute_output(work_index);
        }
    }
    return failed.load(std::memory_order_relaxed) ? 1 : 0;
}

int32_t RunBatchNorm(operators::BatchNormParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->input->dims().size() != 4 ||
        param->out->dims().data() != param->input->dims().data() || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr) return -1;
    ImageShape4D shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), &shape) ||
        param->scale->numel() != shape.c || param->bias->numel() != shape.c || param->mean->numel() != shape.c ||
        param->var->numel() != shape.c) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < shape.c; ++c) {
            const float scale = common_tensor_detail::ReadFloat(param->scale.get(), c);
            const float bias = common_tensor_detail::ReadFloat(param->bias.get(), c);
            const float mean = common_tensor_detail::ReadFloat(param->mean.get(), c);
            const float variance = common_tensor_detail::ReadFloat(param->var.get(), c);
            if (!(variance + param->epsilon > 0.0f)) return -1;
            const float inverse_std = 1.0f / std::sqrt(variance + param->epsilon);
            for (int64_t h = 0; h < shape.h; ++h) {
                for (int64_t w = 0; w < shape.w; ++w) {
                    const int64_t offset = OffsetForImage4D(param->input->layout(), n, c, h, w, shape.c, shape.h, shape.w);
                    const float value = (ReadReal(param->input.get(), offset, input_view) - mean) * inverse_std * scale + bias;
                    if (!WriteReal(param->out.get(), offset, value, output_view)) return -1;
                }
            }
        }
    }
    return 0;
}

int32_t RunPool(operators::PoolParam* param, bool maximum) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->kernel_h <= 0 || param->kernel_w <= 0 ||
        param->stride_h <= 0 || param->stride_w <= 0) return -1;
    if (maximum && param->input->data_type() == DataType::INT8) {
        const int32_t fast_path = RunInt8MaxPool(param);
        if (fast_path != 1) return fast_path;
    }
    const bool image = param->input->dims().size() == 4;
    ImageShape4D input_shape{};
    ImageShape4D output_shape{};
    if (image && (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), &input_shape) ||
                  !DecodeImageShape4D(param->out->dims().data(), NormalizeDataLayout(param->out->layout()), &output_shape))) return -1;
    const int64_t batches = image ? input_shape.n : 1;
    const int64_t channels = image ? input_shape.c : 1;
    const int64_t input_height = image ? input_shape.h : param->input->dims()[0];
    const int64_t input_width = image ? input_shape.w : param->input->dims()[1];
    const int64_t output_height = image ? output_shape.h : param->out->dims()[0];
    const int64_t output_width = image ? output_shape.w : param->out->dims()[1];
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    for (int64_t n = 0; n < batches; ++n) for (int64_t c = 0; c < channels; ++c) {
        for (int64_t oh = 0; oh < output_height; ++oh) for (int64_t ow = 0; ow < output_width; ++ow) {
            float result = maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
            int64_t count = 0;
            for (int64_t kh = 0; kh < param->kernel_h; ++kh) for (int64_t kw = 0; kw < param->kernel_w; ++kw) {
                const int64_t ih = oh * param->stride_h + kh - param->pad_h;
                const int64_t iw = ow * param->stride_w + kw - param->pad_w;
                if (ih < 0 || ih >= input_height || iw < 0 || iw >= input_width) continue;
                const int64_t input_offset = image ? OffsetForImage4D(param->input->layout(), n, c, ih, iw, channels, input_height, input_width) : ih * input_width + iw;
                const float value = ReadReal(param->input.get(), input_offset, input_view);
                if (maximum) result = std::max(result, value); else { result += value; ++count; }
            }
            if (!maximum) result = count == 0 ? 0.0f : result / static_cast<float>(count);
            const int64_t output_offset = image ? OffsetForImage4D(param->out->layout(), n, c, oh, ow, channels, output_height, output_width) : oh * output_width + ow;
            if (!WriteReal(param->out.get(), output_offset, result, output_view)) return -1;
        }
    }
    return 0;
}

int32_t RunGlobalAveragePool(operators::GlobalAveragePoolParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->input->dims().size() != 4) return -1;
    ImageShape4D input_shape{};
    ImageShape4D output_shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), NormalizeDataLayout(param->out->layout()), &output_shape) ||
        output_shape.n != input_shape.n || output_shape.c != input_shape.c || output_shape.h != 1 || output_shape.w != 1) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const float divisor = static_cast<float>(input_shape.h * input_shape.w);
    for (int64_t n = 0; n < input_shape.n; ++n) for (int64_t c = 0; c < input_shape.c; ++c) {
        float sum = 0.0f;
        for (int64_t h = 0; h < input_shape.h; ++h) for (int64_t w = 0; w < input_shape.w; ++w) {
            const int64_t offset = OffsetForImage4D(param->input->layout(), n, c, h, w, input_shape.c, input_shape.h, input_shape.w);
            sum += ReadReal(param->input.get(), offset, input_view);
        }
        const int64_t output_offset = OffsetForImage4D(param->out->layout(), n, c, 0, 0, output_shape.c, output_shape.h, output_shape.w);
        if (!WriteReal(param->out.get(), output_offset, sum / divisor, output_view)) return -1;
    }
    return 0;
}

int32_t RunConcat(operators::ConcatParam* param) {
    if (param == nullptr || param->inputs.size() < 2 || !Ready(param->out.get())) return -1;
    const auto output_dims = param->out->dims().data();
    int64_t axis = param->axis < 0 ? param->axis + static_cast<int64_t>(output_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int64_t>(output_dims.size())) return -1;
    const int64_t outer = std::accumulate(output_dims.begin(), output_dims.begin() + axis, int64_t{1}, std::multiplies<int64_t>());
    const int64_t inner = std::accumulate(output_dims.begin() + axis + 1, output_dims.end(), int64_t{1}, std::multiplies<int64_t>());
    View output_view;
    if (!BuildOutputView(param->out.get(), &output_view)) return -1;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        int64_t axis_offset = 0;
        for (const auto& input : param->inputs) {
            if (!Ready(input.get()) || input->dims().size() != output_dims.size()) return -1;
            View input_view;
            if (!BuildView(input.get(), &input_view)) return -1;
            const int64_t count = input->dims()[axis] * inner;
            const int64_t input_base = outer_index * count;
            const int64_t output_base = (outer_index * output_dims[axis] + axis_offset) * inner;
            for (int64_t index = 0; index < count; ++index) if (!WriteReal(param->out.get(), output_base + index, ReadReal(input.get(), input_base + index, input_view), output_view)) return -1;
            axis_offset += input->dims()[axis];
        }
    }
    return 0;
}

int32_t RunSplit(operators::SplitParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || param->outputs.empty()) return -1;
    const auto input_dims = param->input->dims().data();
    int64_t axis = param->axis < 0 ? param->axis + static_cast<int64_t>(input_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int64_t>(input_dims.size())) return -1;
    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + axis, int64_t{1}, std::multiplies<int64_t>());
    const int64_t inner = std::accumulate(input_dims.begin() + axis + 1, input_dims.end(), int64_t{1}, std::multiplies<int64_t>());
    View input_view;
    if (!BuildView(param->input.get(), &input_view)) return -1;
    for (const auto& output : param->outputs) if (!Ready(output.get())) return -1;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        int64_t axis_offset = 0;
        for (const auto& output : param->outputs) {
            View output_view;
            if (!BuildOutputView(output.get(), &output_view)) return -1;
            const int64_t count = output->dims()[axis] * inner;
            const int64_t input_base = (outer_index * input_dims[axis] + axis_offset) * inner;
            const int64_t output_base = outer_index * count;
            for (int64_t index = 0; index < count; ++index) if (!WriteReal(output.get(), output_base + index, ReadReal(param->input.get(), input_base + index, input_view), output_view)) return -1;
            axis_offset += output->dims()[axis];
        }
    }
    return 0;
}

int32_t RunTranspose(operators::TransposeParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->perm.size() != param->input->dims().size()) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto input_strides = Strides(param->input->dims().data());
    const auto output_strides = Strides(param->out->dims().data());
    if (input_strides.empty() || output_strides.empty()) return -1;
    std::vector<int64_t> output_coordinates;
    std::vector<int64_t> input_coordinates(param->input->dims().size(), 0);
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, param->out->dims().data(), output_strides, &output_coordinates);
        std::fill(input_coordinates.begin(), input_coordinates.end(), 0);
        for (size_t axis = 0; axis < param->perm.size(); ++axis) input_coordinates[static_cast<size_t>(param->perm[axis])] = output_coordinates[axis];
        int64_t input_index = 0;
        for (size_t axis = 0; axis < input_coordinates.size(); ++axis) input_index += input_coordinates[axis] * input_strides[axis];
        if (!WriteReal(param->out.get(), index, ReadReal(param->input.get(), input_index, input_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunSlice(operators::SliceParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get())) return -1;
    const auto input_dims = param->input->dims().data();
    int64_t axis = param->axis < 0 ? param->axis + static_cast<int64_t>(input_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int64_t>(input_dims.size())) return -1;
    int64_t start = param->start < 0 ? param->start + input_dims[axis] : param->start;
    start = std::max<int64_t>(0, std::min<int64_t>(input_dims[axis], start));
    int64_t end = param->end == std::numeric_limits<int32_t>::max() ? input_dims[axis] : param->end;
    if (end < 0) end += input_dims[axis];
    end = std::max<int64_t>(0, std::min<int64_t>(input_dims[axis], end));
    if (end < start || param->input->dims().size() != param->out->dims().size()) return -1;
    for (size_t dimension = 0; dimension < input_dims.size(); ++dimension) {
        const int64_t expected = dimension == static_cast<size_t>(axis) ? end - start : input_dims[dimension];
        if (param->out->dims()[dimension] != expected) return -1;
    }
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(param->out->dims().data());
    if (input_strides.empty() || output_strides.empty()) return -1;
    std::vector<int64_t> output_coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, param->out->dims().data(), output_strides, &output_coordinates);
        int64_t input_index = 0;
        for (size_t dimension = 0; dimension < output_coordinates.size(); ++dimension) {
            const int64_t coordinate = dimension == static_cast<size_t>(axis)
                                           ? output_coordinates[dimension] + start
                                           : output_coordinates[dimension];
            input_index += coordinate * input_strides[dimension];
        }
        if (!WriteReal(param->out.get(), index, ReadReal(param->input.get(), input_index, input_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunExpand(operators::ExpandParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->shape == nullptr ||
        (param->shape->data_type() != DataType::INT32 && param->shape->data_type() != DataType::INT64)) return -1;
    std::vector<int64_t> shape;
    for (int64_t index = 0; index < param->shape->numel(); ++index) shape.push_back(common_tensor_detail::ReadInteger(param->shape.get(), index));
    if (shape.size() < param->input->dims().size() || shape != param->out->dims().data()) return -1;
    const size_t gap = shape.size() - param->input->dims().size();
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        const int64_t input_dim = axis < gap ? 1 : param->input->dims()[axis - gap];
        if (shape[axis] <= 0 || (input_dim != 1 && input_dim != shape[axis])) return -1;
    }
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto output_strides = Strides(shape);
    const auto input_strides = Strides(param->input->dims().data());
    std::vector<int64_t> coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, shape, output_strides, &coordinates);
        const int64_t input_index = BroadcastOffset(coordinates, param->input->dims().data(), input_strides);
        if (!WriteReal(param->out.get(), index, ReadReal(param->input.get(), input_index, input_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunGather(operators::GatherParam* param) {
    if (param == nullptr || !Ready(param->data.get()) || param->indices == nullptr || !Ready(param->out.get()) ||
        (param->indices->data_type() != DataType::INT32 && param->indices->data_type() != DataType::INT64)) return -1;
    const auto data_dims = param->data->dims().data();
    const auto indices_dims = param->indices->dims().data();
    const int64_t rank = static_cast<int64_t>(data_dims.size());
    const int64_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    View data_view;
    View output_view;
    if (!BuildView(param->data.get(), &data_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto data_strides = Strides(data_dims);
    const auto indices_strides = Strides(indices_dims);
    const auto output_strides = Strides(param->out->dims().data());
    std::vector<int64_t> coordinates;
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        LinearToCoords(linear, param->out->dims().data(), output_strides, &coordinates);
        int64_t indices_index = 0;
        for (size_t i = 0; i < indices_dims.size(); ++i) indices_index += coordinates[static_cast<size_t>(axis) + i] * indices_strides[i];
        int64_t selected = common_tensor_detail::ReadIndex(param->indices.get(), indices_index);
        if (selected < 0) selected += data_dims[static_cast<size_t>(axis)];
        if (selected < 0 || selected >= data_dims[static_cast<size_t>(axis)]) return -1;
        int64_t data_index = 0;
        for (int64_t i = 0; i < axis; ++i) data_index += coordinates[static_cast<size_t>(i)] * data_strides[static_cast<size_t>(i)];
        data_index += selected * data_strides[static_cast<size_t>(axis)];
        for (int64_t i = axis + 1; i < rank; ++i) data_index += coordinates[static_cast<size_t>(i - 1 + static_cast<int64_t>(indices_dims.size()))] * data_strides[static_cast<size_t>(i)];
        if (!WriteReal(param->out.get(), linear, ReadReal(param->data.get(), data_index, data_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunWhere(operators::WhereParam* param) {
    if (param == nullptr || param->condition == nullptr || param->condition->data_type() != DataType::BOOL ||
        !Ready(param->x.get()) || !Ready(param->y.get()) || !Ready(param->out.get())) return -1;
    std::vector<int64_t> xy_dims;
    std::vector<int64_t> output_dims;
    if (!BroadcastShape(param->x->dims().data(), param->y->dims().data(), &xy_dims) ||
        !BroadcastShape(xy_dims, param->condition->dims().data(), &output_dims) ||
        output_dims != param->out->dims().data()) return -1;
    View x_view;
    View y_view;
    View output_view;
    if (!BuildView(param->x.get(), &x_view) || !BuildView(param->y.get(), &y_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto output_strides = Strides(output_dims);
    const auto condition_strides = Strides(param->condition->dims().data());
    const auto x_strides = Strides(param->x->dims().data());
    const auto y_strides = Strides(param->y->dims().data());
    std::vector<int64_t> coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, output_dims, output_strides, &coordinates);
        const int64_t condition_index = BroadcastOffset(coordinates, param->condition->dims().data(), condition_strides);
        const bool condition = param->condition->data<uint8_t>()[condition_index] != 0;
        const Tensor* source = condition ? param->x.get() : param->y.get();
        const View& source_view = condition ? x_view : y_view;
        const auto& source_dims = condition ? param->x->dims().data() : param->y->dims().data();
        const auto& source_strides = condition ? x_strides : y_strides;
        const int64_t source_index = BroadcastOffset(coordinates, source_dims, source_strides);
        if (!WriteReal(param->out.get(), index, ReadReal(source, source_index, source_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunResize(operators::ResizeParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || !Ready(param->out.get()) || param->scales.size() != param->input->dims().size()) return -1;
    View input_view;
    View output_view;
    if (!BuildView(param->input.get(), &input_view) || !BuildOutputView(param->out.get(), &output_view)) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    std::vector<int64_t> coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, output_dims, output_strides, &coordinates);
        int64_t input_index = 0;
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            const int64_t coordinate = std::max<int64_t>(0, std::min<int64_t>(input_dims[axis] - 1, static_cast<int64_t>(coordinates[axis] / param->scales[axis])));
            input_index += coordinate * input_strides[axis];
        }
        if (!WriteReal(param->out.get(), index, ReadReal(param->input.get(), input_index, input_view), output_view)) return -1;
    }
    return 0;
}

int32_t RunEqual(operators::EqualParam* param) {
    if (param == nullptr || !Ready(param->lhs.get()) || !Ready(param->rhs.get()) || param->out == nullptr ||
        param->out->data_type() != DataType::BOOL || !param->out->IsInitialized()) return -1;
    View lhs_view;
    View rhs_view;
    if (!BuildView(param->lhs.get(), &lhs_view) || !BuildView(param->rhs.get(), &rhs_view)) return -1;
    std::vector<int64_t> output_dims;
    if (!BroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) || output_dims != param->out->dims().data()) return -1;
    const auto output_strides = Strides(output_dims);
    const auto lhs_strides = Strides(param->lhs->dims().data());
    const auto rhs_strides = Strides(param->rhs->dims().data());
    auto* result = param->out->mutable_data<uint8_t>();
    std::vector<int64_t> coordinates;
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        LinearToCoords(index, output_dims, output_strides, &coordinates);
        result[index] = ReadReal(param->lhs.get(), BroadcastOffset(coordinates, param->lhs->dims().data(), lhs_strides), lhs_view) ==
                                ReadReal(param->rhs.get(), BroadcastOffset(coordinates, param->rhs->dims().data(), rhs_strides), rhs_view);
    }
    param->out->set_data_type(DataType::BOOL);
    return 0;
}

int32_t RunConstantOfShape(operators::ConstantOfShapeParam* param) {
    if (param == nullptr || param->shape == nullptr || param->out == nullptr || param->output_type != DataType::INT8 ||
        !param->out->IsInitialized() || param->shape->numel() <= 0) return -1;
    std::vector<int64_t> dims;
    for (int64_t index = 0; index < param->shape->numel(); ++index) dims.push_back(common_tensor_detail::ReadInteger(param->shape.get(), index));
    if (dims != param->out->dims().data()) return -1;
    View output_view;
    if (!BuildOutputView(param->out.get(), &output_view)) return -1;
    const float value = param->use_float_value ? param->float_value : static_cast<float>(param->int_value);
    for (int64_t index = 0; index < param->out->numel(); ++index) if (!WriteReal(param->out.get(), index, value, output_view)) return -1;
    return 0;
}

int32_t RunCast(operators::CastParam* param) {
    if (param == nullptr || !Ready(param->input.get()) || param->out == nullptr || param->out->dims().data() != param->input->dims().data() ||
        param->out->data_type() != param->to || !param->out->IsInitialized()) return -1;
    View input_view;
    if (!BuildView(param->input.get(), &input_view)) return -1;
    for (int64_t index = 0; index < param->input->numel(); ++index) {
        const float value = ReadReal(param->input.get(), index, input_view);
        switch (param->to) {
            case DataType::INT8: { View output_view; if (!BuildOutputView(param->out.get(), &output_view) || !WriteReal(param->out.get(), index, value, output_view)) return -1; break; }
            case DataType::FP32: param->out->mutable_data<float>()[index] = value; break;
            case DataType::FP16: param->out->mutable_data<uint16_t>()[index] = FloatToHalf(value); break;
            case DataType::BF16: param->out->mutable_data<BFloat16>()[index].bits = FloatToBFloat16(value); break;
            case DataType::INT32: param->out->mutable_data<int32_t>()[index] = static_cast<int32_t>(value); break;
            case DataType::INT64: param->out->mutable_data<int64_t>()[index] = static_cast<int64_t>(value); break;
            case DataType::UINT8: param->out->mutable_data<uint8_t>()[index] = static_cast<uint8_t>(value); break;
            case DataType::BOOL: param->out->mutable_data<uint8_t>()[index] = value != 0.0f; break;
            default: return -1;
        }
    }
    return 0;
}

template <DeviceType Device> int32_t RegisterStandardInt8Kernels() {
    auto& dispatcher = KernelDispatcher::instance();
#define REG(OP, TYPE) dispatcher.registerKernel(Device, DataType::INT8, OP, []() { return std::make_unique<TYPE<Device, DataType::INT8>>(); })
    REG("Add", AddKernel); REG("Sub", SubKernel); REG("Mul", MulKernel); REG("Div", DivKernel);
    REG("ReLU", ReluKernel); REG("Neg", NegKernel); REG("Sigmoid", SigmoidKernel); REG("SiLU", SiluKernel);
    REG("Exp", ExpKernel); REG("Sqrt", SqrtKernel); REG("Tanh", TanhKernel); REG("Erf", ErfKernel);
    REG("Sin", SinKernel); REG("Cos", CosKernel); REG("Softplus", SoftplusKernel); REG("Pow", PowKernel);
    REG("BatchNormalization", BatchNormalizationKernel); REG("AvgPool", AvgPoolKernel); REG("MaxPool", MaxPoolKernel);
    REG("GlobalAveragePool", GlobalAveragePoolKernel); REG("ReduceSum", ReduceSumKernel); REG("ReduceMean", ReduceMeanKernel);
    REG("Identity", IdentityKernel); REG("Reshape", ReshapeKernel); REG("Flatten", FlattenKernel);
    REG("Transpose", TransposeKernel); REG("Squeeze", SqueezeKernel); REG("Unsqueeze", UnsqueezeKernel);
    REG("Slice", SliceKernel); REG("Split", SplitKernel); REG("Concat", ConcatKernel); REG("Expand", ExpandKernel);
    REG("Gather", GatherKernel); REG("Where", WhereKernel); REG("Resize", ResizeKernel); REG("Softmax", SoftmaxKernel);
    REG("Equal", EqualKernel); REG("Cast", CastKernel); REG("ConstantOfShape", ConstantOfShapeKernel);
#undef REG
    return 0;
}

}  // namespace

#define DEFINE_UNARY(KERNEL, KIND) \
template <> int32_t KERNEL<DeviceType::COMMON, DataType::INT8>::compute() { \
    AutoTimer timer("Common::" #KERNEL "::INT8"); \
    return RunUnary(static_cast<operators::UnaryParam*>(param_), UnaryKind::KIND); \
} \
template <> int32_t KERNEL<DeviceType::X86, DataType::INT8>::compute() { \
    AutoTimer timer("X86::" #KERNEL "::INT8"); \
    return RunUnary(static_cast<operators::UnaryParam*>(param_), UnaryKind::KIND); \
}

DEFINE_UNARY(ReluKernel, kRelu)
DEFINE_UNARY(NegKernel, kNeg)
DEFINE_UNARY(SigmoidKernel, kSigmoid)
DEFINE_UNARY(SiluKernel, kSilu)
DEFINE_UNARY(ExpKernel, kExp)
DEFINE_UNARY(SqrtKernel, kSqrt)
DEFINE_UNARY(TanhKernel, kTanh)
DEFINE_UNARY(ErfKernel, kErf)
DEFINE_UNARY(SinKernel, kSin)
DEFINE_UNARY(CosKernel, kCos)
DEFINE_UNARY(SoftplusKernel, kSoftplus)
#undef DEFINE_UNARY

#define DEFINE_BINARY(KERNEL, OP) \
template <> int32_t KERNEL<DeviceType::COMMON, DataType::INT8>::compute() { \
    AutoTimer timer("Common::" #KERNEL "::INT8"); \
    return RunBinary(static_cast<operators::BinaryParam*>(param_), OP); \
} \
template <> int32_t KERNEL<DeviceType::X86, DataType::INT8>::compute() { \
    AutoTimer timer("X86::" #KERNEL "::INT8"); \
    return RunBinary(static_cast<operators::BinaryParam*>(param_), OP); \
}

DEFINE_BINARY(AddKernel, '+')
DEFINE_BINARY(SubKernel, '-')
DEFINE_BINARY(MulKernel, '*')
DEFINE_BINARY(DivKernel, '/')
#undef DEFINE_BINARY

#define DEFINE_COPY(KERNEL, PARAM) \
template <> int32_t KERNEL<DeviceType::COMMON, DataType::INT8>::compute() { \
    AutoTimer timer("Common::" #KERNEL "::INT8"); \
    auto* p = static_cast<operators::PARAM*>(param_); \
    return p == nullptr ? -1 : Copy(p->input.get(), p->out.get()); \
} \
template <> int32_t KERNEL<DeviceType::X86, DataType::INT8>::compute() { \
    AutoTimer timer("X86::" #KERNEL "::INT8"); \
    auto* p = static_cast<operators::PARAM*>(param_); \
    return p == nullptr ? -1 : Copy(p->input.get(), p->out.get()); \
}

DEFINE_COPY(IdentityKernel, UnaryParam)
DEFINE_COPY(ReshapeKernel, ReshapeParam)
DEFINE_COPY(FlattenKernel, FlattenParam)
DEFINE_COPY(SqueezeKernel, AxesParam)
DEFINE_COPY(UnsqueezeKernel, AxesParam)
#undef DEFINE_COPY

#define DEFINE_SIMPLE(KERNEL, PARAM, FUNCTION) \
template <> int32_t KERNEL<DeviceType::COMMON, DataType::INT8>::compute() { \
    AutoTimer timer("Common::" #KERNEL "::INT8"); \
    return FUNCTION(static_cast<operators::PARAM*>(param_)); \
} \
template <> int32_t KERNEL<DeviceType::X86, DataType::INT8>::compute() { \
    AutoTimer timer("X86::" #KERNEL "::INT8"); \
    return FUNCTION(static_cast<operators::PARAM*>(param_)); \
}

DEFINE_SIMPLE(BatchNormalizationKernel, BatchNormParam, RunBatchNorm)
DEFINE_SIMPLE(AvgPoolKernel, PoolParam, [](operators::PoolParam* p) { return RunPool(p, false); })
DEFINE_SIMPLE(MaxPoolKernel, PoolParam, [](operators::PoolParam* p) { return RunPool(p, true); })
DEFINE_SIMPLE(GlobalAveragePoolKernel, GlobalAveragePoolParam, RunGlobalAveragePool)
DEFINE_SIMPLE(ReduceSumKernel, ReduceSumParam, [](operators::ReduceSumParam* p) { return Reduce(p, false); })
DEFINE_SIMPLE(ReduceMeanKernel, ReduceMeanParam, ReduceMean)
DEFINE_SIMPLE(TransposeKernel, TransposeParam, RunTranspose)
DEFINE_SIMPLE(SliceKernel, SliceParam, RunSlice)
DEFINE_SIMPLE(ExpandKernel, ExpandParam, RunExpand)
DEFINE_SIMPLE(GatherKernel, GatherParam, RunGather)
DEFINE_SIMPLE(WhereKernel, WhereParam, RunWhere)
template <> int32_t ResizeKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::ResizeKernel::INT8");
    return RunResize(static_cast<operators::ResizeParam*>(param_));
}
DEFINE_SIMPLE(EqualKernel, EqualParam, RunEqual)
DEFINE_SIMPLE(CastKernel, CastParam, RunCast)
DEFINE_SIMPLE(ConstantOfShapeKernel, ConstantOfShapeParam, RunConstantOfShape)
DEFINE_SIMPLE(SoftmaxKernel, SoftmaxParam, Softmax)
template <> int32_t ConcatKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::ConcatKernel::INT8");
    return RunConcat(static_cast<operators::ConcatParam*>(param_));
}
DEFINE_SIMPLE(SplitKernel, SplitParam, RunSplit)
#undef DEFINE_SIMPLE

int32_t RunPow(operators::PowParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    if (param->exponent_tensor != nullptr) {
        if (!param->exponent_tensor->IsInitialized() || param->exponent_tensor->numel() != 1) return -1;
        param->exponent = common_tensor_detail::ReadFloat(param->exponent_tensor.get(), 0);
    }
    if (!std::isfinite(param->exponent)) return -1;
    operators::UnaryParam unary;
    unary.input = param->input;
    unary.out = param->out;
    return Unary(&unary, [param](float value) { return std::pow(value, param->exponent); });
}

template <> int32_t PowKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::Pow::INT8");
    return RunPow(static_cast<operators::PowParam*>(param_));
}

template <> int32_t PowKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::Pow::INT8");
    return RunPow(static_cast<operators::PowParam*>(param_));
}

void EnsureStandardCommonInt8KernelsRegistered() { static const int registered = RegisterStandardInt8Kernels<DeviceType::COMMON>(); (void)registered; }
void EnsureStandardX86Int8KernelsRegistered() { static const int registered = RegisterStandardInt8Kernels<DeviceType::X86>(); (void)registered; }

namespace {
const bool g_standard_int8_registered = []() {
    EnsureStandardCommonInt8KernelsRegistered();
    EnsureStandardX86Int8KernelsRegistered();
    return true;
}();
}

}  // namespace kernel
}  // namespace feather
