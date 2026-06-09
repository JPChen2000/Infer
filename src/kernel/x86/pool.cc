#include "src/kernel/pool.h"

#include <algorithm>
#include <limits>

#include "src/kernel/common/kernel_io.h"
#include "util/fp16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

template <typename T>
inline float ValueToFloat(T value) {
    return static_cast<float>(value);
}

template <>
inline float ValueToFloat<uint16_t>(uint16_t value) {
    return HalfToFloat(value);
}

template <typename T>
inline T FloatToValue(float value) {
    return static_cast<T>(value);
}

template <>
inline uint16_t FloatToValue<uint16_t>(float value) {
    return FloatToHalf(value);
}

template <DataType dtype, bool is_max>
int32_t ComputePoolFallback(feather::operators::PoolParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const bool is_4d = param->input->dims().size() == 4;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (is_4d &&
        (!DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
         !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape))) {
        return -1;
    }
    const int batch = is_4d ? static_cast<int>(input_shape.n) : 1;
    const int channels = is_4d ? static_cast<int>(input_shape.c) : 1;
    const int in_h = is_4d ? static_cast<int>(input_shape.h) : static_cast<int>(param->input->dims()[0]);
    const int in_w = is_4d ? static_cast<int>(input_shape.w) : static_cast<int>(param->input->dims()[1]);
    const int out_h = is_4d ? static_cast<int>(output_shape.h) : static_cast<int>(param->out->dims()[0]);
    const int out_w = is_4d ? static_cast<int>(output_shape.w) : static_cast<int>(param->out->dims()[1]);

    param->out->set_data_type(dtype);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float value = is_max ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < param->kernel_h; ++kh) {
                        for (int kw = 0; kw < param->kernel_w; ++kw) {
                            const int ih = oh * param->stride_h + kh - param->pad_h;
                            const int iw = ow * param->stride_w + kw - param->pad_w;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                continue;
                            }
                            const int64_t input_offset = is_4d
                                ? OffsetForImage4D(layout, n, c, ih, iw, channels, in_h, in_w)
                                : static_cast<int64_t>(ih) * in_w + iw;
                            const float input_value = TensorIO<dtype>::Read(param->input.get(), input_offset);
                            if (is_max) {
                                value = std::max(value, input_value);
                            } else {
                                value += input_value;
                                ++count;
                            }
                        }
                    }
                    if (!is_max) {
                        value = count == 0 ? 0.0f : value / static_cast<float>(count);
                    }
                    const int64_t output_offset = is_4d
                        ? OffsetForImage4D(layout, n, c, oh, ow, channels, out_h, out_w)
                        : static_cast<int64_t>(oh) * out_w + ow;
                    TensorIO<dtype>::Write(param->out.get(), output_offset, value);
                }
            }
        }
    }
    return 0;
}

template <typename T, bool is_max>
int32_t ComputePoolRaw(feather::operators::PoolParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const bool is_4d = param->input->dims().size() == 4;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (is_4d &&
        (!DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
         !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape))) {
        return -1;
    }
    const int batch = is_4d ? static_cast<int>(input_shape.n) : 1;
    const int channels = is_4d ? static_cast<int>(input_shape.c) : 1;
    const int in_h = is_4d ? static_cast<int>(input_shape.h) : static_cast<int>(param->input->dims()[0]);
    const int in_w = is_4d ? static_cast<int>(input_shape.w) : static_cast<int>(param->input->dims()[1]);
    const int out_h = is_4d ? static_cast<int>(output_shape.h) : static_cast<int>(param->out->dims()[0]);
    const int out_w = is_4d ? static_cast<int>(output_shape.w) : static_cast<int>(param->out->dims()[1]);

    const T* input = param->input->data<T>();
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                const int h_base = oh * param->stride_h - param->pad_h;
                for (int ow = 0; ow < out_w; ++ow) {
                    const int w_base = ow * param->stride_w - param->pad_w;
                    float value = is_max ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < param->kernel_h; ++kh) {
                        const int ih = h_base + kh;
                        if (ih < 0 || ih >= in_h) {
                            continue;
                        }
                        for (int kw = 0; kw < param->kernel_w; ++kw) {
                            const int iw = w_base + kw;
                            if (iw < 0 || iw >= in_w) {
                                continue;
                            }
                            const int64_t input_offset = is_4d
                                ? OffsetForImage4D(layout, n, c, ih, iw, channels, in_h, in_w)
                                : static_cast<int64_t>(ih) * in_w + iw;
                            const float current = ValueToFloat(input[input_offset]);
                            if (is_max) {
                                value = std::max(value, current);
                            } else {
                                value += current;
                                ++count;
                            }
                        }
                    }
                    if (!is_max) {
                        value = count == 0 ? 0.0f : value / static_cast<float>(count);
                    }
                    const int64_t output_offset = is_4d
                        ? OffsetForImage4D(layout, n, c, oh, ow, channels, out_h, out_w)
                        : static_cast<int64_t>(oh) * out_w + ow;
                    output[output_offset] = FloatToValue<T>(value);
                }
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t AvgPoolKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::AvgPool::FP32");
    auto* param = static_cast<feather::operators::PoolParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputePoolFallback<DataType::FP32, false>(param);
    }
    return ComputePoolRaw<float, false>(param, DataType::FP32);
}

template <>
int32_t AvgPoolKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::AvgPool::FP16");
    auto* param = static_cast<feather::operators::PoolParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputePoolFallback<DataType::FP16, false>(param);
    }
    return ComputePoolRaw<uint16_t, false>(param, DataType::FP16);
}

template <>
int32_t MaxPoolKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::MaxPool::FP32");
    auto* param = static_cast<feather::operators::PoolParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputePoolFallback<DataType::FP32, true>(param);
    }
    return ComputePoolRaw<float, true>(param, DataType::FP32);
}

template <>
int32_t MaxPoolKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::MaxPool::FP16");
    auto* param = static_cast<feather::operators::PoolParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputePoolFallback<DataType::FP16, true>(param);
    }
    return ComputePoolRaw<uint16_t, true>(param, DataType::FP16);
}

void EnsureX86PoolKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "AvgPool",
            []() { return std::make_unique<AvgPoolKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "AvgPool",
            []() { return std::make_unique<AvgPoolKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MaxPool",
            []() { return std::make_unique<MaxPoolKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "MaxPool",
            []() { return std::make_unique<MaxPoolKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
