#include "src/kernel/resize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

template <DataType dtype>
int32_t ComputeResizeFallback(feather::operators::ResizeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size()) {
        return -1;
    }

    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);

    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }

        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            const float scale = param->scales[axis];
            int64_t coord = static_cast<int64_t>(static_cast<double>(out_coords[axis]) / scale);
            coord = std::max<int64_t>(0, std::min<int64_t>(coord, in_dims[axis] - 1));
            input_offset += coord * in_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <typename T>
int32_t ComputeResizeRaw(feather::operators::ResizeParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size()) {
        return -1;
    }

    const T* input = param->input->data<T>();
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);

    if (in_dims.size() == 4) {
        ImageShape4D input_shape;
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(in_dims, param->input->layout(), &input_shape) ||
            !DecodeImageShape4D(out_dims, param->out->layout(), &output_shape)) {
            return -1;
        }
        const DataLayout layout = NormalizeDataLayout(param->input->layout());
        const int64_t batch = input_shape.n;
        const int64_t channels = input_shape.c;
        const int64_t in_h = input_shape.h;
        const int64_t in_w = input_shape.w;
        const int64_t out_h = output_shape.h;
        const int64_t out_w = output_shape.w;
        const float scale_h = param->scales[static_cast<size_t>(HeightAxisForLayout(layout))];
        const float scale_w = param->scales[static_cast<size_t>(WidthAxisForLayout(layout))];
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    const int64_t ih = std::max<int64_t>(
                        0, std::min<int64_t>(static_cast<int64_t>(static_cast<double>(oh) / scale_h), in_h - 1));
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        const int64_t iw = std::max<int64_t>(
                            0, std::min<int64_t>(static_cast<int64_t>(static_cast<double>(ow) / scale_w), in_w - 1));
                        const int64_t input_offset = OffsetForImage4D(layout, n, c, ih, iw, channels, in_h, in_w);
                        const int64_t output_offset = OffsetForImage4D(layout, n, c, oh, ow, channels, out_h, out_w);
                        output[output_offset] = input[input_offset];
                    }
                }
            }
        }
        return 0;
    }

    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            const float scale = param->scales[axis];
            int64_t coord = static_cast<int64_t>(static_cast<double>(out_coords[axis]) / scale);
            coord = std::max<int64_t>(0, std::min<int64_t>(coord, in_dims[axis] - 1));
            input_offset += coord * in_strides[axis];
        }
        output[linear] = input[input_offset];
    }
    return 0;
}

// YOLOv5 uses NCHW nearest-neighbor resize with integral scale factors. The
// generic INT8 path used to redo coordinate division and floating-point
// requantization for every output element. Handle this common case by copying
// source rows and, when scales differ, applying one 256-entry byte lookup.
// Return 1 when the shape is not covered so the caller can use its fallback.
int32_t ComputeResizeInt8NchwNearest(feather::operators::ResizeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::INT8 || param->out->data_type() != DataType::INT8 ||
        param->input->dims().size() != 4 || param->out->dims().size() != 4 || param->scales.size() != 4 ||
        NormalizeDataLayout(param->input->layout()) != DataLayout::NCHW ||
        NormalizeDataLayout(param->out->layout()) != DataLayout::NCHW) {
        return 1;
    }

    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return 1;
    }
    if (input_shape.n <= 0 || input_shape.c <= 0 || input_shape.h <= 0 || input_shape.w <= 0 ||
        output_shape.n != input_shape.n || output_shape.c != input_shape.c) {
        return 1;
    }

    const float scale_h = param->scales[2];
    const float scale_w = param->scales[3];
    if (!std::isfinite(scale_h) || !std::isfinite(scale_w) || scale_h < 1.0f || scale_w < 1.0f ||
        std::fabs(param->scales[0] - 1.0f) > 1e-6f || std::fabs(param->scales[1] - 1.0f) > 1e-6f) {
        return 1;
    }
    const int64_t scale_h_int = static_cast<int64_t>(std::llround(scale_h));
    const int64_t scale_w_int = static_cast<int64_t>(std::llround(scale_w));
    if (scale_h_int <= 0 || scale_w_int <= 0 || std::fabs(scale_h - static_cast<float>(scale_h_int)) > 1e-6f ||
        std::fabs(scale_w - static_cast<float>(scale_w_int)) > 1e-6f ||
        output_shape.h != input_shape.h * scale_h_int || output_shape.w != input_shape.w * scale_w_int) {
        return 1;
    }

    const auto& input_q = param->input->quantization();
    const auto& output_q = param->out->quantization();
    if (!input_q.enabled || !output_q.enabled ||
        input_q.granularity != QuantizationGranularity::kPerTensor ||
        output_q.granularity != QuantizationGranularity::kPerTensor) {
        return 1;
    }
    const float input_scale = input_q.scale_at(0);
    const float output_scale = output_q.scale_at(0);
    if (!std::isfinite(input_scale) || input_scale <= 0.0f || !std::isfinite(output_scale) || output_scale <= 0.0f) {
        return -1;
    }
    const int32_t input_zero = input_q.zero_point_at(0);
    const int32_t output_zero = output_q.zero_point_at(0);
    const bool same_quantization = input_scale == output_scale && input_zero == output_zero;
    std::array<int8_t, 256> lookup{};
    if (!same_quantization) {
        for (int32_t value = -128; value <= 127; ++value) {
            const double transformed =
                (static_cast<double>(value - input_zero) * static_cast<double>(input_scale)) /
                    static_cast<double>(output_scale) + static_cast<double>(output_zero);
            if (!std::isfinite(transformed)) return -1;
            const double rounded = std::round(transformed);
            lookup[static_cast<size_t>(value + 128)] =
                static_cast<int8_t>(std::max(-128.0, std::min(127.0, rounded)));
        }
    }

    const int8_t* input = param->input->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    const int64_t planes = input_shape.n * input_shape.c;
    const int64_t rows = planes * input_shape.h;
    auto copy_row = [&](int64_t row_index) {
        const int64_t plane = row_index / input_shape.h;
        const int64_t input_y = row_index % input_shape.h;
        const int8_t* source = input + (plane * input_shape.h + input_y) * input_shape.w;
        int8_t* destination = output + plane * output_shape.h * output_shape.w +
                              input_y * scale_h_int * output_shape.w;
        for (int64_t repeat_y = 0; repeat_y < scale_h_int; ++repeat_y) {
            int8_t* row = destination + repeat_y * output_shape.w;
            if (scale_w_int == 1 && same_quantization) {
                std::memcpy(row, source, static_cast<size_t>(input_shape.w));
                continue;
            }
            for (int64_t input_x = 0; input_x < input_shape.w; ++input_x) {
                const int8_t value = same_quantization
                                         ? source[input_x]
                                         : lookup[static_cast<size_t>(static_cast<int32_t>(source[input_x]) + 128)];
                for (int64_t repeat_x = 0; repeat_x < scale_w_int; ++repeat_x) {
                    row[input_x * scale_w_int + repeat_x] = value;
                }
            }
        }
    };

#if defined(FEATHER_WITH_OPENMP)
    if (param->out->numel() >= 32768 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int64_t row_index = 0; row_index < rows; ++row_index) copy_row(row_index);
        return 0;
    }
#endif
    for (int64_t row_index = 0; row_index < rows; ++row_index) copy_row(row_index);
    return 0;
}

}  // namespace

template <>
int32_t ResizeKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Resize::FP32");
    auto* param = static_cast<feather::operators::ResizeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeResizeFallback<DataType::FP32>(param);
    }
    return ComputeResizeRaw<float>(param, DataType::FP32);
}

template <>
int32_t ResizeKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Resize::FP16");
    auto* param = static_cast<feather::operators::ResizeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeResizeFallback<DataType::FP16>(param);
    }
    return ComputeResizeRaw<uint16_t>(param, DataType::FP16);
}

template <>
int32_t ResizeKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::Resize::INT8");
    auto* param = static_cast<feather::operators::ResizeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::INT8 || param->out->data_type() != DataType::INT8 ||
        param->input->dims().size() != param->out->dims().size() ||
        param->input->dims().size() != param->scales.size()) return -1;
    const int32_t fast_status = ComputeResizeInt8NchwNearest(param);
    if (fast_status != 1) return fast_status;
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    const auto& input_q = param->input->quantization();
    const auto& output_q = param->out->quantization();
    const float input_scale = input_q.scale_at(0);
    const float output_scale = output_q.scale_at(0);
    const int32_t input_zero = input_q.zero_point_at(0);
    const int32_t output_zero = output_q.zero_point_at(0);
    if (!input_q.enabled || !output_q.enabled || !std::isfinite(input_scale) || input_scale <= 0.0f ||
        !std::isfinite(output_scale) || output_scale <= 0.0f) return -1;
    const int8_t* input = param->input->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            const int64_t out_coord = remaining / out_strides[axis];
            remaining %= out_strides[axis];
            const float scale = param->scales[axis];
            if (!(scale > 0.0f) || !std::isfinite(scale)) return -1;
            int64_t in_coord = static_cast<int64_t>(static_cast<double>(out_coord) / scale);
            in_coord = std::max<int64_t>(0, std::min<int64_t>(in_coord, in_dims[axis] - 1));
            input_offset += in_coord * in_strides[axis];
        }
        const int32_t real_q = static_cast<int32_t>(input[input_offset]) - input_zero;
        const double real_value = static_cast<double>(real_q) * static_cast<double>(input_scale);
        const double transformed = real_value / static_cast<double>(output_scale) + output_zero;
        if (!std::isfinite(transformed)) return -1;
        const double rounded = std::round(transformed);
        output[linear] = static_cast<int8_t>(std::max(-128.0, std::min(127.0, rounded)));
    }
    return 0;
}

void EnsureX86ResizeKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::INT8, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::INT8>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
