#include "src/kernel/common/int8_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "src/kernel/common/int8_standard_utils.h"
#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/yolo_decode.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace common {

namespace int8_fused_detail {

inline int64_t PhysicalIndex(bool channel_last, int64_t n, int64_t c, int64_t y, int64_t x,
                             int64_t channels, int64_t height, int64_t width) {
    if (channel_last) return ((n * height + y) * width + x) * channels + c;
    return ((n * channels + c) * height + y) * width + x;
}

inline int64_t ClampCoordinate(int64_t value, int64_t extent) {
    if (extent <= 0) return 0;
    return std::max<int64_t>(0, std::min<int64_t>(value, extent - 1));
}

inline float ScaleAt(const std::vector<float>& scales, size_t index) {
    return index < scales.size() && scales[index] > 0.0f ? scales[index] : 1.0f;
}

inline float StableSigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

inline float AuxValue(const std::shared_ptr<Tensor>& tensor, int64_t batch, int64_t anchor,
                      int64_t y, int64_t x, int component, int64_t anchors,
                      int64_t height, int64_t width) {
    if (!tensor || !tensor->IsInitialized() || tensor->data_type() != DataType::FP32) return 0.0f;
    const auto dims = tensor->dims().data();
    const int64_t count = tensor->numel();
    const float* data = tensor->data<float>();
    if (data == nullptr || count <= 0) return 0.0f;
    if (count == 1) return data[0];
    const int64_t spatial = height * width;
    if (dims.size() == 4 && dims[1] == 2 && dims[2] == height && dims[3] == width) {
        return data[(static_cast<int64_t>(component) * spatial + y * width + x) % count];
    }
    if (!dims.empty() && dims.back() == 2 && count >= spatial * 2) {
        return data[((y * width + x) * 2 + component) % count];
    }
    if (count == anchors * spatial * 2) {
        return data[(((anchor * spatial + y * width + x) * 2) + component) % count];
    }
    if (count == anchors * 2) return data[(anchor * 2 + component) % count];
    if (count == spatial * 2) return data[((y * width + x) * 2 + component) % count];
    if (count == anchors * spatial) return data[(anchor * spatial + y * width + x) % count];
    if (count == anchors) return data[anchor % count];
    if (count == spatial) return data[(y * width + x) % count];
    if (count == 2) return data[component % 2];
    const int64_t fallback = ((batch * anchors + anchor) * spatial + y * width + x) * 2 + component;
    return data[fallback % count];
}

}  // namespace int8_fused_detail

Int8Quantization GetInt8Quantization(const std::shared_ptr<Tensor>& tensor) {
    Int8Quantization result;
    if (!tensor) return result;
    int8_standard_detail::View view;
    if (!int8_standard_detail::BuildView(tensor.get(), &view)) return result;
    result.scale = view.scale;
    result.zero_point = view.zero_point;
    return result;
}

int32_t ComputeShapeInt8(feather::operators::ShapeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        !int8_standard_detail::Ready(param->input.get()) || !param->out->IsInitialized()) return -1;
    const auto dims = param->input->dims().data();
    const int64_t rank = static_cast<int64_t>(dims.size());
    int64_t start = param->start < 0 ? param->start + rank : param->start;
    int64_t end = param->end < 0 ? param->end + rank : param->end;
    start = std::max<int64_t>(0, std::min<int64_t>(start, rank));
    end = std::max<int64_t>(0, std::min<int64_t>(end, rank));
    if (end < start || param->out->numel() != end - start ||
        param->out->memory_size() < static_cast<size_t>(end - start) * sizeof(int64_t)) return -1;
    param->out->set_data_type(DataType::INT64);
    auto* output = param->out->mutable_data<int64_t>();
    if (output == nullptr) return -1;
    for (int64_t index = start; index < end; ++index) output[index - start] = dims[static_cast<size_t>(index)];
    return 0;
}

int32_t ComputeResizeConcatInt8(feather::operators::ResizeConcatParam* param) {
    if (param == nullptr || param->resize_input == nullptr || param->concat_input == nullptr || param->out == nullptr ||
        !int8_standard_detail::Ready(param->resize_input.get()) ||
        !int8_standard_detail::Ready(param->concat_input.get()) ||
        !int8_standard_detail::Ready(param->out.get())) return -1;
    const auto resize_dims = param->resize_input->dims().data();
    const auto concat_dims = param->concat_input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (resize_dims.size() != 4 || concat_dims.size() != 4 || output_dims.size() != 4) return -1;
    const bool channel_last = IsChannelLastLayout(param->out->layout());
    const int axis = param->axis < 0 ? param->axis + 4 : param->axis;
    const int channel_axis = channel_last ? 3 : 1;
    if (axis != channel_axis || (param->resize_input_index != 0 && param->resize_input_index != 1)) return -1;
    const int64_t batch = output_dims[0];
    const int64_t resize_channels = channel_last ? resize_dims[3] : resize_dims[1];
    const int64_t concat_channels = channel_last ? concat_dims[3] : concat_dims[1];
    const int64_t output_channels = channel_last ? output_dims[3] : output_dims[1];
    const int64_t resize_height = channel_last ? resize_dims[1] : resize_dims[2];
    const int64_t resize_width = channel_last ? resize_dims[2] : resize_dims[3];
    const int64_t concat_height = channel_last ? concat_dims[1] : concat_dims[2];
    const int64_t concat_width = channel_last ? concat_dims[2] : concat_dims[3];
    const int64_t output_height = channel_last ? output_dims[1] : output_dims[2];
    const int64_t output_width = channel_last ? output_dims[2] : output_dims[3];
    if (output_channels != resize_channels + concat_channels || batch != resize_dims[0] || batch != concat_dims[0]) return -1;
    int8_standard_detail::View resize_view;
    int8_standard_detail::View concat_view;
    int8_standard_detail::View output_view;
    if (!int8_standard_detail::BuildView(param->resize_input.get(), &resize_view) ||
        !int8_standard_detail::BuildView(param->concat_input.get(), &concat_view) ||
        !int8_standard_detail::BuildOutputView(param->out.get(), &output_view)) return -1;
    const float scale_h = int8_fused_detail::ScaleAt(param->scales, channel_last ? 1 : 2);
    const float scale_w = int8_fused_detail::ScaleAt(param->scales, channel_last ? 2 : 3);
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < output_channels; ++c) {
            const bool from_resize = param->resize_input_index == 0 ? c < resize_channels : c >= concat_channels;
            const int64_t local_c = from_resize
                ? (param->resize_input_index == 0 ? c : c - concat_channels)
                : (param->resize_input_index == 0 ? c - resize_channels : c);
            for (int64_t y = 0; y < output_height; ++y) {
                for (int64_t x = 0; x < output_width; ++x) {
                    float value = 0.0f;
                    if (from_resize) {
                        const int64_t source_y = int8_fused_detail::ClampCoordinate(
                            static_cast<int64_t>(std::floor(static_cast<float>(y) / scale_h)), resize_height);
                        const int64_t source_x = int8_fused_detail::ClampCoordinate(
                            static_cast<int64_t>(std::floor(static_cast<float>(x) / scale_w)), resize_width);
                        const int64_t source_index = int8_fused_detail::PhysicalIndex(
                            channel_last, n, local_c, source_y, source_x, resize_channels, resize_height, resize_width);
                        value = int8_standard_detail::ReadReal(param->resize_input.get(), source_index, resize_view);
                    } else {
                        const int64_t source_y = int8_fused_detail::ClampCoordinate(y, concat_height);
                        const int64_t source_x = int8_fused_detail::ClampCoordinate(x, concat_width);
                        const int64_t source_index = int8_fused_detail::PhysicalIndex(
                            channel_last, n, local_c, source_y, source_x, concat_channels, concat_height, concat_width);
                        value = int8_standard_detail::ReadReal(param->concat_input.get(), source_index, concat_view);
                    }
                    const int64_t output_index = int8_fused_detail::PhysicalIndex(
                        channel_last, n, c, y, x, output_channels, output_height, output_width);
                    if (!int8_standard_detail::WriteReal(param->out.get(), output_index, value, output_view)) return -1;
                }
            }
        }
    }
    return 0;
}

int32_t ComputeYoloDecodeInt8(feather::operators::YoloDecodeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->xy_scale == nullptr || param->grid == nullptr || param->stride == nullptr ||
        param->wh_scale == nullptr || param->anchor_grid == nullptr ||
        !int8_standard_detail::Ready(param->input.get()) || !int8_standard_detail::Ready(param->out.get())) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (input_dims.size() != 4 || output_dims.size() != 3 || input_dims[0] != output_dims[0] ||
        output_dims[2] <= 0 || input_dims[1] <= 0 || input_dims[1] % output_dims[2] != 0) return -1;
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];
    const int64_t attrs = output_dims[2];
    const int64_t anchors = channels / attrs;
    const int64_t spatial = height * width;
    if (output_dims[1] != anchors * spatial) return -1;
    int8_standard_detail::View input_view;
    int8_standard_detail::View output_view;
    if (!int8_standard_detail::BuildView(param->input.get(), &input_view) ||
        !int8_standard_detail::BuildOutputView(param->out.get(), &output_view)) return -1;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t a = 0; a < anchors; ++a) {
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t output_base = (b * anchors * spatial + (a * spatial + y * width + x)) * attrs;
                    for (int64_t attr = 0; attr < attrs; ++attr) {
                        const int64_t input_index = ((b * channels + a * attrs + attr) * height + y) * width + x;
                        const float value = int8_fused_detail::StableSigmoid(
                            int8_standard_detail::ReadReal(param->input.get(), input_index, input_view));
                        float decoded = value;
                        if (attr < 2) {
                            const float xy = int8_fused_detail::AuxValue(
                                param->xy_scale, b, a, y, x, static_cast<int>(attr), anchors, height, width);
                            const float grid = int8_fused_detail::AuxValue(
                                param->grid, b, a, y, x, static_cast<int>(attr), anchors, height, width);
                            const float stride = int8_fused_detail::AuxValue(
                                param->stride, b, a, y, x, 0, anchors, height, width);
                            decoded = (value * xy + grid) * stride;
                        } else if (attr < 4) {
                            const int component = static_cast<int>(attr - 2);
                            const float wh = value * int8_fused_detail::AuxValue(
                                param->wh_scale, b, a, y, x, component, anchors, height, width);
                            const float anchor = int8_fused_detail::AuxValue(
                                param->anchor_grid, b, a, y, x, component, anchors, height, width);
                            decoded = wh * wh * anchor;
                        }
                        if (!int8_standard_detail::WriteReal(param->out.get(), output_base + attr, decoded, output_view)) return -1;
                    }
                }
            }
        }
    }
    return 0;
}

}  // namespace common

namespace {

bool g_common_int8_fused_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT8, "Shape", []() {
        return std::make_unique<ShapeKernel<DeviceType::COMMON, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT8, "ResizeConcat", []() {
        return std::make_unique<ResizeConcatKernel<DeviceType::COMMON, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT8, "YoloDecode", []() {
        return std::make_unique<YoloDecodeKernel<DeviceType::COMMON, DataType::INT8>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ShapeKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::Shape::INT8");
    return common::ComputeShapeInt8(static_cast<feather::operators::ShapeParam*>(param_));
}

template <>
int32_t ResizeConcatKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::ResizeConcat::INT8");
    return common::ComputeResizeConcatInt8(static_cast<feather::operators::ResizeConcatParam*>(param_));
}

template <>
int32_t YoloDecodeKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::YoloDecode::INT8");
    return common::ComputeYoloDecodeInt8(static_cast<feather::operators::YoloDecodeParam*>(param_));
}

void EnsureCommonInt8FusedKernelsRegistered() {
    (void)g_common_int8_fused_kernels_registered;
}

}  // namespace kernel
}  // namespace feather

