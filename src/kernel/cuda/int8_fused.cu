#include "src/kernel/common/int8_fused.h"

#include <cmath>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/yolo_decode.h"

namespace feather {
namespace kernel {
namespace {

__device__ inline int64_t ClampCuda(int64_t value, int64_t extent) {
    if (extent <= 0) return 0;
    return value < 0 ? 0 : (value >= extent ? extent - 1 : value);
}

__device__ inline int64_t PhysicalIndexCuda(bool channel_last, int64_t n, int64_t c, int64_t y, int64_t x,
                                            int64_t channels, int64_t height, int64_t width) {
    if (channel_last) return ((n * height + y) * width + x) * channels + c;
    return ((n * channels + c) * height + y) * width + x;
}

__device__ inline float DequantizeInt8(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<int>(value) - zero_point) * (scale > 0.0f ? scale : 1.0f);
}

__device__ inline int8_t QuantizeInt8(float value, float scale, int32_t zero_point) {
    const float safe_scale = scale > 0.0f ? scale : 1.0f;
    int q = static_cast<int>(rintf(value / safe_scale)) + zero_point;
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

__global__ void ResizeConcatInt8CudaKernel(
    const int8_t* resize_input, const int8_t* concat_input, int8_t* output, int64_t total,
    int64_t resize_channels, int64_t concat_channels, int64_t output_channels,
    int64_t resize_height, int64_t resize_width, int64_t concat_height, int64_t concat_width,
    int64_t output_height, int64_t output_width, float scale_h, float scale_w,
    float resize_scale, int32_t resize_zero, float concat_scale, int32_t concat_zero,
    float output_scale, int32_t output_zero, int resize_first, bool channel_last) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int64_t n = 0, c = 0, y = 0, x = 0;
    if (channel_last) {
        c = index % output_channels;
        x = (index / output_channels) % output_width;
        y = (index / (output_channels * output_width)) % output_height;
        n = index / (output_channels * output_width * output_height);
    } else {
        x = index % output_width;
        y = (index / output_width) % output_height;
        c = (index / (output_width * output_height)) % output_channels;
        n = index / (output_channels * output_height * output_width);
    }
    const bool from_resize = resize_first ? c < resize_channels : c >= concat_channels;
    const int64_t local_c = from_resize
        ? (resize_first ? c : c - concat_channels)
        : (resize_first ? c - resize_channels : c);
    float value = 0.0f;
    if (from_resize) {
        const int64_t source_y = ClampCuda(static_cast<int64_t>(floorf(y / (scale_h > 0.0f ? scale_h : 1.0f))), resize_height);
        const int64_t source_x = ClampCuda(static_cast<int64_t>(floorf(x / (scale_w > 0.0f ? scale_w : 1.0f))), resize_width);
        const int64_t source_index = PhysicalIndexCuda(channel_last, n, local_c, source_y, source_x,
                                                       resize_channels, resize_height, resize_width);
        value = DequantizeInt8(resize_input[source_index], resize_scale, resize_zero);
    } else {
        const int64_t source_y = ClampCuda(y, concat_height);
        const int64_t source_x = ClampCuda(x, concat_width);
        const int64_t source_index = PhysicalIndexCuda(channel_last, n, local_c, source_y, source_x,
                                                       concat_channels, concat_height, concat_width);
        value = DequantizeInt8(concat_input[source_index], concat_scale, concat_zero);
    }
    output[index] = QuantizeInt8(value, output_scale, output_zero);
}

__device__ inline float AuxValueCuda(const float* data, int64_t count, int64_t batch, int64_t anchor,
                                     int64_t y, int64_t x, int component, int64_t anchors,
                                     int64_t height, int64_t width) {
    if (data == nullptr || count <= 0) return 0.0f;
    if (count == 1) return data[0];
    const int64_t spatial = height * width;
    if (count == anchors * spatial * 2) return data[((anchor * spatial + y * width + x) * 2 + component) % count];
    if (count == spatial * 2) return data[((y * width + x) * 2 + component) % count];
    if (count == anchors * 2) return data[(anchor * 2 + component) % count];
    if (count == 2) return data[component % 2];
    if (count == anchors * spatial) return data[(anchor * spatial + y * width + x) % count];
    if (count == anchors) return data[anchor % count];
    if (count == spatial) return data[(y * width + x) % count];
    const int64_t fallback = ((batch * anchors + anchor) * spatial + y * width + x) * 2 + component;
    return data[fallback % count];
}

__device__ inline float StableSigmoidCuda(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

__global__ void YoloDecodeInt8CudaKernel(
    const int8_t* input, int8_t* output, const float* xy_scale, int64_t xy_count,
    const float* grid, int64_t grid_count, const float* stride, int64_t stride_count,
    const float* wh_scale, int64_t wh_count, const float* anchor_grid, int64_t anchor_count,
    int64_t batch, int64_t anchors, int64_t attrs, int64_t height, int64_t width,
    float input_scale, int32_t input_zero, float output_scale, int32_t output_zero) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * anchors * height * width * attrs;
    if (index >= total) return;
    int64_t rest = index;
    const int64_t attr = rest % attrs; rest /= attrs;
    const int64_t x = rest % width; rest /= width;
    const int64_t y = rest % height; rest /= height;
    const int64_t anchor = rest % anchors; rest /= anchors;
    const int64_t batch_index = rest;
    const int64_t spatial = height * width;
    const int64_t input_index = ((batch_index * anchors * attrs + anchor * attrs + attr) * height + y) * width + x;
    const float value = StableSigmoidCuda(DequantizeInt8(input[input_index], input_scale, input_zero));
    float decoded = value;
    if (attr < 2) {
        const float xy = AuxValueCuda(xy_scale, xy_count, batch_index, anchor, y, x,
                                      static_cast<int>(attr), anchors, height, width);
        const float grid_value = AuxValueCuda(grid, grid_count, batch_index, anchor, y, x,
                                              static_cast<int>(attr), anchors, height, width);
        const float stride_value = AuxValueCuda(stride, stride_count, batch_index, anchor, y, x,
                                                0, anchors, height, width);
        decoded = (value * xy + grid_value) * stride_value;
    } else if (attr < 4) {
        const int component = static_cast<int>(attr - 2);
        const float wh = value * AuxValueCuda(wh_scale, wh_count, batch_index, anchor, y, x,
                                              component, anchors, height, width);
        const float anchor_value = AuxValueCuda(anchor_grid, anchor_count, batch_index, anchor, y, x,
                                                component, anchors, height, width);
        decoded = wh * wh * anchor_value;
    }
    const int64_t output_index = ((batch_index * anchors * spatial + anchor * spatial + y * width + x) * attrs) + attr;
    output[output_index] = QuantizeInt8(decoded, output_scale, output_zero);
}

inline int64_t TensorElements(const std::shared_ptr<Tensor>& tensor) {
    return tensor == nullptr ? 0 : tensor->numel();
}

}  // namespace

namespace {

bool g_cuda_int8_fused_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "Shape", []() {
        return std::make_unique<ShapeKernel<DeviceType::CUDA, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "ResizeConcat", []() {
        return std::make_unique<ResizeConcatKernel<DeviceType::CUDA, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "YoloDecode", []() {
        return std::make_unique<YoloDecodeKernel<DeviceType::CUDA, DataType::INT8>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ShapeKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    return common::ComputeShapeInt8(static_cast<feather::operators::ShapeParam*>(param_));
}

template <>
int32_t ResizeConcatKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    auto* param = static_cast<feather::operators::ResizeConcatParam*>(param_);
    if (param == nullptr || param->resize_input == nullptr || param->concat_input == nullptr || param->out == nullptr) return -1;
    const auto resize_dims = param->resize_input->dims().data();
    const auto concat_dims = param->concat_input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (resize_dims.size() != 4 || concat_dims.size() != 4 || output_dims.size() != 4) return -1;
    const bool channel_last = IsChannelLastLayout(param->out->layout());
    const int axis = param->axis < 0 ? param->axis + 4 : param->axis;
    const int channel_axis = channel_last ? 3 : 1;
    if (axis != channel_axis || (param->resize_input_index != 0 && param->resize_input_index != 1)) return -1;
    const int64_t resize_channels = channel_last ? resize_dims[3] : resize_dims[1];
    const int64_t concat_channels = channel_last ? concat_dims[3] : concat_dims[1];
    const int64_t output_channels = channel_last ? output_dims[3] : output_dims[1];
    const int64_t resize_height = channel_last ? resize_dims[1] : resize_dims[2];
    const int64_t resize_width = channel_last ? resize_dims[2] : resize_dims[3];
    const int64_t concat_height = channel_last ? concat_dims[1] : concat_dims[2];
    const int64_t concat_width = channel_last ? concat_dims[2] : concat_dims[3];
    const int64_t output_height = channel_last ? output_dims[1] : output_dims[2];
    const int64_t output_width = channel_last ? output_dims[2] : output_dims[3];
    if (output_channels != resize_channels + concat_channels) return -1;
    cuda_detail::DeviceBuffer<int8_t> resize_device;
    cuda_detail::DeviceBuffer<int8_t> concat_device;
    cuda_detail::DeviceBuffer<int8_t> output_device;
    if (cuda_detail::CopyTensorToDevice(param->resize_input.get(), &resize_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->concat_input.get(), &concat_device) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output_device) != 0) return -1;
    const auto resize_q = common::GetInt8Quantization(param->resize_input);
    const auto concat_q = common::GetInt8Quantization(param->concat_input);
    const auto output_q = common::GetInt8Quantization(param->out);
    const float scale_h = param->scales.size() > static_cast<size_t>(channel_last ? 1 : 2) ? param->scales[channel_last ? 1 : 2] : 1.0f;
    const float scale_w = param->scales.size() > static_cast<size_t>(channel_last ? 2 : 3) ? param->scales[channel_last ? 2 : 3] : 1.0f;
    const int block = 256;
    const int64_t total = param->out->numel();
    ResizeConcatInt8CudaKernel<<<static_cast<int>((total + block - 1) / block), block>>>(
        resize_device.get(), concat_device.get(), output_device.get(), total,
        resize_channels, concat_channels, output_channels, resize_height, resize_width,
        concat_height, concat_width, output_height, output_width, scale_h, scale_w,
        resize_q.scale, resize_q.zero_point, concat_q.scale, concat_q.zero_point,
        output_q.scale, output_q.zero_point, param->resize_input_index == 0 ? 1 : 0, channel_last);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return cuda_detail::CopyDeviceToTensor(&output_device, param->out.get());
}

template <>
int32_t YoloDecodeKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    auto* param = static_cast<feather::operators::YoloDecodeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->xy_scale == nullptr ||
        param->grid == nullptr || param->stride == nullptr || param->wh_scale == nullptr || param->anchor_grid == nullptr) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (input_dims.size() != 4 || output_dims.size() != 3 || input_dims[1] <= 0 ||
        output_dims[2] <= 0 || input_dims[1] % output_dims[2] != 0) return -1;
    cuda_detail::DeviceBuffer<int8_t> input_device;
    cuda_detail::DeviceBuffer<int8_t> output_device;
    cuda_detail::DeviceBuffer<float> xy_device;
    cuda_detail::DeviceBuffer<float> grid_device;
    cuda_detail::DeviceBuffer<float> stride_device;
    cuda_detail::DeviceBuffer<float> wh_device;
    cuda_detail::DeviceBuffer<float> anchor_device;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input_device) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->xy_scale.get(), &xy_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->grid.get(), &grid_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->stride.get(), &stride_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->wh_scale.get(), &wh_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->anchor_grid.get(), &anchor_device) != 0) return -1;
    const auto input_q = common::GetInt8Quantization(param->input);
    const auto output_q = common::GetInt8Quantization(param->out);
    const int64_t batch = input_dims[0];
    const int64_t attrs = output_dims[2];
    const int64_t anchors = input_dims[1] / attrs;
    const int64_t total = param->out->numel();
    const int block = 256;
    YoloDecodeInt8CudaKernel<<<static_cast<int>((total + block - 1) / block), block>>>(
        input_device.get(), output_device.get(), xy_device.get(), TensorElements(param->xy_scale),
        grid_device.get(), TensorElements(param->grid), stride_device.get(), TensorElements(param->stride),
        wh_device.get(), TensorElements(param->wh_scale), anchor_device.get(), TensorElements(param->anchor_grid),
        batch, anchors, attrs, input_dims[2], input_dims[3], input_q.scale, input_q.zero_point,
        output_q.scale, output_q.zero_point);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return cuda_detail::CopyDeviceToTensor(&output_device, param->out.get());
}

void EnsureCudaInt8FusedKernelsRegistered() {
    (void)g_cuda_int8_fused_kernels_registered;
}

}  // namespace kernel
}  // namespace feather

