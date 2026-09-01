#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "quant/quantization.h"
#include "src/kernel/common/int8_kernel_utils.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/fc.h"
#include "src/kernel/gemm.h"
#include "src/kernel/matmul.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

using cuda_detail::DeviceBuffer;
using int8_detail::QuantizationView;

struct HostInt8Quantization {
    std::vector<float> scales;
    std::vector<int32_t> zero_points;
};

bool BuildHostInt8Quantization(const Tensor* tensor, bool require_int8, bool allow_per_channel,
                               int64_t expected_axis, int64_t expected_channels,
                               HostInt8Quantization* result) {
    if (tensor == nullptr || result == nullptr || !tensor->IsInitialized() ||
        (require_int8 && tensor->data_type() != DataType::INT8) ||
        (!require_int8 && tensor->data_type() != DataType::INT8 && tensor->data_type() != DataType::UNKNOWN)) {
        return false;
    }
    const QuantizationParams& params = tensor->quantization();
    if (!params.enabled || !ValidateQuantizationParams(params, tensor->dims().data())) {
        return false;
    }
    if (params.granularity == QuantizationGranularity::kPerTensor) {
        result->scales = {params.scale_at(0)};
        result->zero_points = {params.zero_point_at(0)};
        return true;
    }
    if (!allow_per_channel || params.granularity != QuantizationGranularity::kPerChannel ||
        params.axis != expected_axis || expected_channels <= 0 ||
        params.scales.size() != static_cast<size_t>(expected_channels)) {
        return false;
    }
    result->scales = params.scales;
    result->zero_points.resize(result->scales.size());
    for (size_t index = 0; index < result->zero_points.size(); ++index) {
        result->zero_points[index] = params.zero_point_at(index);
    }
    return true;
}

bool CopyQuantizationToDevice(const HostInt8Quantization& host, DeviceBuffer<float>* scales,
                             DeviceBuffer<int32_t>* zero_points) {
    return scales != nullptr && zero_points != nullptr &&
           cuda_detail::CopyHostVectorToDevice(host.scales, scales) == 0 &&
           cuda_detail::CopyHostVectorToDevice(host.zero_points, zero_points) == 0;
}

bool IsInt32Bias(const std::shared_ptr<Tensor>& bias) {
    return bias == nullptr || (bias->data_type() == DataType::INT32 && bias->IsInitialized());
}

__device__ inline int8_t QuantizeInt8Device(float value, float scale, int32_t zero_point) {
    const float transformed = value / scale + static_cast<float>(zero_point);
    if (!isfinite(transformed) || transformed <= -128.0f) {
        return static_cast<int8_t>(-128);
    }
    if (transformed >= 127.0f) {
        return static_cast<int8_t>(127);
    }
    const int rounded = static_cast<int>(roundf(transformed));
    return static_cast<int8_t>(max(-128, min(127, rounded)));
}

__device__ inline void SetInt8Error(int* error) {
    if (error != nullptr) {
        atomicExch(error, 1);
    }
}

__device__ inline bool Int32AccumulatorIsValid(long long accumulator) {
    return accumulator >= static_cast<long long>(-2147483648LL) &&
           accumulator <= static_cast<long long>(2147483647LL);
}

__global__ void Int8LinearKernel(const int8_t* input, const int8_t* weight, const int32_t* bias,
                                 int8_t* output, int64_t rows, int64_t k, int64_t channels,
                                 bool trans_b, int bias_mode, int32_t input_zero_point,
                                 const float* weight_scales, const int32_t* weight_zero_points,
                                 int weight_parameter_count, float input_scale, float output_scale,
                                 int32_t output_zero_point, float alpha, float beta, int* error) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * channels;
    if (linear >= total) {
        return;
    }
    const int64_t row = linear / channels;
    const int64_t channel = linear % channels;
    long long accumulator = 0;
    const int32_t weight_zero_point = weight_zero_points[weight_parameter_count == 1 ? 0 : channel];
    for (int64_t index = 0; index < k; ++index) {
        const int64_t weight_offset = trans_b ? channel * k + index : index * channels + channel;
        accumulator += static_cast<long long>(static_cast<int32_t>(input[row * k + index]) - input_zero_point) *
                       (static_cast<int32_t>(weight[weight_offset]) - weight_zero_point);
    }
    if (!Int32AccumulatorIsValid(accumulator)) {
        SetInt8Error(error);
        return;
    }
    const float weight_scale = weight_scales[weight_parameter_count == 1 ? 0 : channel];
    float real_value = alpha * static_cast<float>(accumulator) * input_scale * weight_scale;
    if (bias != nullptr) {
        const int64_t bias_offset = bias_mode == 1 ? channel : linear;
        real_value += beta * static_cast<float>(bias[bias_offset]) * input_scale * weight_scale;
    }
    output[linear] = QuantizeInt8Device(real_value, output_scale, output_zero_point);
}

struct Int8MatMulSpec {
    cuda_detail::CudaShape a_shape{};
    cuda_detail::CudaShape b_shape{};
    int64_t batch_strides[cuda_detail::kMaxCudaRank]{};
    int batch_rank{0};
    int64_t batch_count{0};
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
};

__global__ void Int8MatMulKernel(const int8_t* lhs, const int8_t* rhs, int8_t* output,
                                 Int8MatMulSpec spec, int32_t input_zero_point,
                                 const float* weight_scales, const int32_t* weight_zero_points,
                                 int weight_parameter_count, float input_scale, float output_scale,
                                 int32_t output_zero_point, int* error) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t matrix_size = spec.m * spec.n;
    const int64_t total = spec.batch_count * matrix_size;
    if (linear >= total) {
        return;
    }
    const int64_t batch = linear / matrix_size;
    const int64_t matrix_index = linear % matrix_size;
    const int64_t row = matrix_index / spec.n;
    const int64_t channel = matrix_index % spec.n;
    const int a_batch_rank = spec.a_shape.rank - 2;
    const int b_batch_rank = spec.b_shape.rank - 2;
    const int a_gap = spec.batch_rank - a_batch_rank;
    const int b_gap = spec.batch_rank - b_batch_rank;
    int64_t remaining = batch;
    int64_t lhs_batch_offset = 0;
    int64_t rhs_batch_offset = 0;
    for (int axis = 0; axis < spec.batch_rank; ++axis) {
        const int64_t coordinate = remaining / spec.batch_strides[axis];
        remaining %= spec.batch_strides[axis];
        const int lhs_axis = axis - a_gap;
        if (lhs_axis >= 0 && spec.a_shape.dims[lhs_axis] != 1) {
            lhs_batch_offset += coordinate * spec.a_shape.strides[lhs_axis];
        }
        const int rhs_axis = axis - b_gap;
        if (rhs_axis >= 0 && spec.b_shape.dims[rhs_axis] != 1) {
            rhs_batch_offset += coordinate * spec.b_shape.strides[rhs_axis];
        }
    }

    long long accumulator = 0;
    const int32_t weight_zero_point = weight_zero_points[weight_parameter_count == 1 ? 0 : channel];
    for (int64_t index = 0; index < spec.k; ++index) {
        const int64_t lhs_offset = lhs_batch_offset + row * spec.a_shape.strides[a_batch_rank] +
                                   index * spec.a_shape.strides[a_batch_rank + 1];
        const int64_t rhs_offset = rhs_batch_offset + index * spec.b_shape.strides[b_batch_rank] +
                                   channel * spec.b_shape.strides[b_batch_rank + 1];
        accumulator += static_cast<long long>(static_cast<int32_t>(lhs[lhs_offset]) - input_zero_point) *
                       (static_cast<int32_t>(rhs[rhs_offset]) - weight_zero_point);
    }
    if (!Int32AccumulatorIsValid(accumulator)) {
        SetInt8Error(error);
        return;
    }
    const float weight_scale = weight_scales[weight_parameter_count == 1 ? 0 : channel];
    const float real_value = static_cast<float>(accumulator) * input_scale * weight_scale;
    output[linear] = QuantizeInt8Device(real_value, output_scale, output_zero_point);
}

__device__ inline int64_t ImageOffsetDevice(bool channel_last, int64_t batch, int64_t channel,
                                            int64_t height, int64_t width, int64_t channels,
                                            int64_t image_height, int64_t image_width) {
    return channel_last ? ((batch * image_height + height) * image_width + width) * channels + channel
                        : ((batch * channels + channel) * image_height + height) * image_width + width;
}

__global__ void Int8Conv2DKernel(const int8_t* input, const int8_t* weight, const int32_t* bias,
                                 int8_t* output, int64_t batches, int64_t input_channels,
                                 int64_t input_height, int64_t input_width, int64_t output_channels,
                                 int64_t output_height, int64_t output_width, int64_t kernel_channels,
                                 int64_t kernel_height, int64_t kernel_width, int64_t stride_h,
                                 int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dilation_h,
                                 int64_t dilation_w, int64_t groups, bool input_channel_last,
                                 bool output_channel_last, int32_t input_zero_point,
                                 const float* weight_scales, const int32_t* weight_zero_points,
                                 int weight_parameter_count, float input_scale, float output_scale,
                                 int32_t output_zero_point, int* error) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batches * output_channels * output_height * output_width;
    if (linear >= total) {
        return;
    }
    int64_t remaining = linear;
    int64_t batch = 0;
    int64_t output_channel = 0;
    int64_t output_height_index = 0;
    int64_t output_width_index = 0;
    if (output_channel_last) {
        batch = remaining / (output_height * output_width * output_channels);
        remaining %= output_height * output_width * output_channels;
        output_height_index = remaining / (output_width * output_channels);
        remaining %= output_width * output_channels;
        output_width_index = remaining / output_channels;
        output_channel = remaining % output_channels;
    } else {
        batch = remaining / (output_channels * output_height * output_width);
        remaining %= output_channels * output_height * output_width;
        output_channel = remaining / (output_height * output_width);
        remaining %= output_height * output_width;
        output_height_index = remaining / output_width;
        output_width_index = remaining % output_width;
    }

    const int64_t input_channels_per_group = input_channels / groups;
    const int64_t output_channels_per_group = output_channels / groups;
    const int64_t group = output_channel / output_channels_per_group;
    const int32_t weight_zero_point = weight_zero_points[weight_parameter_count == 1 ? 0 : output_channel];
    long long accumulator = bias == nullptr ? 0 : bias[output_channel];
    for (int64_t input_channel = 0; input_channel < input_channels_per_group; ++input_channel) {
        const int64_t global_input_channel = group * input_channels_per_group + input_channel;
        for (int64_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            const int64_t input_y = output_height_index * stride_h + kernel_y * dilation_h - pad_h;
            if (input_y < 0 || input_y >= input_height) continue;
            for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                const int64_t input_x = output_width_index * stride_w + kernel_x * dilation_w - pad_w;
                if (input_x < 0 || input_x >= input_width) continue;
                const int64_t input_offset = ImageOffsetDevice(
                    input_channel_last, batch, global_input_channel, input_y, input_x,
                    input_channels, input_height, input_width);
                const int64_t weight_offset =
                    ((output_channel * kernel_channels + input_channel) * kernel_height + kernel_y) *
                        kernel_width + kernel_x;
                accumulator += static_cast<long long>(static_cast<int32_t>(input[input_offset]) - input_zero_point) *
                               (static_cast<int32_t>(weight[weight_offset]) - weight_zero_point);
            }
        }
    }
    if (!Int32AccumulatorIsValid(accumulator)) {
        SetInt8Error(error);
        return;
    }
    const float weight_scale = weight_scales[weight_parameter_count == 1 ? 0 : output_channel];
    const float real_value = static_cast<float>(accumulator) * input_scale * weight_scale;
    output[linear] = QuantizeInt8Device(real_value, output_scale, output_zero_point);
}

bool CheckCudaKernel(int* error) {
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return false;
    }
    int host_error = 0;
    if (cuda_detail::CudaCheck(cudaMemcpyAsync(&host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost,
                                               cuda_detail::InferenceStream())) != 0 ||
        cuda_detail::CudaCheck(cudaStreamSynchronize(cuda_detail::InferenceStream())) != 0) {
        return false;
    }
    return host_error == 0;
}

bool PrepareCommonInt8Inputs(const Tensor* input, const Tensor* weight, Tensor* output,
                             const std::vector<int64_t>& expected_output, int64_t weight_axis,
                             int64_t weight_channels, HostInt8Quantization* input_quantization,
                             HostInt8Quantization* weight_quantization, HostInt8Quantization* output_quantization) {
    return input != nullptr && weight != nullptr && output != nullptr &&
           cuda_detail::IsTensorReady<DataType::INT8>(input) &&
           cuda_detail::IsTensorReady<DataType::INT8>(weight) &&
           cuda_detail::IsOutputReady<DataType::INT8>(output, &expected_output) &&
           BuildHostInt8Quantization(input, true, false, -1, 1, input_quantization) &&
           BuildHostInt8Quantization(weight, true, true, weight_axis, weight_channels, weight_quantization) &&
           BuildHostInt8Quantization(output, false, false, -1, 1, output_quantization);
}

int32_t RunCudaInt8Linear(const Tensor* input, const Tensor* weight, const std::shared_ptr<Tensor>& bias,
                          Tensor* output, const std::vector<int64_t>& expected_output, bool trans_b,
                          float alpha, float beta, const char* timer_name) {
    AutoTimer timer(timer_name);
    if (input == nullptr || weight == nullptr || output == nullptr || !std::isfinite(alpha) ||
        !std::isfinite(beta) || input->dims().size() < 2 || weight->dims().size() != 2) {
        return -1;
    }
    const int64_t k = input->dims()[input->dims().size() - 1];
    const int64_t rows = k > 0 ? input->numel() / k : 0;
    const int64_t weight_k = trans_b ? weight->dims()[1] : weight->dims()[0];
    const int64_t channels = trans_b ? weight->dims()[0] : weight->dims()[1];
    if (k <= 0 || rows <= 0 || channels <= 0 || weight_k != k ||
        output->numel() != rows * channels || !IsInt32Bias(bias) ||
        !int8_detail::ValidateLinearBias(bias, rows, channels)) {
        return -1;
    }
    HostInt8Quantization input_quantization;
    HostInt8Quantization weight_quantization;
    HostInt8Quantization output_quantization;
    if (!PrepareCommonInt8Inputs(input, weight, output, expected_output, trans_b ? 0 : 1, channels,
                                 &input_quantization, &weight_quantization, &output_quantization)) {
        return -1;
    }

    DeviceBuffer<int8_t> input_device;
    DeviceBuffer<int8_t> weight_device;
    DeviceBuffer<int32_t> bias_device;
    DeviceBuffer<int8_t> output_device;
    DeviceBuffer<float> weight_scales_device;
    DeviceBuffer<int32_t> weight_zero_points_device;
    DeviceBuffer<int> error_device;
    if (cuda_detail::CopyTensorToDevice(input, &input_device) != 0 ||
        cuda_detail::CopyTensorToDevice(weight, &weight_device) != 0 ||
        cuda_detail::AllocateTensorOnDevice(output, &output_device) != 0 ||
        !CopyQuantizationToDevice(weight_quantization, &weight_scales_device, &weight_zero_points_device) ||
        error_device.allocate(1) != 0) {
        return -1;
    }
    int bias_mode = 0;
    if (bias != nullptr) {
        if (cuda_detail::CopyTensorToDevice(bias.get(), &bias_device) != 0) return -1;
        bias_mode = bias->numel() == channels ? 1 : 2;
    }
    if (cuda_detail::CudaCheck(cudaMemsetAsync(error_device.get(), 0, sizeof(int), cuda_detail::InferenceStream())) != 0) {
        return -1;
    }
    const int64_t total = rows * channels;
    Int8LinearKernel<<<static_cast<unsigned int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                       cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input_device.get(), weight_device.get(), bias == nullptr ? nullptr : bias_device.get(), output_device.get(),
        rows, k, channels, trans_b, bias_mode, input_quantization.zero_points[0], weight_scales_device.get(),
        weight_zero_points_device.get(), static_cast<int>(weight_quantization.scales.size()), input_quantization.scales[0],
        output_quantization.scales[0], output_quantization.zero_points[0], alpha, beta, error_device.get());
    return CheckCudaKernel(error_device.get()) ? cuda_detail::CopyDeviceToTensor(&output_device, output) : -1;
}

bool BuildMatMulSpec(const Tensor* lhs, const Tensor* rhs, const Tensor* output, Int8MatMulSpec* spec) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr || spec == nullptr || lhs->dims().size() < 2 ||
        rhs->dims().size() < 2 || output->dims().size() != std::max(lhs->dims().size(), rhs->dims().size())) {
        return false;
    }
    const auto& lhs_dims = lhs->dims().data();
    const auto& rhs_dims = rhs->dims().data();
    const auto& output_dims = output->dims().data();
    const size_t output_rank = output_dims.size();
    const int64_t m = lhs_dims[lhs_dims.size() - 2];
    const int64_t k = lhs_dims[lhs_dims.size() - 1];
    const int64_t rhs_k = rhs_dims[rhs_dims.size() - 2];
    const int64_t n = rhs_dims[rhs_dims.size() - 1];
    if (m <= 0 || k <= 0 || n <= 0 || rhs_k != k || output_dims[output_rank - 2] != m ||
        output_dims[output_rank - 1] != n || !cuda_detail::MakeCudaShape(lhs_dims, &spec->a_shape) ||
        !cuda_detail::MakeCudaShape(rhs_dims, &spec->b_shape)) {
        return false;
    }
    spec->batch_rank = static_cast<int>(output_rank - 2);
    spec->m = m;
    spec->k = k;
    spec->n = n;
    spec->batch_count = 1;
    for (int axis = 0; axis < spec->batch_rank; ++axis) {
        const size_t output_axis = static_cast<size_t>(axis);
        const size_t lhs_gap = output_rank - lhs_dims.size();
        const size_t rhs_gap = output_rank - rhs_dims.size();
        const int64_t lhs_dim = output_axis < lhs_gap ? 1 : lhs_dims[output_axis - lhs_gap];
        const int64_t rhs_dim = output_axis < rhs_gap ? 1 : rhs_dims[output_axis - rhs_gap];
        const int64_t expected_dim = output_dims[output_axis];
        if ((lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim) || expected_dim != std::max(lhs_dim, rhs_dim) ||
            spec->batch_count > std::numeric_limits<int64_t>::max() / expected_dim) {
            return false;
        }
        spec->batch_count *= expected_dim;
    }
    const std::vector<int64_t> batch_dims(output_dims.begin(), output_dims.begin() + spec->batch_rank);
    const auto batch_strides = cuda_detail::ComputeStrides(batch_dims);
    for (int axis = 0; axis < spec->batch_rank; ++axis) {
        spec->batch_strides[axis] = batch_strides[static_cast<size_t>(axis)];
    }
    return true;
}

int32_t RunCudaInt8MatMul(feather::operators::MatMulParam* param) {
    AutoTimer timer("CUDA::MatMul::INT8");
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) return -1;
    Int8MatMulSpec spec;
    if (!BuildMatMulSpec(param->a.get(), param->b.get(), param->out.get(), &spec)) return -1;
    HostInt8Quantization input_quantization;
    HostInt8Quantization weight_quantization;
    HostInt8Quantization output_quantization;
    if (!PrepareCommonInt8Inputs(param->a.get(), param->b.get(), param->out.get(), param->out->dims().data(),
                                 static_cast<int64_t>(spec.b_shape.rank - 1), spec.n,
                                 &input_quantization, &weight_quantization, &output_quantization)) return -1;
    DeviceBuffer<int8_t> lhs_device;
    DeviceBuffer<int8_t> rhs_device;
    DeviceBuffer<int8_t> output_device;
    DeviceBuffer<float> weight_scales_device;
    DeviceBuffer<int32_t> weight_zero_points_device;
    DeviceBuffer<int> error_device;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &lhs_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &rhs_device) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output_device) != 0 ||
        !CopyQuantizationToDevice(weight_quantization, &weight_scales_device, &weight_zero_points_device) ||
        error_device.allocate(1) != 0) return -1;
    if (cuda_detail::CudaCheck(cudaMemsetAsync(error_device.get(), 0, sizeof(int), cuda_detail::InferenceStream())) != 0) {
        return -1;
    }
    const int64_t total = spec.batch_count * spec.m * spec.n;
    Int8MatMulKernel<<<static_cast<unsigned int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                       cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        lhs_device.get(), rhs_device.get(), output_device.get(), spec, input_quantization.zero_points[0],
        weight_scales_device.get(), weight_zero_points_device.get(), static_cast<int>(weight_quantization.scales.size()),
        input_quantization.scales[0], output_quantization.scales[0], output_quantization.zero_points[0], error_device.get());
    return CheckCudaKernel(error_device.get()) ? cuda_detail::CopyDeviceToTensor(&output_device, param->out.get()) : -1;
}

int32_t RunCudaInt8Conv2D(feather::operators::Conv2dParam* param) {
    AutoTimer timer("CUDA::Conv2D::INT8");
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr ||
        param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4 ||
        param->stride_h <= 0 || param->stride_w <= 0 || param->dilation_h <= 0 || param->dilation_w <= 0 ||
        param->group <= 0 || !int8_detail::ValidateConvBias(param->bias, param->w->dims()[0])) return -1;
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) return -1;
    const int64_t output_channels = param->w->dims()[0];
    const int64_t kernel_channels = param->w->dims()[1];
    const int64_t kernel_height = param->w->dims()[2];
    const int64_t kernel_width = param->w->dims()[3];
    const int64_t expected_height = (input_shape.h + 2 * param->pad_h -
                                     param->dilation_h * (kernel_height - 1) - 1) /
                                        param->stride_h + 1;
    const int64_t expected_width = (input_shape.w + 2 * param->pad_w -
                                    param->dilation_w * (kernel_width - 1) - 1) /
                                       param->stride_w + 1;
    if (input_shape.n <= 0 || input_shape.c <= 0 || input_shape.h <= 0 || input_shape.w <= 0 ||
        output_channels <= 0 || kernel_channels <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
        input_shape.c % param->group != 0 || output_channels % param->group != 0 ||
        kernel_channels != input_shape.c / param->group || output_shape.n != input_shape.n ||
        output_shape.c != output_channels || output_shape.h != expected_height || output_shape.w != expected_width) return -1;
    HostInt8Quantization input_quantization;
    HostInt8Quantization weight_quantization;
    HostInt8Quantization output_quantization;
    if (!PrepareCommonInt8Inputs(param->input.get(), param->w.get(), param->out.get(), param->out->dims().data(),
                                 0, output_channels, &input_quantization, &weight_quantization,
                                 &output_quantization)) return -1;
    DeviceBuffer<int8_t> input_device;
    DeviceBuffer<int8_t> weight_device;
    DeviceBuffer<int32_t> bias_device;
    DeviceBuffer<int8_t> output_device;
    DeviceBuffer<float> weight_scales_device;
    DeviceBuffer<int32_t> weight_zero_points_device;
    DeviceBuffer<int> error_device;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input_device) != 0 ||
        cuda_detail::CopyTensorToDevice(param->w.get(), &weight_device) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output_device) != 0 ||
        !CopyQuantizationToDevice(weight_quantization, &weight_scales_device, &weight_zero_points_device) ||
        error_device.allocate(1) != 0) return -1;
    if (param->bias != nullptr && cuda_detail::CopyTensorToDevice(param->bias.get(), &bias_device) != 0) return -1;
    if (cuda_detail::CudaCheck(cudaMemsetAsync(error_device.get(), 0, sizeof(int), cuda_detail::InferenceStream())) != 0) {
        return -1;
    }
    const int64_t total = input_shape.n * output_channels * output_shape.h * output_shape.w;
    Int8Conv2DKernel<<<static_cast<unsigned int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                       cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input_device.get(), weight_device.get(), param->bias == nullptr ? nullptr : bias_device.get(), output_device.get(),
        input_shape.n, input_shape.c, input_shape.h, input_shape.w, output_channels, output_shape.h, output_shape.w,
        kernel_channels, kernel_height, kernel_width, param->stride_h, param->stride_w, param->pad_h, param->pad_w,
        param->dilation_h, param->dilation_w, param->group, IsChannelLastLayout(param->input->layout()),
        IsChannelLastLayout(param->out->layout()), input_quantization.zero_points[0], weight_scales_device.get(),
        weight_zero_points_device.get(), static_cast<int>(weight_quantization.scales.size()), input_quantization.scales[0],
        output_quantization.scales[0], output_quantization.zero_points[0], error_device.get());
    return CheckCudaKernel(error_device.get()) ? cuda_detail::CopyDeviceToTensor(&output_device, param->out.get()) : -1;
}

}  // namespace

void EnsureCudaInt8KernelsRegistered() {
    static const bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "FC", []() {
            return std::make_unique<FcKernel<DeviceType::CUDA, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "Gemm", []() {
            return std::make_unique<GemmKernel<DeviceType::CUDA, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "MatMul", []() {
            return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, "Conv2D", []() {
            return std::make_unique<Conv2DKernel<DeviceType::CUDA, DataType::INT8>>();
        });
        return true;
    }();
    (void)registered;
}

template <>
int32_t FcKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) return -1;
    return RunCudaInt8Linear(param->input.get(), param->w.get(), param->bias, param->out.get(),
                             param->out->dims().data(), false, 1.0f, 1.0f, "CUDA::FC::INT8");
}

template <>
int32_t GemmKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) return -1;
    return RunCudaInt8Linear(param->a.get(), param->b.get(), param->bias, param->out.get(), param->out->dims().data(),
                             param->trans_b, param->alpha, param->beta, "CUDA::Gemm::INT8");
}

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    return RunCudaInt8MatMul(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t Conv2DKernel<DeviceType::CUDA, DataType::INT8>::compute() {
    return RunCudaInt8Conv2D(static_cast<feather::operators::Conv2dParam*>(param_));
}

}  // namespace kernel
}  // namespace feather
