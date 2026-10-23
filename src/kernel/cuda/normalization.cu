// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/kernel/normalization.h"

#include <algorithm>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

struct AffineShape {
    int rank{0};
    int64_t input_strides[cuda_detail::kMaxCudaRank]{};
    int64_t weight_strides[cuda_detail::kMaxCudaRank]{};
    int64_t bias_strides[cuda_detail::kMaxCudaRank]{};
    int flat_weight{0}, flat_bias{0};
};

struct FloatView {
    const void* data{nullptr};
    DataType type{DataType::UNKNOWN};
};

struct DeviceAffine {
    cuda_detail::DeviceBuffer<float> fp32;
    cuda_detail::DeviceBuffer<uint16_t> fp16;
    cuda_detail::DeviceBuffer<BFloat16> bf16;
    FloatView view;

    int Acquire(const std::shared_ptr<Tensor>& tensor) {
        if (!tensor) return 0;
        view.type = tensor->data_type();
        int status = -1;
        switch (view.type) {
            case DataType::FP32:
                status = cuda_detail::CopyTensorToDevice(tensor.get(), &fp32); view.data = fp32.get(); break;
            case DataType::FP16:
                status = cuda_detail::CopyTensorToDevice(tensor.get(), &fp16); view.data = fp16.get(); break;
            case DataType::BF16:
                status = cuda_detail::CopyTensorToDevice(tensor.get(), &bf16); view.data = bf16.get(); break;
            default: return -1;
        }
        return status;
    }
};

__device__ float ReadAffine(FloatView value, int64_t index) {
    if (!value.data) return 0.0f;
    switch (value.type) {
        case DataType::FP16: return cuda_detail::ReadDevice(static_cast<const uint16_t*>(value.data), index);
        case DataType::BF16: return cuda_detail::ReadDevice(static_cast<const BFloat16*>(value.data), index);
        default: return cuda_detail::ReadDevice(static_cast<const float*>(value.data), index);
    }
}

__device__ int64_t BroadcastOffset(int64_t index, const AffineShape& shape, const int64_t* strides) {
    int64_t result = 0;
    for (int axis = 0; axis < shape.rank; ++axis) {
        const int64_t coordinate = index / shape.input_strides[axis];
        index %= shape.input_strides[axis];
        result += coordinate * strides[axis];
    }
    return result;
}

template <typename T>
__global__ void Normalize(const T* input, FloatView weight, FloatView bias, T* out, float* means,
                          float* inv_stds, int64_t rows, int64_t width, float epsilon,
                          float weight_offset, bool layer_norm, AffineShape affine) {
    __shared__ float partial_mean[cuda_detail::kCudaThreads];
    __shared__ float partial_sum[cuda_detail::kCudaThreads];
    __shared__ int64_t partial_count[cuda_detail::kCudaThreads];
    const int lane = static_cast<int>(threadIdx.x);
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const int64_t base = row * width;
        float mean = 0.0f, sum = 0.0f;
        int64_t count = 0;
        for (int64_t col = lane; col < width; col += blockDim.x) {
            const float value = cuda_detail::ReadDevice(input, base + col);
            ++count;
            if (layer_norm) {
                const float delta = value - mean;
                mean += delta / static_cast<float>(count);
                sum += delta * (value - mean);
            } else {
                sum += value * value;
            }
        }
        partial_mean[lane] = mean;
        partial_sum[lane] = sum;
        partial_count[lane] = count;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const int64_t left = partial_count[lane], right = partial_count[lane + stride];
                if (layer_norm && right != 0) {
                    const float delta = partial_mean[lane + stride] - partial_mean[lane];
                    const float fraction = static_cast<float>(right) / static_cast<float>(left + right);
                    partial_mean[lane] += delta * fraction;
                    partial_sum[lane] += partial_sum[lane + stride] + delta * delta * static_cast<float>(left) * fraction;
                } else {
                    partial_sum[lane] += partial_sum[lane + stride];
                }
                partial_count[lane] = left + right;
            }
            __syncthreads();
        }
        mean = layer_norm ? partial_mean[0] : 0.0f;
        const float inv_std = rsqrtf(partial_sum[0] / static_cast<float>(width) + epsilon);
        if (lane == 0) {
            if (means) means[row] = mean;
            if (inv_stds) inv_stds[row] = inv_std;
        }
        for (int64_t col = lane; col < width; col += blockDim.x) {
            const int64_t index = base + col;
            const int64_t wi = affine.flat_weight ? col : BroadcastOffset(index, affine, affine.weight_strides);
            const int64_t bi = affine.flat_bias ? col : BroadcastOffset(index, affine, affine.bias_strides);
            const float scale = ReadAffine(weight, wi) + weight_offset;
            cuda_detail::WriteDevice(out, index,
                (cuda_detail::ReadDevice(input, index) - mean) * inv_std * scale + ReadAffine(bias, bi));
        }
        __syncthreads();
    }
}

template <DataType dtype>
int32_t Run(operators::NormalizationParam* param) {
    operators::normalization_detail::Shape shape;
    if (!param || !operators::normalization_detail::ValidateStorage(*param, &shape) ||
        param->input->data_type() != dtype || param->input->dims().size() > cuda_detail::kMaxCudaRank) return -1;
    using T = cuda_detail::StorageT<dtype>;
    cuda_detail::DeviceBuffer<T> input, output;
    cuda_detail::DeviceBuffer<float> means, inv_stds;
    DeviceAffine weight, bias;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || weight.Acquire(param->weight) != 0 ||
        bias.Acquire(param->bias) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0 ||
        (param->mean && cuda_detail::AllocateTensorOnDevice(param->mean.get(), &means) != 0) ||
        (param->inv_std_dev && cuda_detail::AllocateTensorOnDevice(param->inv_std_dev.get(), &inv_stds) != 0)) return -1;
    AffineShape affine;
    const auto& dims = param->input->dims().data();
    affine.rank = static_cast<int>(dims.size());
    const std::vector<int64_t> suffix(dims.begin() + shape.axis, dims.end());
    affine.flat_weight = param->weight->dims().data() == suffix;
    affine.flat_bias = param->bias && param->bias->dims().data() == suffix;
    for (int axis = 0; axis < affine.rank; ++axis) {
        affine.input_strides[axis] = shape.input_strides[axis];
        affine.weight_strides[axis] = shape.weight_strides[axis];
        affine.bias_strides[axis] = shape.bias_strides[axis];
    }
    const int blocks = static_cast<int>(std::min<int64_t>(shape.rows, 65535));
    Normalize<T><<<blocks, cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), weight.view, bias.view, output.get(), means.get(), inv_stds.get(), shape.rows,
        shape.width, param->epsilon, param->weight_offset, param->layer_norm, affine);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0 ||
        cuda_detail::CopyDeviceToTensor(&output, param->out.get()) != 0 ||
        (param->mean && cuda_detail::CopyDeviceToTensor(&means, param->mean.get()) != 0) ||
        (param->inv_std_dev && cuda_detail::CopyDeviceToTensor(&inv_stds, param->inv_std_dev.get()) != 0)) return -1;
    return 0;
}

}  // namespace

template <> int32_t NormalizationKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::Normalization::FP32");
    return Run<DataType::FP32>(ParamAs<operators::NormalizationParam>());
}
template <> int32_t NormalizationKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    AutoTimer timer("CUDA::Normalization::FP16");
    return Run<DataType::FP16>(ParamAs<operators::NormalizationParam>());
}
template <> int32_t NormalizationKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    AutoTimer timer("CUDA::Normalization::BF16");
    return Run<DataType::BF16>(ParamAs<operators::NormalizationParam>());
}

namespace {
template <DataType dtype> void Register() {
    for (const char* name : {"RMSNormalization", "LayerNormalization"}) {
        KernelDispatcher::instance().registerKernel(DeviceType::CUDA, dtype, name,
            [] { return std::make_unique<NormalizationKernel<DeviceType::CUDA, dtype>>(); });
    }
}
bool registered = [] {
    Register<DataType::FP32>(); Register<DataType::FP16>(); Register<DataType::BF16>();
    return true;
}();
}  // namespace

void EnsureCudaNormalizationKernelsRegistered() { (void)registered; }

}  // namespace kernel
}  // namespace feather
