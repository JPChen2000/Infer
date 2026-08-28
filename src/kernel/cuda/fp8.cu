#include "src/kernel/batch_normalization.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/cos.h"
#include "src/kernel/equal.h"
#include "src/kernel/erf.h"
#include "src/kernel/exp.h"
#include "src/kernel/expand.h"
#include "src/kernel/flatten.h"
#include "src/kernel/gather.h"
#include "src/kernel/global_average_pool.h"
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_cuda_fp8_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
#define REGISTER_CUDA_FP8(op, klass)                                                                      \
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E4M3, op, []() {                          \
        return std::make_unique<klass<DeviceType::CUDA, DataType::FP8E4M3>>();                          \
    });                                                                                                   \
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP8E5M2, op, []() {                          \
        return std::make_unique<klass<DeviceType::CUDA, DataType::FP8E5M2>>();                          \
    });

    REGISTER_CUDA_FP8("BatchNormalization", BatchNormalizationKernel)
    REGISTER_CUDA_FP8("Concat", ConcatKernel)
    REGISTER_CUDA_FP8("Conv2D", Conv2DKernel)
    REGISTER_CUDA_FP8("Cos", CosKernel)
    REGISTER_CUDA_FP8("Equal", EqualKernel)
    REGISTER_CUDA_FP8("Erf", ErfKernel)
    REGISTER_CUDA_FP8("Exp", ExpKernel)
    REGISTER_CUDA_FP8("Expand", ExpandKernel)
    REGISTER_CUDA_FP8("Flatten", FlattenKernel)
    REGISTER_CUDA_FP8("Gather", GatherKernel)
    REGISTER_CUDA_FP8("GlobalAveragePool", GlobalAveragePoolKernel)
    REGISTER_CUDA_FP8("Identity", IdentityKernel)
    REGISTER_CUDA_FP8("Neg", NegKernel)
    REGISTER_CUDA_FP8("AvgPool", AvgPoolKernel)
    REGISTER_CUDA_FP8("MaxPool", MaxPoolKernel)
    REGISTER_CUDA_FP8("Pow", PowKernel)
    REGISTER_CUDA_FP8("ReduceMean", ReduceMeanKernel)
    REGISTER_CUDA_FP8("ReduceSum", ReduceSumKernel)
    REGISTER_CUDA_FP8("ReLU", ReluKernel)
    REGISTER_CUDA_FP8("Reshape", ReshapeKernel)
    REGISTER_CUDA_FP8("Resize", ResizeKernel)
    REGISTER_CUDA_FP8("ResizeConcat", ResizeConcatKernel)
    REGISTER_CUDA_FP8("Sigmoid", SigmoidKernel)
    REGISTER_CUDA_FP8("SiLU", SiluKernel)
    REGISTER_CUDA_FP8("Sin", SinKernel)
    REGISTER_CUDA_FP8("Slice", SliceKernel)
    REGISTER_CUDA_FP8("Softmax", SoftmaxKernel)
    REGISTER_CUDA_FP8("Softplus", SoftplusKernel)
    REGISTER_CUDA_FP8("Split", SplitKernel)
    REGISTER_CUDA_FP8("Sqrt", SqrtKernel)
    REGISTER_CUDA_FP8("Tanh", TanhKernel)
    REGISTER_CUDA_FP8("Transpose", TransposeKernel)
    REGISTER_CUDA_FP8("Unsqueeze", UnsqueezeKernel)
    REGISTER_CUDA_FP8("Squeeze", SqueezeKernel)
    REGISTER_CUDA_FP8("Where", WhereKernel)
#undef REGISTER_CUDA_FP8
    return true;
}();

inline bool HasValidFp8Shape(const std::vector<int64_t>& dims, bool allow_scalar = true) {
    if (!allow_scalar && dims.empty()) return false;
    int64_t product = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0) return false;
        if (product > std::numeric_limits<int64_t>::max() / dim) return false;
        product *= dim;
    }
    return true;
}

inline bool SafeFp8Numel(const Tensor* tensor, int64_t* numel) {
    if (tensor == nullptr || numel == nullptr || !HasValidFp8Shape(tensor->dims().data())) return false;
    int64_t product = 1;
    for (const int64_t dim : tensor->dims().data()) {
        if (product > std::numeric_limits<int64_t>::max() / dim) return false;
        product *= dim;
    }
    if (tensor->numel() != product) return false;
    *numel = product;
    return true;
}

inline bool IsValidFp8Scale(const Tensor* tensor) {
    return tensor != nullptr && HasCompatiblePerTensorQuantization(tensor->quantization()) &&
           std::isfinite(tensor->quantization_scale()) && tensor->quantization_scale() > 0.0f;
}

inline bool TryFp8WindowOutputDimension(int64_t input_dim, int64_t kernel, int64_t stride, int64_t pad,
                                        int64_t dilation, int64_t* output_dim) {
    if (output_dim == nullptr || input_dim <= 0 || kernel <= 0 || stride <= 0 || pad < 0 || dilation <= 0 ||
        pad > std::numeric_limits<int64_t>::max() / 2 ||
        kernel - 1 > (std::numeric_limits<int64_t>::max() - 1) / dilation ||
        input_dim > std::numeric_limits<int64_t>::max() - 2 * pad) return false;
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;
    const int64_t padded = input_dim + 2 * pad;
    if (padded < input_dim || padded < effective_kernel) return false;
    const int64_t result = (padded - effective_kernel) / stride + 1;
    if (result <= 0) return false;
    *output_dim = result;
    return true;
}

inline bool TryFp8ScaledDimension(int64_t input_dim, float scale, int64_t* output_dim) {
    if (output_dim == nullptr || input_dim <= 0 || !std::isfinite(scale) || scale <= 0.0f) return false;
    const long double scaled = static_cast<long double>(input_dim) * static_cast<long double>(scale);
    const long double rounded = std::floor(scaled + 0.5L);
    if (!std::isfinite(static_cast<double>(scaled)) || rounded < 1.0L ||
        rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) return false;
    *output_dim = static_cast<int64_t>(rounded);
    return true;
}

template <DataType dtype>
inline bool IsValidFp8Tensor(const Tensor* tensor) {
    int64_t numel = 0;
    if (tensor == nullptr || !tensor->IsInitialized() || tensor->data_type() != dtype || !SafeFp8Numel(tensor, &numel) ||
        !IsValidFp8Scale(tensor)) {
        return false;
    }
    const auto count = static_cast<uint64_t>(numel);
    const auto element_bytes = DataTypeBytes(dtype);
    return element_bytes != 0 && count <= std::numeric_limits<size_t>::max() / element_bytes &&
           tensor->memory_size() >= static_cast<size_t>(count) * element_bytes;
}

template <DataType dtype>
inline bool IsValidFp8Output(Tensor* tensor, const std::vector<int64_t>* expected_dims = nullptr) {
    int64_t numel = 0;
    if (tensor == nullptr || !tensor->IsInitialized() || !HasValidFp8Shape(tensor->dims().data()) ||
        (tensor->data_type() != DataType::UNKNOWN && tensor->data_type() != dtype)) {
        return false;
    }
    if (expected_dims != nullptr &&
        (!HasValidFp8Shape(*expected_dims) || tensor->dims().data() != *expected_dims)) return false;
    if (!SafeFp8Numel(tensor, &numel) || !IsValidFp8Scale(tensor)) return false;
    const auto count = static_cast<uint64_t>(numel);
    const auto element_bytes = DataTypeBytes(dtype);
    return element_bytes != 0 && count <= std::numeric_limits<size_t>::max() / element_bytes &&
           tensor->memory_size() >= static_cast<size_t>(count) * element_bytes;
}

template <typename T, int Op>
__global__ void Fp8UnaryKernelCuda(const T* input, T* output, int64_t numel, float exponent, float input_scale,
                                   float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) return;
    const float x = cuda_detail::ReadDevice(input, index, input_scale);
    float y = x;
    if constexpr (Op == 0) y = fmaxf(x, 0.0f);
    if constexpr (Op == 1) y = 1.0f / (1.0f + expf(-x));
    if constexpr (Op == 2) y = x / (1.0f + expf(-x));
    if constexpr (Op == 3) y = tanhf(x);
    if constexpr (Op == 4) y = expf(x);
    if constexpr (Op == 5) y = sinf(x);
    if constexpr (Op == 6) y = cosf(x);
    if constexpr (Op == 7) y = erff(x);
    if constexpr (Op == 8) y = sqrtf(x);
    if constexpr (Op == 9) y = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    if constexpr (Op == 10) y = -x;
    if constexpr (Op == 11) y = powf(x, exponent);
    cuda_detail::WriteDevice(output, index, y, output_scale);
}

template <typename T, int Op>
__global__ void Fp8BinaryKernelCuda(const T* lhs, const T* rhs, T* output, int64_t numel,
                                    cuda_detail::CudaShape output_shape, cuda_detail::CudaShape lhs_shape,
                                    cuda_detail::CudaShape rhs_shape, float lhs_scale, float rhs_scale,
                                    float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) return;
    int64_t remaining = linear;
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;
    const int lhs_gap = output_shape.rank - lhs_shape.rank;
    const int rhs_gap = output_shape.rank - rhs_shape.rank;
    for (int axis = 0; axis < output_shape.rank; ++axis) {
        const int64_t coordinate = remaining / output_shape.strides[axis];
        remaining %= output_shape.strides[axis];
        const int lhs_axis = axis - lhs_gap;
        const int rhs_axis = axis - rhs_gap;
        if (lhs_axis >= 0 && lhs_shape.dims[lhs_axis] != 1) lhs_offset += coordinate * lhs_shape.strides[lhs_axis];
        if (rhs_axis >= 0 && rhs_shape.dims[rhs_axis] != 1) rhs_offset += coordinate * rhs_shape.strides[rhs_axis];
    }
    const float a = cuda_detail::ReadDevice(lhs, lhs_offset, lhs_scale);
    const float b = cuda_detail::ReadDevice(rhs, rhs_offset, rhs_scale);
    float result = a;
    if constexpr (Op == 0) result = a + b;
    if constexpr (Op == 1) result = a * b;
    if constexpr (Op == 2) result = a - b;
    if constexpr (Op == 3) result = a / b;
    cuda_detail::WriteDevice(output, linear, result, output_scale);
}

template <typename T>
__global__ void Fp8CopyKernelCuda(const T* input, T* output, int64_t numel, float input_scale, float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) cuda_detail::WriteDevice(output, index, cuda_detail::ReadDevice(input, index, input_scale), output_scale);
}

struct Fp8TransposePerm {
    int values[cuda_detail::kMaxCudaRank]{};
};

template <typename T>
__global__ void Fp8TransposeKernelCuda(const T* input, T* output, int64_t numel, cuda_detail::CudaShape input_shape,
                                       cuda_detail::CudaShape output_shape, Fp8TransposePerm perm,
                                       float input_scale, float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) return;
    int64_t remaining = linear;
    int64_t input_offset = 0;
    for (int axis = 0; axis < output_shape.rank; ++axis) {
        const int64_t coordinate = remaining / output_shape.strides[axis];
        remaining %= output_shape.strides[axis];
        input_offset += coordinate * input_shape.strides[perm.values[axis]];
    }
    cuda_detail::WriteDevice(output, linear, cuda_detail::ReadDevice(input, input_offset, input_scale), output_scale);
}

template <typename T>
__global__ void Fp8ConcatKernelCuda(const T* input, T* output, int64_t total, int64_t input_axis, int64_t inner,
                                    int64_t output_axis, int64_t axis_offset, float input_scale, float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t axis_index = (linear / inner) % input_axis;
    const int64_t outer_index = linear / (input_axis * inner);
    const int64_t inner_index = linear % inner;
    const int64_t output_index = (outer_index * output_axis + axis_offset + axis_index) * inner + inner_index;
    cuda_detail::WriteDevice(output, output_index, cuda_detail::ReadDevice(input, linear, input_scale), output_scale);
}

template <typename T>
__global__ void Fp8SplitKernelCuda(const T* input, T* output, int64_t total, int64_t output_axis, int64_t inner,
                                   int64_t input_axis, int64_t axis_offset, float input_scale, float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t axis_index = (linear / inner) % output_axis;
    const int64_t outer_index = linear / (output_axis * inner);
    const int64_t input_index = (outer_index * input_axis + axis_offset + axis_index) * inner + linear % inner;
    cuda_detail::WriteDevice(output, linear, cuda_detail::ReadDevice(input, input_index, input_scale), output_scale);
}

template <typename T>
__global__ void Fp8SliceKernelCuda(const T* input, T* output, int64_t numel, cuda_detail::CudaShape input_shape,
                                   cuda_detail::CudaShape output_shape, int axis, int64_t start, float input_scale,
                                   float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) return;
    int64_t remaining = linear;
    int64_t input_offset = 0;
    for (int i = 0; i < output_shape.rank; ++i) {
        int64_t coordinate = remaining / output_shape.strides[i];
        remaining %= output_shape.strides[i];
        if (i == axis) coordinate += start;
        input_offset += coordinate * input_shape.strides[i];
    }
    cuda_detail::WriteDevice(output, linear, cuda_detail::ReadDevice(input, input_offset, input_scale), output_scale);
}

template <typename T>
__global__ void Fp8SoftmaxKernelCuda(const T* input, T* output, int64_t vectors, int64_t axis_dim, int64_t inner,
                                     float input_scale, float output_scale) {
    const int64_t vector_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vectors) return;
    const int64_t outer_index = vector_index / inner;
    const int64_t inner_index = vector_index % inner;
    const int64_t base = outer_index * axis_dim * inner + inner_index;
    float max_value = -3.402823466e+38F;
    for (int64_t axis = 0; axis < axis_dim; ++axis) max_value = fmaxf(max_value, cuda_detail::ReadDevice(input, base + axis * inner, input_scale));
    float sum = 0.0f;
    for (int64_t axis = 0; axis < axis_dim; ++axis) sum += expf(cuda_detail::ReadDevice(input, base + axis * inner, input_scale) - max_value);
    for (int64_t axis = 0; axis < axis_dim; ++axis) {
        const float value = expf(cuda_detail::ReadDevice(input, base + axis * inner, input_scale) - max_value) / sum;
        cuda_detail::WriteDevice(output, base + axis * inner, value, output_scale);
    }
}

struct Fp8ReduceSpec {
    cuda_detail::CudaShape input_shape{};
    cuda_detail::CudaShape output_shape{};
    int reduced_axes[cuda_detail::kMaxCudaRank]{};
    int reduced_rank{0};
    int keepdims{0};
    int64_t reduce_count{1};
};

template <typename T, bool kMean>
__global__ void Fp8ReduceKernelCuda(const T* input, T* output, int64_t output_numel, Fp8ReduceSpec spec,
                                    float input_scale, float output_scale) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) return;
    int64_t output_remaining = output_index;
    int64_t output_coordinates[cuda_detail::kMaxCudaRank]{};
    for (int axis = 0; axis < spec.output_shape.rank; ++axis) {
        output_coordinates[axis] = output_remaining / spec.output_shape.strides[axis];
        output_remaining %= spec.output_shape.strides[axis];
    }
    int reduced_mask[cuda_detail::kMaxCudaRank]{};
    for (int i = 0; i < spec.reduced_rank; ++i) reduced_mask[spec.reduced_axes[i]] = 1;
    int64_t base_coordinates[cuda_detail::kMaxCudaRank]{};
    int output_axis = 0;
    for (int axis = 0; axis < spec.input_shape.rank; ++axis) {
        if (reduced_mask[axis]) {
            base_coordinates[axis] = 0;
            if (spec.keepdims) ++output_axis;
        } else {
            base_coordinates[axis] = output_coordinates[output_axis++];
        }
    }
    float sum = 0.0f;
    for (int64_t reduction = 0; reduction < spec.reduce_count; ++reduction) {
        int64_t reduction_remaining = reduction;
        int64_t input_offset = 0;
        for (int axis = 0; axis < spec.input_shape.rank; ++axis) {
            int64_t coordinate = base_coordinates[axis];
            if (reduced_mask[axis]) {
                const int64_t dim = spec.input_shape.dims[axis];
                coordinate = reduction_remaining % dim;
                reduction_remaining /= dim;
            }
            input_offset += coordinate * spec.input_shape.strides[axis];
        }
        sum += cuda_detail::ReadDevice(input, input_offset, input_scale);
    }
    if constexpr (kMean) sum /= static_cast<float>(spec.reduce_count);
    cuda_detail::WriteDevice(output, output_index, sum, output_scale);
}

__device__ inline int64_t Fp8BroadcastOffset(int64_t output_index, const cuda_detail::CudaShape& output_shape,
                                             const cuda_detail::CudaShape& input_shape) {
    int64_t remaining = output_index;
    int64_t offset = 0;
    const int gap = output_shape.rank - input_shape.rank;
    for (int axis = 0; axis < output_shape.rank; ++axis) {
        const int64_t coordinate = remaining / output_shape.strides[axis];
        remaining %= output_shape.strides[axis];
        const int input_axis = axis - gap;
        if (input_axis >= 0 && input_shape.dims[input_axis] != 1) offset += coordinate * input_shape.strides[input_axis];
    }
    return offset;
}

template <typename T, typename IndexT>
__global__ void Fp8GatherKernelCuda(const T* data, const IndexT* indices, T* output, int64_t output_numel,
                                    cuda_detail::CudaShape data_shape, cuda_detail::CudaShape indices_shape,
                                    cuda_detail::CudaShape output_shape, int axis, float data_scale,
                                    float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= output_numel) return;
    int64_t remaining = linear;
    int64_t output_coords[cuda_detail::kMaxCudaRank]{};
    for (int i = 0; i < output_shape.rank; ++i) {
        output_coords[i] = remaining / output_shape.strides[i];
        remaining %= output_shape.strides[i];
    }
    int64_t indices_offset = 0;
    for (int i = 0; i < indices_shape.rank; ++i) indices_offset += output_coords[axis + i] * indices_shape.strides[i];
    int64_t index = indices[indices_offset];
    if (index < 0) index += data_shape.dims[axis];
    int64_t data_offset = 0;
    for (int i = 0; i < axis; ++i) data_offset += output_coords[i] * data_shape.strides[i];
    data_offset += index * data_shape.strides[axis];
    for (int i = axis + 1; i < data_shape.rank; ++i) data_offset += output_coords[i - 1 + indices_shape.rank] * data_shape.strides[i];
    cuda_detail::WriteDevice(output, linear, cuda_detail::ReadDevice(data, data_offset, data_scale), output_scale);
}

template <typename T>
__global__ void Fp8WhereKernelCuda(const uint8_t* condition, const T* x, const T* y, T* output, int64_t numel,
                                   cuda_detail::CudaShape output_shape, cuda_detail::CudaShape condition_shape,
                                   cuda_detail::CudaShape x_shape, cuda_detail::CudaShape y_shape, float x_scale,
                                   float y_scale, float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) return;
    const int64_t condition_offset = Fp8BroadcastOffset(linear, output_shape, condition_shape);
    const int64_t x_offset = Fp8BroadcastOffset(linear, output_shape, x_shape);
    const int64_t y_offset = Fp8BroadcastOffset(linear, output_shape, y_shape);
    const float value = condition[condition_offset] != 0 ? cuda_detail::ReadDevice(x, x_offset, x_scale)
                                                          : cuda_detail::ReadDevice(y, y_offset, y_scale);
    cuda_detail::WriteDevice(output, linear, value, output_scale);
}

template <typename T>
__global__ void Fp8EqualKernelCuda(const T* lhs, const T* rhs, uint8_t* output, int64_t numel,
                                   cuda_detail::CudaShape output_shape, cuda_detail::CudaShape lhs_shape,
                                   cuda_detail::CudaShape rhs_shape, float lhs_scale, float rhs_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) return;
    const float lhs_value = cuda_detail::ReadDevice(lhs, Fp8BroadcastOffset(linear, output_shape, lhs_shape), lhs_scale);
    const float rhs_value = cuda_detail::ReadDevice(rhs, Fp8BroadcastOffset(linear, output_shape, rhs_shape), rhs_scale);
    output[linear] = lhs_value == rhs_value ? 1 : 0;
}

template <typename T>
__global__ void Fp8ExpandKernelCuda(const T* input, T* output, int64_t numel, cuda_detail::CudaShape output_shape,
                                    cuda_detail::CudaShape input_shape, float input_scale, float output_scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        const int64_t input_offset = Fp8BroadcastOffset(linear, output_shape, input_shape);
        cuda_detail::WriteDevice(output, linear, cuda_detail::ReadDevice(input, input_offset, input_scale), output_scale);
    }
}

__device__ inline int64_t Fp8ImageOffset(int layout, int64_t n, int64_t c, int64_t h, int64_t w, int64_t channels,
                                         int64_t height, int64_t width) {
    return layout == static_cast<int>(DataLayout::NHWC) ? ((n * height + h) * width + w) * channels + c
                                                         : ((n * channels + c) * height + h) * width + w;
}

__device__ inline int64_t Fp8ClampCoordinate(int64_t value, int64_t upper_bound) {
    if (value < 0) return 0;
    if (value >= upper_bound) return upper_bound - 1;
    return value;
}

template <typename T, bool kMax>
__global__ void Fp8PoolKernelCuda(const T* input, T* output, int64_t total, int64_t batch, int64_t channels,
                                  int64_t input_h, int64_t input_w, int64_t output_h, int64_t output_w,
                                  int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                  int64_t pad_h, int64_t pad_w, int layout, float input_scale, float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int64_t n = 0;
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    int64_t remaining = index;
    if (layout == static_cast<int>(DataLayout::NHWC)) {
        c = remaining % channels;
        remaining /= channels;
        ow = remaining % output_w;
        remaining /= output_w;
        oh = remaining % output_h;
        n = remaining / output_h;
    } else {
        ow = remaining % output_w;
        remaining /= output_w;
        oh = remaining % output_h;
        remaining /= output_h;
        c = remaining % channels;
        n = remaining / channels;
    }
    float value = kMax ? -3.402823466e+38F : 0.0f;
    int count = 0;
    for (int64_t kh = 0; kh < kernel_h; ++kh) for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int64_t ih = oh * stride_h + kh - pad_h;
        const int64_t iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) continue;
        const float x = cuda_detail::ReadDevice(input, Fp8ImageOffset(layout, n, c, ih, iw, channels, input_h, input_w), input_scale);
        if constexpr (kMax) value = fmaxf(value, x); else { value += x; ++count; }
    }
    if constexpr (!kMax) value = count == 0 ? 0.0f : value / static_cast<float>(count);
    const int64_t output_offset = Fp8ImageOffset(layout, n, c, oh, ow, channels, output_h, output_w);
    cuda_detail::WriteDevice(output, output_offset, value, output_scale);
}

template <typename T>
__global__ void Fp8GlobalAveragePoolKernelCuda(const T* input, T* output, int64_t total, int64_t channels,
                                               int64_t height, int64_t width, int layout, float input_scale,
                                               float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t c = index % channels;
    const int64_t n = index / channels;
    float sum = 0.0f;
    for (int64_t h = 0; h < height; ++h) for (int64_t w = 0; w < width; ++w)
        sum += cuda_detail::ReadDevice(input, Fp8ImageOffset(layout, n, c, h, w, channels, height, width), input_scale);
    const int64_t output_index = Fp8ImageOffset(layout, n, c, 0, 0, channels, 1, 1);
    cuda_detail::WriteDevice(output, output_index, sum / static_cast<float>(height * width), output_scale);
}

template <typename T>
__global__ void Fp8BatchNormKernelCuda(const T* input, const T* scale, const T* bias, const T* mean, const T* variance,
                                       T* output, int64_t total, int64_t channels, int64_t height, int64_t width,
                                       int layout, float epsilon, float input_scale, float scale_scale,
                                       float bias_scale, float mean_scale, float variance_scale, float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int64_t remaining = index;
    int64_t c = 0;
    if (layout == static_cast<int>(DataLayout::NHWC)) {
        c = remaining % channels;
    } else {
        remaining /= width;
        remaining /= height;
        c = remaining % channels;
    }
    const float x = cuda_detail::ReadDevice(input, index, input_scale);
    const float s = cuda_detail::ReadDevice(scale, c, scale_scale);
    const float b = cuda_detail::ReadDevice(bias, c, bias_scale);
    const float m = cuda_detail::ReadDevice(mean, c, mean_scale);
    const float v = cuda_detail::ReadDevice(variance, c, variance_scale);
    cuda_detail::WriteDevice(output, index, (x - m) / sqrtf(v + epsilon) * s + b, output_scale);
}

template <typename T>
__global__ void Fp8Conv2DKernelCuda(const T* input, const T* weight, const T* bias, T* output, int64_t total,
                                    int64_t input_channels, int64_t input_height, int64_t input_width,
                                    int64_t output_channels, int64_t output_height, int64_t output_width,
                                    int64_t kernel_channels, int64_t kernel_height, int64_t kernel_width,
                                    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
                                    int64_t dilation_h, int64_t dilation_w, int64_t group, int layout,
                                    float input_scale, float weight_scale, float bias_scale, float output_scale,
                                    int64_t bias_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int64_t remaining = index;
    int64_t n = 0, oc = 0, oh = 0, ow = 0;
    if (layout == static_cast<int>(DataLayout::NHWC)) {
        oc = remaining % output_channels;
        remaining /= output_channels;
        ow = remaining % output_width;
        remaining /= output_width;
        oh = remaining % output_height;
        n = remaining / output_height;
    } else {
        ow = remaining % output_width;
        remaining /= output_width;
        oh = remaining % output_height;
        remaining /= output_height;
        oc = remaining % output_channels;
        n = remaining / output_channels;
    }
    const int64_t out_channels_per_group = output_channels / group;
    const int64_t in_channels_per_group = input_channels / group;
    const int64_t current_group = oc / out_channels_per_group;
    float sum = 0.0f;
    for (int64_t ic = 0; ic < kernel_channels; ++ic) {
        const int64_t global_ic = current_group * in_channels_per_group + ic;
        for (int64_t kh = 0; kh < kernel_height; ++kh) for (int64_t kw = 0; kw < kernel_width; ++kw) {
            const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
            const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
            if (ih < 0 || ih >= input_height || iw < 0 || iw >= input_width) continue;
            const int64_t input_offset = Fp8ImageOffset(layout, n, global_ic, ih, iw, input_channels, input_height, input_width);
            const int64_t weight_offset = ((oc * kernel_channels + ic) * kernel_height + kh) * kernel_width + kw;
            sum += cuda_detail::ReadDevice(input, input_offset, input_scale) * cuda_detail::ReadDevice(weight, weight_offset, weight_scale);
        }
    }
    if (bias != nullptr && bias_count == output_channels) sum += cuda_detail::ReadDevice(bias, oc, bias_scale);
    cuda_detail::WriteDevice(output, index, sum, output_scale);
}

struct Fp8ResizeSpec {
    cuda_detail::CudaShape input_shape{};
    cuda_detail::CudaShape output_shape{};
    float scales[cuda_detail::kMaxCudaRank]{};
};

template <typename T>
__global__ void Fp8ResizeKernelCuda(const T* input, T* output, int64_t numel, Fp8ResizeSpec spec, float input_scale,
                                    float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) return;
    int64_t remaining = index;
    int64_t input_offset = 0;
    for (int axis = 0; axis < spec.output_shape.rank; ++axis) {
        const int64_t coordinate = remaining / spec.output_shape.strides[axis];
        remaining %= spec.output_shape.strides[axis];
        int64_t input_coordinate = static_cast<int64_t>(static_cast<double>(coordinate) / spec.scales[axis]);
        input_coordinate = Fp8ClampCoordinate(input_coordinate, spec.input_shape.dims[axis]);
        input_offset += input_coordinate * spec.input_shape.strides[axis];
    }
    cuda_detail::WriteDevice(output, index, cuda_detail::ReadDevice(input, input_offset, input_scale), output_scale);
}

template <typename T>
__global__ void Fp8ResizeConcatKernelCuda(const T* resize_input, const T* concat_input, T* output, int64_t total,
                                          int64_t resize_channels, int64_t concat_channels, int64_t resize_height,
                                          int64_t resize_width, int64_t output_height, int64_t output_width,
                                          float scale_h, float scale_w, int resize_input_index, int layout,
                                          float resize_scale, float concat_scale, float output_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t output_channels = resize_channels + concat_channels;
    int64_t n = 0, oc = 0, oh = 0, ow = 0;
    int64_t remaining = index;
    if (layout == static_cast<int>(DataLayout::NHWC)) {
        oc = remaining % output_channels;
        remaining /= output_channels;
        ow = remaining % output_width;
        remaining /= output_width;
        oh = remaining % output_height;
        n = remaining / output_height;
    } else {
        ow = remaining % output_width;
        remaining /= output_width;
        oh = remaining % output_height;
        remaining /= output_height;
        oc = remaining % output_channels;
        n = remaining / output_channels;
    }
    const bool use_resize = resize_input_index == 0 ? oc < resize_channels : oc >= concat_channels;
    float value = 0.0f;
    if (use_resize) {
        const int64_t rc = resize_input_index == 0 ? oc : oc - concat_channels;
        const int64_t ih = Fp8ClampCoordinate(static_cast<int64_t>(oh / scale_h), resize_height);
        const int64_t iw = Fp8ClampCoordinate(static_cast<int64_t>(ow / scale_w), resize_width);
        value = cuda_detail::ReadDevice(resize_input, Fp8ImageOffset(layout, n, rc, ih, iw, resize_channels, resize_height, resize_width), resize_scale);
    } else {
        const int64_t cc = resize_input_index == 0 ? oc - resize_channels : oc;
        value = cuda_detail::ReadDevice(concat_input, Fp8ImageOffset(layout, n, cc, oh, ow, concat_channels, output_height, output_width), concat_scale);
    }
    cuda_detail::WriteDevice(output, index, value, output_scale);
}

template <DataType dtype, int Op>
int32_t RunFp8Unary(operators::UnaryParam* param, const char* timer_name, float exponent = 1.0f) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        !IsValidFp8Output<dtype>(param->out.get(), &param->input->dims().data())) return -1;
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    const int64_t numel = param->input->numel();
    Fp8UnaryKernelCuda<T, Op><<<static_cast<int>(cuda_detail::DivUp(numel, cuda_detail::kCudaThreads)),
                                cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), numel, exponent, param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype, int Op>
int32_t RunFp8Binary(operators::BinaryParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->lhs.get()) ||
        !IsValidFp8Tensor<dtype>(param->rhs.get())) return -1;
    std::vector<int64_t> output_dims;
    if (!cuda_detail::InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        !IsValidFp8Output<dtype>(param->out.get(), &output_dims)) return -1;
    cuda_detail::CudaShape output_shape{}, lhs_shape{}, rhs_shape{};
    if (!cuda_detail::MakeCudaShape(output_dims, &output_shape) || !cuda_detail::MakeCudaShape(param->lhs->dims().data(), &lhs_shape) ||
        !cuda_detail::MakeCudaShape(param->rhs->dims().data(), &rhs_shape)) return -1;
    cuda_detail::DeviceBuffer<T> lhs, rhs, output;
    if (cuda_detail::CopyTensorToDevice(param->lhs.get(), &lhs) != 0 || cuda_detail::CopyTensorToDevice(param->rhs.get(), &rhs) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8BinaryKernelCuda<T, Op><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                 cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        lhs.get(), rhs.get(), output.get(), param->out->numel(), output_shape, lhs_shape, rhs_shape,
        param->lhs->quantization_scale(), param->rhs->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype, typename Param>
int32_t RunFp8Copy(Param* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        !IsValidFp8Output<dtype>(param->out.get()) || param->input->numel() != param->out->numel()) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8CopyKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->input->numel(), cuda_detail::kCudaThreads)),
                           cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->input->numel(), param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Transpose(operators::TransposeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->out == nullptr ||
        param->perm.size() != param->input->dims().size() || param->out->dims().size() != param->perm.size() ||
        param->input->dims().size() > cuda_detail::kMaxCudaRank) return -1;
    if (!HasValidFp8Shape(param->input->dims().data()) || !HasValidFp8Shape(param->out->dims().data())) return -1;
    std::vector<bool> seen(param->perm.size(), false);
    for (size_t i = 0; i < param->perm.size(); ++i) {
        const int64_t source = param->perm[i];
        if (source < 0 || source >= static_cast<int64_t>(param->perm.size()) || seen[static_cast<size_t>(source)] ||
            param->out->dims()[i] != param->input->dims()[static_cast<size_t>(source)]) return -1;
        seen[static_cast<size_t>(source)] = true;
    }
    if (!IsValidFp8Output<dtype>(param->out.get())) return -1;
    cuda_detail::CudaShape input_shape{}, output_shape{};
    if (!cuda_detail::MakeCudaShape(param->input->dims().data(), &input_shape) || !cuda_detail::MakeCudaShape(param->out->dims().data(), &output_shape)) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8TransposePerm perm{};
    for (size_t i = 0; i < param->perm.size(); ++i) perm.values[i] = static_cast<int>(param->perm[i]);
    Fp8TransposeKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), input_shape, output_shape, perm,
        param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Concat(operators::ConcatParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) return -1;
    const auto output_dims = param->out->dims().data();
    const int axis = param->axis < 0 ? param->axis + static_cast<int>(output_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(output_dims.size()) || output_dims.size() > cuda_detail::kMaxCudaRank ||
        !HasValidFp8Shape(output_dims) || !IsValidFp8Output<dtype>(param->out.get())) return -1;
    const int64_t outer = cuda_detail::ComputeProduct(output_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = cuda_detail::ComputeProduct(output_dims, static_cast<size_t>(axis) + 1, output_dims.size());
    const int64_t output_axis = output_dims[static_cast<size_t>(axis)];
    int64_t input_axis_sum = 0;
    for (const auto& tensor : param->inputs) {
        if (!IsValidFp8Tensor<dtype>(tensor.get()) || tensor->dims().size() != output_dims.size() ||
            !HasValidFp8Shape(tensor->dims().data())) return -1;
        for (size_t i = 0; i < output_dims.size(); ++i) {
            if (static_cast<int>(i) != axis && tensor->dims()[i] != output_dims[i]) return -1;
        }
        const int64_t input_axis = tensor->dims()[static_cast<size_t>(axis)];
        if (input_axis_sum > std::numeric_limits<int64_t>::max() - input_axis) return -1;
        input_axis_sum += input_axis;
    }
    if (input_axis_sum != output_axis) return -1;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    int64_t axis_offset = 0;
    for (const auto& tensor : param->inputs) {
        cuda_detail::DeviceBuffer<T> input;
        if (cuda_detail::CopyTensorToDevice(tensor.get(), &input) != 0) return -1;
        const int64_t input_axis = tensor->dims()[static_cast<size_t>(axis)];
        const int64_t total = outer * input_axis * inner;
        Fp8ConcatKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                 cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), output.get(), total, input_axis, inner, output_axis, axis_offset,
            tensor->quantization_scale(), param->out->quantization_scale());
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
        axis_offset += input_axis;
    }
    return axis_offset == output_axis ? cuda_detail::CopyDeviceToTensor(&output, param->out.get()) : -1;
}

template <DataType dtype>
int32_t RunFp8Split(operators::SplitParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->outputs.empty() ||
        param->input->dims().size() > cuda_detail::kMaxCudaRank || !HasValidFp8Shape(param->input->dims().data())) return -1;
    const auto input_dims = param->input->dims().data();
    const int axis = param->axis < 0 ? param->axis + static_cast<int>(input_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(input_dims.size())) return -1;
    const int64_t outer = cuda_detail::ComputeProduct(input_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = cuda_detail::ComputeProduct(input_dims, static_cast<size_t>(axis) + 1, input_dims.size());
    const int64_t input_axis = input_dims[static_cast<size_t>(axis)];
    int64_t output_axis_sum = 0;
    for (const auto& tensor : param->outputs) {
        if (tensor == nullptr || tensor->dims().size() != input_dims.size() ||
            !HasValidFp8Shape(tensor->dims().data()) || !IsValidFp8Output<dtype>(tensor.get())) return -1;
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (static_cast<int>(i) != axis && tensor->dims()[i] != input_dims[i]) return -1;
        }
        const int64_t output_axis = tensor->dims()[static_cast<size_t>(axis)];
        if (output_axis_sum > std::numeric_limits<int64_t>::max() - output_axis) return -1;
        output_axis_sum += output_axis;
    }
    if (output_axis_sum != input_axis) return -1;
    cuda_detail::DeviceBuffer<T> input;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0) return -1;
    int64_t axis_offset = 0;
    for (const auto& tensor : param->outputs) {
        const int64_t output_axis = tensor->dims()[static_cast<size_t>(axis)];
        const int64_t total = outer * output_axis * inner;
        cuda_detail::DeviceBuffer<T> output;
        if (cuda_detail::AllocateTensorOnDevice(tensor.get(), &output) != 0) return -1;
        Fp8SplitKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), output.get(), total, output_axis, inner, input_axis, axis_offset,
            param->input->quantization_scale(), tensor->quantization_scale());
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0 || cuda_detail::CopyDeviceToTensor(&output, tensor.get()) != 0) return -1;
        axis_offset += output_axis;
    }
    return axis_offset == input_axis ? 0 : -1;
}

template <DataType dtype>
int32_t RunFp8Slice(operators::SliceParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        param->input->dims().size() > cuda_detail::kMaxCudaRank || param->out == nullptr ||
        param->out->dims().size() != param->input->dims().size()) return -1;
    const auto input_dims = param->input->dims().data();
    const int rank = static_cast<int>(input_dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    int64_t start = param->start < 0 ? param->start + input_dims[axis] : param->start;
    int64_t end = param->end < 0 ? param->end + input_dims[axis] : param->end;
    start = std::max<int64_t>(0, std::min<int64_t>(input_dims[axis], start));
    end = std::max<int64_t>(start, std::min<int64_t>(input_dims[axis], end));
    if (param->out->dims()[axis] != end - start) return -1;
    for (int i = 0; i < rank; ++i) {
        if (i != axis && param->out->dims()[static_cast<size_t>(i)] != input_dims[static_cast<size_t>(i)]) return -1;
    }
    if (!IsValidFp8Output<dtype>(param->out.get())) return -1;
    cuda_detail::CudaShape input_shape{}, output_shape{};
    if (!cuda_detail::MakeCudaShape(input_dims, &input_shape) || !cuda_detail::MakeCudaShape(param->out->dims().data(), &output_shape)) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8SliceKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                            cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), input_shape, output_shape, axis, start,
        param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Softmax(operators::SoftmaxParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        !IsValidFp8Output<dtype>(param->out.get(), &param->input->dims().data()) ||
        param->input->dims().size() > cuda_detail::kMaxCudaRank) return -1;
    const auto dims = param->input->dims().data();
    const int rank = static_cast<int>(dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= dims[static_cast<size_t>(i)];
    for (int i = axis + 1; i < rank; ++i) inner *= dims[static_cast<size_t>(i)];
    const int64_t axis_dim = dims[static_cast<size_t>(axis)];
    if (!HasValidFp8Shape(dims, false) || axis_dim <= 0 || outer <= 0 || inner <= 0) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8SoftmaxKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(outer * inner, cuda_detail::kCudaThreads)),
                              cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), outer * inner, axis_dim, inner, param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

inline bool NormalizeFp8Axes(const std::vector<int64_t>& axes, int rank, std::vector<int>* normalized) {
    if (rank < 0 || normalized == nullptr) return false;
    normalized->clear();
    if (axes.empty()) {
        normalized->resize(static_cast<size_t>(rank));
        for (int axis = 0; axis < rank; ++axis) (*normalized)[static_cast<size_t>(axis)] = axis;
        return true;
    }
    for (int64_t axis_value : axes) {
        if (axis_value < -static_cast<int64_t>(rank) || axis_value >= static_cast<int64_t>(rank)) return false;
        const int64_t axis = axis_value < 0 ? axis_value + rank : axis_value;
        normalized->push_back(static_cast<int>(axis));
    }
    std::sort(normalized->begin(), normalized->end());
    return std::adjacent_find(normalized->begin(), normalized->end()) == normalized->end();
}

template <typename Param>
inline std::vector<int64_t> Fp8ReducedShape(const std::vector<int64_t>& input_dims, const std::vector<int>& axes,
                                            bool keepdims) {
    (void)sizeof(Param);
    std::vector<int64_t> output;
    for (int axis = 0; axis < static_cast<int>(input_dims.size()); ++axis) {
        if (std::binary_search(axes.begin(), axes.end(), axis)) {
            if (keepdims) output.push_back(1);
        } else {
            output.push_back(input_dims[static_cast<size_t>(axis)]);
        }
    }
    if (output.empty()) output.push_back(1);
    return output;
}

template <DataType dtype, bool kMean>
int32_t RunFp8Reduce(typename std::conditional<kMean, operators::ReduceMeanParam, operators::ReduceSumParam>::type* param,
                     const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        param->input->dims().size() > cuda_detail::kMaxCudaRank ||
        !HasValidFp8Shape(param->input->dims().data())) return -1;
    std::vector<int> axes;
    if (!NormalizeFp8Axes(param->axes, static_cast<int>(param->input->dims().size()), &axes)) return -1;
    const auto output_dims = Fp8ReducedShape<typename std::conditional<kMean, operators::ReduceMeanParam, operators::ReduceSumParam>::type>(param->input->dims().data(), axes, param->keepdims);
    if (!IsValidFp8Output<dtype>(param->out.get(), &output_dims)) return -1;
    Fp8ReduceSpec spec{};
    if (!cuda_detail::MakeCudaShape(param->input->dims().data(), &spec.input_shape) || !cuda_detail::MakeCudaShape(output_dims, &spec.output_shape)) return -1;
    spec.reduced_rank = static_cast<int>(axes.size());
    spec.keepdims = param->keepdims ? 1 : 0;
    spec.reduce_count = 1;
    for (size_t i = 0; i < axes.size(); ++i) {
        spec.reduced_axes[i] = axes[i];
        spec.reduce_count *= spec.input_shape.dims[axes[i]];
    }
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8ReduceKernelCuda<T, kMean><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                    cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), spec, param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype, typename IndexT>
int32_t RunFp8GatherWithIndex(operators::GatherParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->data.get()) ||
        param->indices == nullptr || !param->indices->IsInitialized() ||
        param->indices->data_type() != DataTypeTrait<IndexT>::type() || param->indices->numel() < 0) return -1;
    const auto index_count = static_cast<uint64_t>(param->indices->numel());
    if (index_count > std::numeric_limits<size_t>::max() / sizeof(IndexT) ||
        param->indices->memory_size() < static_cast<size_t>(index_count) * sizeof(IndexT)) return -1;
    const int rank = static_cast<int>(param->data->dims().size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank || rank > cuda_detail::kMaxCudaRank) return -1;
    const auto& data_dims = param->data->dims().data();
    const auto& indices_dims = param->indices->dims().data();
    if (!HasValidFp8Shape(data_dims) || !HasValidFp8Shape(indices_dims)) return -1;
    std::vector<int64_t> expected_output_dims;
    expected_output_dims.reserve(data_dims.size() - 1 + indices_dims.size());
    expected_output_dims.insert(expected_output_dims.end(), data_dims.begin(), data_dims.begin() + axis);
    expected_output_dims.insert(expected_output_dims.end(), indices_dims.begin(), indices_dims.end());
    expected_output_dims.insert(expected_output_dims.end(), data_dims.begin() + axis + 1, data_dims.end());
    if (param->out == nullptr || !IsValidFp8Output<dtype>(param->out.get(), &expected_output_dims)) return -1;
    cuda_detail::CudaShape data_shape{}, indices_shape{}, output_shape{};
    if (!cuda_detail::MakeCudaShape(param->data->dims().data(), &data_shape) || !cuda_detail::MakeCudaShape(param->indices->dims().data(), &indices_shape) ||
        !cuda_detail::MakeCudaShape(param->out->dims().data(), &output_shape)) return -1;
    const size_t index_bytes = static_cast<size_t>(index_count) * sizeof(IndexT);
    if (cuda_detail::SyncTensorToHostIfNeeded(param->indices.get(), index_bytes, param->indices->raw_data()) != 0) return -1;
    const int64_t axis_dim = data_dims[static_cast<size_t>(axis)];
    for (uint64_t i = 0; i < index_count; ++i) {
        int64_t index = static_cast<int64_t>(param->indices->data<IndexT>()[i]);
        if (index < 0) index += axis_dim;
        if (index < 0 || index >= axis_dim) return -1;
    }
    cuda_detail::DeviceBuffer<T> data, output;
    cuda_detail::DeviceBuffer<IndexT> indices;
    if (cuda_detail::CopyTensorToDevice(param->data.get(), &data) != 0 || cuda_detail::CopyTensorToDevice(param->indices.get(), &indices) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8GatherKernelCuda<T, IndexT><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                     cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        data.get(), indices.get(), output.get(), param->out->numel(), data_shape, indices_shape, output_shape,
        axis, param->data->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Gather(operators::GatherParam* param, const char* timer_name) {
    if (param == nullptr || param->indices == nullptr) return -1;
    if (param->indices->data_type() == DataType::INT32) return RunFp8GatherWithIndex<dtype, int32_t>(param, timer_name);
    if (param->indices->data_type() == DataType::INT64) return RunFp8GatherWithIndex<dtype, int64_t>(param, timer_name);
    return -1;
}

template <DataType dtype>
int32_t RunFp8Expand(operators::ExpandParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->shape == nullptr ||
        param->out == nullptr || (param->shape->data_type() != DataType::INT32 && param->shape->data_type() != DataType::INT64) ||
        !param->shape->IsInitialized() || param->shape->numel() < 0) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (output_dims.size() < input_dims.size() || output_dims.size() > cuda_detail::kMaxCudaRank ||
        !HasValidFp8Shape(input_dims) || !HasValidFp8Shape(output_dims) ||
        param->shape->numel() != static_cast<int64_t>(output_dims.size())) return -1;
    const size_t shape_bytes = static_cast<size_t>(param->shape->numel()) * DataTypeBytes(param->shape->data_type());
    if (shape_bytes > param->shape->memory_size()) return -1;
    if (cuda_detail::SyncTensorToHostIfNeeded(param->shape.get(), shape_bytes, param->shape->raw_data()) != 0) return -1;
    for (int64_t i = 0; i < param->shape->numel(); ++i) {
        const int64_t target = param->shape->data_type() == DataType::INT64 ? param->shape->data<int64_t>()[i] : param->shape->data<int32_t>()[i];
        if (target != output_dims[static_cast<size_t>(i)]) return -1;
        const size_t gap = output_dims.size() - input_dims.size();
        const int64_t input_dim = i < static_cast<int64_t>(gap) ? 1 : input_dims[static_cast<size_t>(i) - gap];
        if (target <= 0 || (input_dim != target && input_dim != 1)) return -1;
    }
    if (!IsValidFp8Output<dtype>(param->out.get(), &output_dims)) return -1;
    cuda_detail::CudaShape input_shape{}, output_shape{};
    if (!cuda_detail::MakeCudaShape(input_dims, &input_shape) || !cuda_detail::MakeCudaShape(output_dims, &output_shape)) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8ExpandKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                             cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), output_shape, input_shape,
        param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Where(operators::WhereParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->condition == nullptr || !IsValidFp8Tensor<dtype>(param->x.get()) ||
        !IsValidFp8Tensor<dtype>(param->y.get()) || param->condition->data_type() != DataType::BOOL ||
        !param->condition->IsInitialized() || param->condition->numel() < 0 ||
        static_cast<uint64_t>(param->condition->numel()) > std::numeric_limits<size_t>::max()) return -1;
    if (param->condition->memory_size() < static_cast<size_t>(param->condition->numel())) return -1;
    std::vector<int64_t> condition_x_dims, output_dims;
    if (!cuda_detail::InferBroadcastShape(param->condition->dims().data(), param->x->dims().data(), &condition_x_dims) ||
        !cuda_detail::InferBroadcastShape(condition_x_dims, param->y->dims().data(), &output_dims) ||
        output_dims.size() > cuda_detail::kMaxCudaRank || !HasValidFp8Shape(output_dims) ||
        !HasValidFp8Shape(param->condition->dims().data()) || !HasValidFp8Shape(param->x->dims().data()) ||
        !HasValidFp8Shape(param->y->dims().data()) || !IsValidFp8Output<dtype>(param->out.get(), &output_dims)) return -1;
    cuda_detail::CudaShape output_shape{}, condition_shape{}, x_shape{}, y_shape{};
    if (!cuda_detail::MakeCudaShape(output_dims, &output_shape) || !cuda_detail::MakeCudaShape(param->condition->dims().data(), &condition_shape) ||
        !cuda_detail::MakeCudaShape(param->x->dims().data(), &x_shape) || !cuda_detail::MakeCudaShape(param->y->dims().data(), &y_shape)) return -1;
    cuda_detail::DeviceBuffer<uint8_t> condition;
    cuda_detail::DeviceBuffer<T> x, y, output;
    if (cuda_detail::CopyTensorToDevice(param->condition.get(), &condition) != 0 || cuda_detail::CopyTensorToDevice(param->x.get(), &x) != 0 ||
        cuda_detail::CopyTensorToDevice(param->y.get(), &y) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8WhereKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                            cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        condition.get(), x.get(), y.get(), output.get(), param->out->numel(), output_shape, condition_shape, x_shape, y_shape,
        param->x->quantization_scale(), param->y->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Equal(operators::EqualParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->lhs.get()) || !IsValidFp8Tensor<dtype>(param->rhs.get()) ||
        param->out == nullptr || param->out->data_type() != DataType::BOOL || !param->out->IsInitialized() ||
        param->out->numel() < 0 || param->out->memory_size() < static_cast<size_t>(param->out->numel())) return -1;
    std::vector<int64_t> output_dims;
    if (!cuda_detail::InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        output_dims.size() > cuda_detail::kMaxCudaRank || !HasValidFp8Shape(output_dims) || output_dims != param->out->dims().data()) return -1;
    cuda_detail::CudaShape output_shape{}, lhs_shape{}, rhs_shape{};
    if (!cuda_detail::MakeCudaShape(output_dims, &output_shape) || !cuda_detail::MakeCudaShape(param->lhs->dims().data(), &lhs_shape) || !cuda_detail::MakeCudaShape(param->rhs->dims().data(), &rhs_shape)) return -1;
    cuda_detail::DeviceBuffer<T> lhs, rhs;
    cuda_detail::DeviceBuffer<uint8_t> output;
    if (cuda_detail::CopyTensorToDevice(param->lhs.get(), &lhs) != 0 || cuda_detail::CopyTensorToDevice(param->rhs.get(), &rhs) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8EqualKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                            cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        lhs.get(), rhs.get(), output.get(), param->out->numel(), output_shape, lhs_shape, rhs_shape,
        param->lhs->quantization_scale(), param->rhs->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0 || cuda_detail::CopyDeviceToTensor(&output, param->out.get()) != 0) return -1;
    param->out->set_data_type(DataType::BOOL);
    return 0;
}

template <DataType dtype, bool kMax>
int32_t RunFp8Pool(operators::PoolParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->out == nullptr ||
        (param->input->dims().size() != 2 && param->input->dims().size() != 4) ||
        param->out->dims().size() != param->input->dims().size() || param->kernel_h <= 0 || param->kernel_w <= 0 ||
        param->stride_h <= 0 || param->stride_w <= 0 || param->pad_h < 0 || param->pad_w < 0) return -1;
    const bool is_4d = param->input->dims().size() == 4;
    const DataLayout layout = is_4d ? NormalizeDataLayout(param->input->layout()) : DataLayout::NCHW;
    ImageShape4D input_shape{}, output_shape{};
    if (is_4d) {
        if (NormalizeDataLayout(param->out->layout()) != layout ||
            !DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
            !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape)) return -1;
    } else {
        input_shape = {1, 1, param->input->dims()[0], param->input->dims()[1]};
        output_shape = {1, 1, param->out->dims()[0], param->out->dims()[1]};
    }
    int64_t expected_h = 0;
    int64_t expected_w = 0;
    if (!TryFp8WindowOutputDimension(input_shape.h, param->kernel_h, param->stride_h, param->pad_h, 1, &expected_h) ||
        !TryFp8WindowOutputDimension(input_shape.w, param->kernel_w, param->stride_w, param->pad_w, 1, &expected_w) ||
        output_shape.n != input_shape.n || output_shape.c != input_shape.c || output_shape.h != expected_h ||
        output_shape.w != expected_w) return -1;
    const auto expected_dims = is_4d ? EncodeImageShape4D(output_shape, layout)
                                     : std::vector<int64_t>{expected_h, expected_w};
    if (!IsValidFp8Output<dtype>(param->out.get(), &expected_dims)) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8PoolKernelCuda<T, kMax><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                 cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), input_shape.n, input_shape.c, input_shape.h, input_shape.w,
        output_shape.h, output_shape.w, param->kernel_h, param->kernel_w, param->stride_h, param->stride_w,
        param->pad_h, param->pad_w, static_cast<int>(layout), param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8GlobalAveragePool(operators::GlobalAveragePoolParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->out == nullptr ||
        param->input->dims().size() != 4 || param->out->dims().size() != 4) return -1;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D input_shape{}, output_shape{};
    if (NormalizeDataLayout(param->out->layout()) != layout ||
        !DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape) ||
        output_shape.n != input_shape.n || output_shape.c != input_shape.c || output_shape.h != 1 ||
        output_shape.w != 1) return -1;
    const auto expected_dims = EncodeImageShape4D(ImageShape4D{input_shape.n, input_shape.c, 1, 1}, layout);
    if (!IsValidFp8Output<dtype>(param->out.get(), &expected_dims)) return -1;
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    const int64_t total = input_shape.n * input_shape.c;
    Fp8GlobalAveragePoolKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(total, cuda_detail::kCudaThreads)),
                                        cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), total, input_shape.c, input_shape.h, input_shape.w, static_cast<int>(layout),
        param->input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
bool ValidateFp8BatchNormParameters(operators::BatchNormParam* param, int64_t channels) {
    if (param == nullptr || channels <= 0 || DataTypeBytes(dtype) == 0 ||
        static_cast<uint64_t>(channels) > std::numeric_limits<size_t>::max() / DataTypeBytes(dtype)) {
        return false;
    }
    Tensor* tensors[] = {param->scale.get(), param->bias.get(), param->mean.get(), param->var.get()};
    const size_t bytes = static_cast<size_t>(channels) * DataTypeBytes(dtype);
    for (Tensor* tensor : tensors) {
        if (tensor == nullptr || cuda_detail::SyncTensorToHostIfNeeded(tensor, bytes, tensor->raw_data()) != 0) {
            return false;
        }
    }
    for (int64_t c = 0; c < channels; ++c) {
        const float scale = TensorIO<dtype>::Read(param->scale.get(), c);
        const float bias = TensorIO<dtype>::Read(param->bias.get(), c);
        const float mean = TensorIO<dtype>::Read(param->mean.get(), c);
        const float variance = TensorIO<dtype>::Read(param->var.get(), c);
        const float denominator = variance + param->epsilon;
        if (!std::isfinite(scale) || !std::isfinite(bias) || !std::isfinite(mean) || !std::isfinite(variance) ||
            !std::isfinite(denominator) || denominator <= 0.0f) {
            return false;
        }
    }
    return true;
}

template <DataType dtype>
int32_t RunFp8BatchNorm(operators::BatchNormParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        !IsValidFp8Tensor<dtype>(param->scale.get()) || !IsValidFp8Tensor<dtype>(param->bias.get()) ||
        !IsValidFp8Tensor<dtype>(param->mean.get()) || !IsValidFp8Tensor<dtype>(param->var.get()) ||
        param->input->dims().size() != 4 || param->out == nullptr || param->out->dims().size() != 4 ||
        !std::isfinite(param->epsilon) || param->epsilon < 0.0f) return -1;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D shape{}, output_shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), layout, &shape) || !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape) ||
        output_shape.n != shape.n || output_shape.c != shape.c || output_shape.h != shape.h || output_shape.w != shape.w ||
        param->scale->dims().data() != std::vector<int64_t>{shape.c} ||
        param->bias->dims().data() != std::vector<int64_t>{shape.c} ||
        param->mean->dims().data() != std::vector<int64_t>{shape.c} ||
        param->var->dims().data() != std::vector<int64_t>{shape.c}) return -1;
    if (NormalizeDataLayout(param->out->layout()) != layout || !HasValidFp8Shape(param->input->dims().data(), false) ||
        !IsValidFp8Output<dtype>(param->out.get(), &param->input->dims().data())) return -1;
    if (!ValidateFp8BatchNormParameters<dtype>(param, shape.c)) return -1;
    cuda_detail::DeviceBuffer<T> input, scale, bias, mean, variance, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::CopyTensorToDevice(param->scale.get(), &scale) != 0 ||
        cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0 || cuda_detail::CopyTensorToDevice(param->mean.get(), &mean) != 0 ||
        cuda_detail::CopyTensorToDevice(param->var.get(), &variance) != 0 || cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    Fp8BatchNormKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), scale.get(), bias.get(), mean.get(), variance.get(), output.get(), param->out->numel(), shape.c, shape.h, shape.w,
        static_cast<int>(layout), param->epsilon, param->input->quantization_scale(), param->scale->quantization_scale(),
        param->bias->quantization_scale(), param->mean->quantization_scale(), param->var->quantization_scale(),
        param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Conv2D(operators::Conv2dParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) ||
        !IsValidFp8Tensor<dtype>(param->w.get()) || param->out == nullptr ||
        param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4 ||
        param->stride_h <= 0 || param->stride_w <= 0 || param->pad_h < 0 || param->pad_w < 0 ||
        param->dilation_h <= 0 || param->dilation_w <= 0 || param->group <= 0) {
        return -1;
    }
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (NormalizeDataLayout(param->out->layout()) != layout) return -1;
    ImageShape4D input_shape{}, output_shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape)) return -1;
    const int64_t output_channels = param->w->dims()[0];
    const int64_t kernel_channels = param->w->dims()[1];
    const int64_t kernel_height = param->w->dims()[2];
    const int64_t kernel_width = param->w->dims()[3];
    const int64_t group = param->group;
    if (output_channels <= 0 || kernel_channels <= 0 || kernel_height <= 0 || kernel_width <= 0 ||
        group > input_shape.c || group > output_channels || input_shape.c % group != 0 ||
        output_channels % group != 0 || kernel_channels != input_shape.c / group) return -1;
    int64_t expected_height = 0;
    int64_t expected_width = 0;
    if (!TryFp8WindowOutputDimension(input_shape.h, kernel_height, param->stride_h, param->pad_h,
                                     param->dilation_h, &expected_height) ||
        !TryFp8WindowOutputDimension(input_shape.w, kernel_width, param->stride_w, param->pad_w,
                                     param->dilation_w, &expected_width) ||
        output_shape.n != input_shape.n || output_shape.c != output_channels ||
        output_shape.h != expected_height || output_shape.w != expected_width) {
        return -1;
    }
    if (param->bias != nullptr &&
        (!IsValidFp8Tensor<dtype>(param->bias.get()) || param->bias->dims().size() != 1 ||
         param->bias->numel() != output_channels)) {
        return -1;
    }
    const std::vector<int64_t> expected_dims =
        EncodeImageShape4D(ImageShape4D{input_shape.n, output_channels, expected_height, expected_width}, layout);
    if (!IsValidFp8Output<dtype>(param->out.get(), &expected_dims)) return -1;
    cuda_detail::DeviceBuffer<T> input, weight, output, bias;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 || cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) return -1;
    const T* bias_ptr = nullptr;
    if (param->bias != nullptr) {
        if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) return -1;
        bias_ptr = bias.get();
    }
    Fp8Conv2DKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                             cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), weight.get(), bias_ptr, output.get(), param->out->numel(), input_shape.c, input_shape.h, input_shape.w,
        output_channels, output_shape.h, output_shape.w, kernel_channels, kernel_height, kernel_width,
        param->stride_h, param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w, group,
        static_cast<int>(layout), param->input->quantization_scale(), param->w->quantization_scale(),
        param->bias == nullptr ? 1.0f : param->bias->quantization_scale(), param->out->quantization_scale(),
        param->bias == nullptr ? 0 : param->bias->numel());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Resize(operators::ResizeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->input.get()) || param->out == nullptr ||
        param->input->dims().size() != param->out->dims().size() ||
        param->input->dims().empty() || param->input->dims().size() > cuda_detail::kMaxCudaRank ||
        param->scales.size() != param->input->dims().size()) {
        return -1;
    }
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    std::vector<int64_t> expected_dims(input_dims.size(), 0);
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        if (!TryFp8ScaledDimension(input_dims[axis], param->scales[axis], &expected_dims[axis])) return -1;
    }
    if (!IsValidFp8Output<dtype>(param->out.get(), &expected_dims)) return -1;
    Fp8ResizeSpec spec{};
    if (!cuda_detail::MakeCudaShape(input_dims, &spec.input_shape) ||
        !cuda_detail::MakeCudaShape(output_dims, &spec.output_shape)) {
        return -1;
    }
    spec.scales[0] = 1.0f;
    for (size_t axis = 0; axis < param->scales.size(); ++axis) spec.scales[axis] = param->scales[axis];
    cuda_detail::DeviceBuffer<T> input, output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    Fp8ResizeKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                             cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), output.get(), param->out->numel(), spec, param->input->quantization_scale(),
        param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8ResizeConcat(operators::ResizeConcatParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || !IsValidFp8Tensor<dtype>(param->resize_input.get()) ||
        !IsValidFp8Tensor<dtype>(param->concat_input.get()) || param->out == nullptr ||
        param->resize_input->dims().size() != 4 ||
        param->concat_input->dims().size() != 4 || param->out->dims().size() != 4 || param->scales.size() != 4 ||
        param->resize_input_index < 0 || param->resize_input_index > 1) {
        return -1;
    }
    const DataLayout resize_layout = NormalizeDataLayout(param->resize_input->layout());
    const DataLayout concat_layout = NormalizeDataLayout(param->concat_input->layout());
    const DataLayout output_layout = NormalizeDataLayout(param->out->layout());
    if (resize_layout != concat_layout || resize_layout != output_layout) return -1;
    const int axis = param->axis < 0 ? param->axis + 4 : param->axis;
    if (axis != ChannelAxisForLayout(output_layout)) return -1;
    for (float scale : param->scales) if (!std::isfinite(scale) || scale <= 0.0f) return -1;
    if (param->scales[0] != 1.0f || param->scales[static_cast<size_t>(axis)] != 1.0f) return -1;
    ImageShape4D resize_shape{}, concat_shape{}, output_shape{};
    if (!DecodeImageShape4D(param->resize_input->dims().data(), resize_layout, &resize_shape) ||
        !DecodeImageShape4D(param->concat_input->dims().data(), concat_layout, &concat_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), output_layout, &output_shape) ||
        resize_shape.n != concat_shape.n || output_shape.n != resize_shape.n ||
        output_shape.c != resize_shape.c + concat_shape.c) {
        return -1;
    }
    const auto& resize_dims = param->resize_input->dims().data();
    const auto& concat_dims = param->concat_input->dims().data();
    const auto& output_dims = param->out->dims().data();
    std::vector<int64_t> expected_resize_dims(4);
    for (size_t i = 0; i < 4; ++i) {
        if (!TryFp8ScaledDimension(resize_dims[i], param->scales[i], &expected_resize_dims[i])) return -1;
        if (static_cast<int>(i) != axis && expected_resize_dims[i] != concat_dims[i]) return -1;
    }
    if (expected_resize_dims[static_cast<size_t>(axis)] + concat_dims[static_cast<size_t>(axis)] !=
        output_dims[static_cast<size_t>(axis)]) return -1;
    for (size_t i = 0; i < 4; ++i) {
        if (static_cast<int>(i) != axis && output_dims[i] != expected_resize_dims[i]) return -1;
    }
    const std::vector<int64_t> expected_output_dims = EncodeImageShape4D(
        ImageShape4D{resize_shape.n, resize_shape.c + concat_shape.c,
                     expected_resize_dims[static_cast<size_t>(HeightAxisForLayout(output_layout))],
                     expected_resize_dims[static_cast<size_t>(WidthAxisForLayout(output_layout))]},
        output_layout);
    if (!IsValidFp8Output<dtype>(param->out.get(), &expected_output_dims)) return -1;
    const float scale_h = param->scales[static_cast<size_t>(HeightAxisForLayout(output_layout))];
    const float scale_w = param->scales[static_cast<size_t>(WidthAxisForLayout(output_layout))];
    cuda_detail::DeviceBuffer<T> resize_input, concat_input, output;
    if (cuda_detail::CopyTensorToDevice(param->resize_input.get(), &resize_input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->concat_input.get(), &concat_input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    Fp8ResizeConcatKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                                  cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        resize_input.get(), concat_input.get(), output.get(), param->out->numel(), resize_shape.c, concat_shape.c,
        resize_shape.h, resize_shape.w, output_shape.h, output_shape.w, scale_h, scale_w,
        param->resize_input_index, static_cast<int>(output_layout), param->resize_input->quantization_scale(),
        param->concat_input->quantization_scale(), param->out->quantization_scale());
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunFp8Pow(operators::PowParam* param, const char* timer_name) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr) {
        if (!IsReadableScalarFloatTensor(param->exponent_tensor.get())) return -1;
        const size_t bytes = DataTypeBytes(param->exponent_tensor->data_type());
        if (cuda_detail::SyncTensorToHostIfNeeded(param->exponent_tensor.get(), bytes,
                                                  param->exponent_tensor->raw_data()) != 0 ||
            !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) {
            return -1;
        }
    }
    operators::UnaryParam unary{};
    unary.input = param->input;
    unary.out = param->out;
    return RunFp8Unary<dtype, 11>(&unary, timer_name, exponent);
}

template <DataType dtype>
int32_t RunFp8ReduceMean(operators::ReduceMeanParam* param, const char* timer_name) {
    return RunFp8Reduce<dtype, true>(param, timer_name);
}

template <DataType dtype>
int32_t RunFp8ReduceSum(operators::ReduceSumParam* param, const char* timer_name) {
    return RunFp8Reduce<dtype, false>(param, timer_name);
}

}  // namespace

#define DEFINE_CUDA_FP8_UNARY(Class, Op, Label)                                                        \
    template <>                                                                                         \
    int32_t Class<DeviceType::CUDA, DataType::FP8E4M3>::compute() {                                    \
        return RunFp8Unary<DataType::FP8E4M3, Op>(static_cast<operators::UnaryParam*>(param_),          \
                                                   "CUDA::" Label "::FP8E4M3");                         \
    }                                                                                                    \
    template <>                                                                                         \
    int32_t Class<DeviceType::CUDA, DataType::FP8E5M2>::compute() {                                    \
        return RunFp8Unary<DataType::FP8E5M2, Op>(static_cast<operators::UnaryParam*>(param_),          \
                                                   "CUDA::" Label "::FP8E5M2");                         \
    }

DEFINE_CUDA_FP8_UNARY(ReluKernel, 0, "ReLU")
DEFINE_CUDA_FP8_UNARY(SigmoidKernel, 1, "Sigmoid")
DEFINE_CUDA_FP8_UNARY(SiluKernel, 2, "SiLU")
DEFINE_CUDA_FP8_UNARY(TanhKernel, 3, "Tanh")
DEFINE_CUDA_FP8_UNARY(ExpKernel, 4, "Exp")
DEFINE_CUDA_FP8_UNARY(SinKernel, 5, "Sin")
DEFINE_CUDA_FP8_UNARY(CosKernel, 6, "Cos")
DEFINE_CUDA_FP8_UNARY(ErfKernel, 7, "Erf")
DEFINE_CUDA_FP8_UNARY(SqrtKernel, 8, "Sqrt")
DEFINE_CUDA_FP8_UNARY(SoftplusKernel, 9, "Softplus")
DEFINE_CUDA_FP8_UNARY(NegKernel, 10, "Neg")

#define DEFINE_CUDA_FP8_PARAM(Class, ParamType, Function, Label)                                      \
    template <>                                                                                         \
    int32_t Class<DeviceType::CUDA, DataType::FP8E4M3>::compute() {                                    \
        return Function<DataType::FP8E4M3>(static_cast<ParamType*>(param_), "CUDA::" Label "::FP8E4M3"); \
    }                                                                                                    \
    template <>                                                                                         \
    int32_t Class<DeviceType::CUDA, DataType::FP8E5M2>::compute() {                                    \
        return Function<DataType::FP8E5M2>(static_cast<ParamType*>(param_), "CUDA::" Label "::FP8E5M2"); \
    }

DEFINE_CUDA_FP8_PARAM(BatchNormalizationKernel, operators::BatchNormParam, RunFp8BatchNorm, "BatchNormalization")
DEFINE_CUDA_FP8_PARAM(ConcatKernel, operators::ConcatParam, RunFp8Concat, "Concat")
DEFINE_CUDA_FP8_PARAM(Conv2DKernel, operators::Conv2dParam, RunFp8Conv2D, "Conv2D")
DEFINE_CUDA_FP8_PARAM(EqualKernel, operators::EqualParam, RunFp8Equal, "Equal")
DEFINE_CUDA_FP8_PARAM(ExpandKernel, operators::ExpandParam, RunFp8Expand, "Expand")
DEFINE_CUDA_FP8_PARAM(GatherKernel, operators::GatherParam, RunFp8Gather, "Gather")
DEFINE_CUDA_FP8_PARAM(ReduceMeanKernel, operators::ReduceMeanParam, RunFp8ReduceMean, "ReduceMean")
DEFINE_CUDA_FP8_PARAM(ReduceSumKernel, operators::ReduceSumParam, RunFp8ReduceSum, "ReduceSum")
DEFINE_CUDA_FP8_PARAM(ResizeKernel, operators::ResizeParam, RunFp8Resize, "Resize")
DEFINE_CUDA_FP8_PARAM(ResizeConcatKernel, operators::ResizeConcatParam, RunFp8ResizeConcat, "ResizeConcat")
DEFINE_CUDA_FP8_PARAM(SliceKernel, operators::SliceParam, RunFp8Slice, "Slice")
DEFINE_CUDA_FP8_PARAM(SoftmaxKernel, operators::SoftmaxParam, RunFp8Softmax, "Softmax")
DEFINE_CUDA_FP8_PARAM(SplitKernel, operators::SplitParam, RunFp8Split, "Split")
DEFINE_CUDA_FP8_PARAM(TransposeKernel, operators::TransposeParam, RunFp8Transpose, "Transpose")
DEFINE_CUDA_FP8_PARAM(WhereKernel, operators::WhereParam, RunFp8Where, "Where")

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Copy<DataType::FP8E4M3, operators::UnaryParam>(static_cast<operators::UnaryParam*>(param_),
                                                                  "CUDA::Identity::FP8E4M3");
}
template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Copy<DataType::FP8E5M2, operators::UnaryParam>(static_cast<operators::UnaryParam*>(param_),
                                                                  "CUDA::Identity::FP8E5M2");
}
template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Copy<DataType::FP8E4M3, operators::ReshapeParam>(static_cast<operators::ReshapeParam*>(param_),
                                                                    "CUDA::Reshape::FP8E4M3");
}
template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Copy<DataType::FP8E5M2, operators::ReshapeParam>(static_cast<operators::ReshapeParam*>(param_),
                                                                    "CUDA::Reshape::FP8E5M2");
}
template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Copy<DataType::FP8E4M3, operators::FlattenParam>(static_cast<operators::FlattenParam*>(param_),
                                                                    "CUDA::Flatten::FP8E4M3");
}
template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Copy<DataType::FP8E5M2, operators::FlattenParam>(static_cast<operators::FlattenParam*>(param_),
                                                                    "CUDA::Flatten::FP8E5M2");
}
template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Copy<DataType::FP8E4M3, operators::AxesParam>(static_cast<operators::AxesParam*>(param_),
                                                                 "CUDA::Unsqueeze::FP8E4M3");
}
template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Copy<DataType::FP8E5M2, operators::AxesParam>(static_cast<operators::AxesParam*>(param_),
                                                                 "CUDA::Unsqueeze::FP8E5M2");
}
template <>
int32_t SqueezeKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Copy<DataType::FP8E4M3, operators::AxesParam>(static_cast<operators::AxesParam*>(param_),
                                                                 "CUDA::Squeeze::FP8E4M3");
}
template <>
int32_t SqueezeKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Copy<DataType::FP8E5M2, operators::AxesParam>(static_cast<operators::AxesParam*>(param_),
                                                                 "CUDA::Squeeze::FP8E5M2");
}
template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Pow<DataType::FP8E4M3>(static_cast<operators::PowParam*>(param_), "CUDA::Pow::FP8E4M3");
}
template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Pow<DataType::FP8E5M2>(static_cast<operators::PowParam*>(param_), "CUDA::Pow::FP8E5M2");
}

template <>
int32_t AvgPoolKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Pool<DataType::FP8E4M3, false>(static_cast<operators::PoolParam*>(param_), "CUDA::AvgPool::FP8E4M3");
}
template <>
int32_t AvgPoolKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Pool<DataType::FP8E5M2, false>(static_cast<operators::PoolParam*>(param_), "CUDA::AvgPool::FP8E5M2");
}
template <>
int32_t MaxPoolKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8Pool<DataType::FP8E4M3, true>(static_cast<operators::PoolParam*>(param_), "CUDA::MaxPool::FP8E4M3");
}
template <>
int32_t MaxPoolKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8Pool<DataType::FP8E5M2, true>(static_cast<operators::PoolParam*>(param_), "CUDA::MaxPool::FP8E5M2");
}

template <>
int32_t GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP8E4M3>::compute() {
    return RunFp8GlobalAveragePool<DataType::FP8E4M3>(static_cast<operators::GlobalAveragePoolParam*>(param_),
                                                       "CUDA::GlobalAveragePool::FP8E4M3");
}
template <>
int32_t GlobalAveragePoolKernel<DeviceType::CUDA, DataType::FP8E5M2>::compute() {
    return RunFp8GlobalAveragePool<DataType::FP8E5M2>(static_cast<operators::GlobalAveragePoolParam*>(param_),
                                                       "CUDA::GlobalAveragePool::FP8E5M2");
}

#undef DEFINE_CUDA_FP8_PARAM
#undef DEFINE_CUDA_FP8_UNARY

void EnsureCudaFp8KernelsRegistered() { (void)g_cuda_fp8_kernels_registered; }

}  // namespace kernel
}  // namespace feather
