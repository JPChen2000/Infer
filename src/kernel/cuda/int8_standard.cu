#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/kernel.h"
#include "quant/quantization.h"
#include "src/kernel/add.h"
#include "src/kernel/batch_normalization.h"
#include "src/kernel/cast.h"
#include "src/kernel/common/tensor_op_utils.h"
#include "src/kernel/concat.h"
#include "src/kernel/constant_of_shape.h"
#include "src/kernel/cos.h"
#include "src/kernel/div.h"
#include "src/kernel/equal.h"
#include "src/kernel/erf.h"
#include "src/kernel/expand.h"
#include "src/kernel/exp.h"
#include "src/kernel/flatten.h"
#include "src/kernel/gather.h"
#include "src/kernel/global_average_pool.h"
#include "src/kernel/identity.h"
#include "src/kernel/mul.h"
#include "src/kernel/neg.h"
#include "src/kernel/pow.h"
#include "src/kernel/pool.h"
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
#include "util/bf16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

constexpr int kMaxRank = 8;
constexpr int kCudaThreads = 256;

struct Int8View {
    float scale{1.0f};
    int32_t zero_point{0};
};

struct CudaShape {
    int rank{0};
    int64_t dims[kMaxRank]{};
    int64_t strides[kMaxRank]{};
};

class DeviceBytes {
   public:
    DeviceBytes() = default;
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    ~DeviceBytes() { reset(); }

    bool allocate(size_t bytes) {
        reset();
        if (bytes == 0) return false;
        return cudaMalloc(reinterpret_cast<void**>(&data_), bytes) == cudaSuccess;
    }

    void reset() {
        if (data_ != nullptr) cudaFree(data_);
        data_ = nullptr;
    }

    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(data_);
    }

   private:
    uint8_t* data_{nullptr};
};

inline int Blocks(int64_t count) {
    return static_cast<int>((count + kCudaThreads - 1) / kCudaThreads);
}

inline bool SameDims(const Tensor* tensor, const std::vector<int64_t>& dims) {
    return tensor != nullptr && tensor->dims().data() == dims;
}

inline bool ReadyStorage(const Tensor* tensor, DataType type) {
    return tensor != nullptr && tensor->data_type() == type && tensor->IsInitialized() && tensor->numel() > 0 &&
           tensor->memory_size() >= static_cast<size_t>(tensor->numel()) * DataTypeBytes(type);
}

inline bool BuildInt8View(const Tensor* tensor, Int8View* view) {
    if (view == nullptr || !ReadyStorage(tensor, DataType::INT8)) return false;
    const auto& params = tensor->quantization();
    if (!params.enabled || params.granularity != QuantizationGranularity::kPerTensor ||
        !ValidateQuantizationParams(params, tensor->dims().data())) return false;
    view->scale = params.scale_at(0);
    view->zero_point = params.zero_point_at(0);
    return std::isfinite(view->scale) && view->scale > 0.0f && view->zero_point >= -128 && view->zero_point <= 127;
}

inline bool BuildInt8OutputView(const Tensor* tensor, const std::vector<int64_t>& dims, Int8View* view) {
    return SameDims(tensor, dims) && BuildInt8View(tensor, view);
}

inline bool CopyToDevice(const Tensor* tensor, DeviceBytes* device) {
    if (tensor == nullptr || device == nullptr || !tensor->IsInitialized() || tensor->numel() <= 0) return false;
    const size_t bytes = static_cast<size_t>(tensor->numel()) * DataTypeBytes(tensor->data_type());
    return device->allocate(bytes) && cudaMemcpy(device->as<uint8_t>(), tensor->raw_data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

inline bool CopyFromDevice(const DeviceBytes& device, Tensor* tensor) {
    if (tensor == nullptr || tensor->numel() <= 0) return false;
    const size_t bytes = static_cast<size_t>(tensor->numel()) * DataTypeBytes(tensor->data_type());
    return tensor->memory_size() >= bytes && cudaMemcpy(tensor->raw_data(), device.as<uint8_t>(), bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

template <typename T>
inline bool UploadVector(const std::vector<T>& values, DeviceBytes* device) {
    if (device == nullptr || values.empty() || !device->allocate(values.size() * sizeof(T))) return false;
    return cudaMemcpy(device->as<uint8_t>(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

inline bool LaunchSucceeded() {
    return cudaGetLastError() == cudaSuccess;
}

inline bool MakeShape(const std::vector<int64_t>& dims, CudaShape* shape) {
    if (shape == nullptr || dims.size() > kMaxRank) return false;
    *shape = CudaShape();
    shape->rank = static_cast<int>(dims.size());
    int64_t suffix = 1;
    for (int axis = shape->rank - 1; axis >= 0; --axis) {
        if (dims[static_cast<size_t>(axis)] <= 0 ||
            suffix > std::numeric_limits<int64_t>::max() / dims[static_cast<size_t>(axis)]) return false;
        shape->dims[axis] = dims[static_cast<size_t>(axis)];
        shape->strides[axis] = suffix;
        suffix *= dims[static_cast<size_t>(axis)];
    }
    return true;
}

inline int64_t Product(const std::vector<int64_t>& dims, size_t begin, size_t end) {
    int64_t result = 1;
    for (size_t i = begin; i < end; ++i) {
        if (dims[i] <= 0 || result > std::numeric_limits<int64_t>::max() / dims[i]) return -1;
        result *= dims[i];
    }
    return result;
}

inline bool InferBroadcastShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                                std::vector<int64_t>* output) {
    if (output == nullptr) return false;
    const size_t rank = std::max(lhs.size(), rhs.size());
    output->assign(rank, 1);
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t lhs_dim = axis < rank - lhs.size() ? 1 : lhs[axis - (rank - lhs.size())];
        const int64_t rhs_dim = axis < rank - rhs.size() ? 1 : rhs[axis - (rank - rhs.size())];
        if (lhs_dim <= 0 || rhs_dim <= 0 || (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1)) return false;
        (*output)[axis] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline bool ReadHostInt64(const Tensor* tensor, std::vector<int64_t>* values) {
    if (tensor == nullptr || values == nullptr || !tensor->IsInitialized() || tensor->numel() <= 0) return false;
    values->resize(static_cast<size_t>(tensor->numel()));
    switch (tensor->data_type()) {
        case DataType::INT64:
            for (int64_t i = 0; i < tensor->numel(); ++i) (*values)[static_cast<size_t>(i)] = tensor->data<int64_t>()[i];
            return true;
        case DataType::INT32:
            for (int64_t i = 0; i < tensor->numel(); ++i) (*values)[static_cast<size_t>(i)] = tensor->data<int32_t>()[i];
            return true;
        case DataType::INT8:
            for (int64_t i = 0; i < tensor->numel(); ++i) (*values)[static_cast<size_t>(i)] = tensor->data<int8_t>()[i];
            return true;
        case DataType::UINT8:
        case DataType::BOOL:
            for (int64_t i = 0; i < tensor->numel(); ++i) (*values)[static_cast<size_t>(i)] = tensor->data<uint8_t>()[i];
            return true;
        default:
            return false;
    }
}

inline bool ReadHostFloat(const Tensor* tensor, int64_t index, float* value) {
    if (tensor == nullptr || value == nullptr || index < 0 || index >= tensor->numel() || !tensor->IsInitialized()) return false;
    switch (tensor->data_type()) {
        case DataType::FP32:
        case DataType::FP16:
        case DataType::BF16:
            *value = common_tensor_detail::ReadFloat(tensor, index);
            return std::isfinite(*value);
        case DataType::INT8: {
            Int8View view;
            if (!BuildInt8View(tensor, &view)) return false;
            *value = (static_cast<int32_t>(tensor->data<int8_t>()[index]) - view.zero_point) * view.scale;
            return true;
        }
        case DataType::INT32:
            *value = static_cast<float>(tensor->data<int32_t>()[index]);
            return true;
        case DataType::INT64:
            *value = static_cast<float>(tensor->data<int64_t>()[index]);
            return true;
        default:
            return false;
    }
}

inline bool ReadHostScalar(const Tensor* tensor, float* value) {
    return tensor != nullptr && tensor->numel() == 1 && ReadHostFloat(tensor, 0, value);
}

inline bool NormalizeAxes(const std::vector<int64_t>& axes, int rank, std::vector<int64_t>* normalized) {
    if (normalized == nullptr || rank < 0) return false;
    normalized->clear();
    if (axes.empty()) {
        normalized->resize(static_cast<size_t>(rank));
        std::iota(normalized->begin(), normalized->end(), 0);
        return true;
    }
    for (int64_t axis : axes) {
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) return false;
        normalized->push_back(axis);
    }
    std::sort(normalized->begin(), normalized->end());
    return std::adjacent_find(normalized->begin(), normalized->end()) == normalized->end();
}

inline bool ReadImageShape(const Tensor* tensor, int64_t* n, int64_t* c, int64_t* h, int64_t* w,
                           bool* channel_last) {
    if (tensor == nullptr || n == nullptr || c == nullptr || h == nullptr || w == nullptr || channel_last == nullptr) return false;
    if (tensor->dims().size() == 2) {
        *n = 1;
        *c = 1;
        *h = tensor->dims()[0];
        *w = tensor->dims()[1];
        *channel_last = false;
        return *h > 0 && *w > 0;
    }
    if (tensor->dims().size() != 4) return false;
    ImageShape4D shape{};
    if (!DecodeImageShape4D(tensor->dims().data(), NormalizeDataLayout(tensor->layout()), &shape)) return false;
    *n = shape.n;
    *c = shape.c;
    *h = shape.h;
    *w = shape.w;
    *channel_last = IsChannelLastLayout(tensor->layout());
    return true;
}

__device__ inline float ReadQ(const int8_t* input, int64_t index, Int8View view) {
    return (static_cast<int32_t>(input[index]) - view.zero_point) * view.scale;
}

__device__ inline int8_t WriteQ(float value, Int8View view) {
    if (isnan(value)) return static_cast<int8_t>(view.zero_point);
    const float q = nearbyintf(value / view.scale) + static_cast<float>(view.zero_point);
    return static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

__device__ inline void DecodeIndex(int64_t index, const CudaShape& shape, int64_t* coordinates) {
    for (int axis = 0; axis < shape.rank; ++axis) {
        coordinates[axis] = index / shape.strides[axis];
        index %= shape.strides[axis];
    }
}

__device__ inline int64_t EncodeIndex(const int64_t* coordinates, const CudaShape& shape) {
    int64_t result = 0;
    for (int axis = 0; axis < shape.rank; ++axis) result += coordinates[axis] * shape.strides[axis];
    return result;
}

__device__ inline int64_t BroadcastIndex(int64_t output_index, const CudaShape& output, const CudaShape& input) {
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(output_index, output, output_coordinates);
    const int gap = output.rank - input.rank;
    for (int axis = 0; axis < input.rank; ++axis) {
        input_coordinates[axis] = input.dims[axis] == 1 ? 0 : output_coordinates[axis + gap];
    }
    return EncodeIndex(input_coordinates, input);
}

__device__ inline float UnaryValue(float value, int op) {
    if (op == 0) return fmaxf(value, 0.0f);
    if (op == 1) return -value;
    if (op == 2) return 1.0f / (1.0f + expf(-value));
    if (op == 3) return value / (1.0f + expf(-value));
    if (op == 4) return expf(value);
    if (op == 5) return sqrtf(value);
    if (op == 6) return tanhf(value);
    if (op == 7) return erff(value);
    if (op == 8) return sinf(value);
    if (op == 9) return cosf(value);
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

__global__ void UnaryQKernel(const int8_t* input, int8_t* output, int64_t count, Int8View input_view,
                             Int8View output_view, int op) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(UnaryValue(ReadQ(input, index, input_view), op), output_view);
}

__global__ void BinaryQKernel(const int8_t* lhs, const int8_t* rhs, int8_t* output, int64_t count,
                             CudaShape output_shape, CudaShape lhs_shape, CudaShape rhs_shape,
                             Int8View lhs_view, Int8View rhs_view, Int8View output_view, int op) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float a = ReadQ(lhs, BroadcastIndex(index, output_shape, lhs_shape), lhs_view);
    const float b = ReadQ(rhs, BroadcastIndex(index, output_shape, rhs_shape), rhs_view);
    const float value = op == 0 ? a + b : op == 1 ? a - b : op == 2 ? a * b : a / b;
    output[index] = WriteQ(value, output_view);
}

__global__ void CopyQKernel(const int8_t* input, int8_t* output, int64_t count, Int8View input_view,
                            Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(ReadQ(input, index, input_view), output_view);
}

__global__ void PowQKernel(const int8_t* input, int8_t* output, int64_t count, Int8View input_view,
                           Int8View output_view, float exponent) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(powf(ReadQ(input, index, input_view), exponent), output_view);
}

__global__ void TransposeQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                                 CudaShape output_shape, const int* perm, Int8View input_view,
                                 Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(index, output_shape, output_coordinates);
    for (int axis = 0; axis < output_shape.rank; ++axis) input_coordinates[perm[axis]] = output_coordinates[axis];
    output[index] = WriteQ(ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view), output_view);
}

__global__ void SliceQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                             CudaShape output_shape, const int64_t* starts, const int64_t* steps,
                             Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(index, output_shape, output_coordinates);
    for (int axis = 0; axis < input_shape.rank; ++axis) input_coordinates[axis] = starts[axis] + output_coordinates[axis] * steps[axis];
    output[index] = WriteQ(ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view), output_view);
}

__global__ void ExpandQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                              CudaShape output_shape, Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(ReadQ(input, BroadcastIndex(index, output_shape, input_shape), input_view), output_view);
}

__global__ void WhereQKernel(const uint8_t* condition, const int8_t* x, const int8_t* y, int8_t* output, int64_t count,
                             CudaShape output_shape, CudaShape condition_shape, CudaShape x_shape, CudaShape y_shape,
                             Int8View x_view, Int8View y_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const bool selected = condition[BroadcastIndex(index, output_shape, condition_shape)] != 0;
    const int64_t source = selected ? BroadcastIndex(index, output_shape, x_shape) : BroadcastIndex(index, output_shape, y_shape);
    output[index] = WriteQ(ReadQ(selected ? x : y, source, selected ? x_view : y_view), output_view);
}

__global__ void GatherQKernel(const int8_t* input, const int64_t* indices, int8_t* output, int64_t count,
                              CudaShape input_shape, CudaShape indices_shape, CudaShape output_shape, int axis,
                              Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    int64_t indices_coordinates[kMaxRank]{};
    DecodeIndex(index, output_shape, output_coordinates);
    for (int a = 0; a < axis; ++a) input_coordinates[a] = output_coordinates[a];
    for (int a = 0; a < indices_shape.rank; ++a) indices_coordinates[a] = output_coordinates[axis + a];
    int64_t gathered = indices[EncodeIndex(indices_coordinates, indices_shape)];
    if (gathered < 0) gathered += input_shape.dims[axis];
    input_coordinates[axis] = gathered;
    for (int a = axis + 1; a < input_shape.rank; ++a) input_coordinates[a] = output_coordinates[axis + indices_shape.rank + a - axis - 1];
    output[index] = WriteQ(ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view), output_view);
}

__global__ void ResizeQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                              CudaShape output_shape, Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(index, output_shape, output_coordinates);
    for (int axis = 0; axis < output_shape.rank; ++axis) {
        const int64_t source = output_coordinates[axis] * input_shape.dims[axis] / output_shape.dims[axis];
        input_coordinates[axis] = source < input_shape.dims[axis] ? source : input_shape.dims[axis] - 1;
    }
    output[index] = WriteQ(ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view), output_view);
}

__global__ void ConcatQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                              CudaShape output_shape, int axis, int64_t axis_offset, Int8View input_view,
                              Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t input_coordinates[kMaxRank]{};
    int64_t output_coordinates[kMaxRank]{};
    DecodeIndex(index, input_shape, input_coordinates);
    for (int axis_index = 0; axis_index < input_shape.rank; ++axis_index) output_coordinates[axis_index] = input_coordinates[axis_index];
    output_coordinates[axis] += axis_offset;
    output[EncodeIndex(output_coordinates, output_shape)] = WriteQ(ReadQ(input, index, input_view), output_view);
}

__global__ void SplitQKernel(const int8_t* input, int8_t* output, int64_t count, CudaShape input_shape,
                             CudaShape output_shape, int axis, int64_t axis_offset, Int8View input_view,
                             Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(index, output_shape, output_coordinates);
    for (int axis_index = 0; axis_index < output_shape.rank; ++axis_index) input_coordinates[axis_index] = output_coordinates[axis_index];
    input_coordinates[axis] += axis_offset;
    output[index] = WriteQ(ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view), output_view);
}

__global__ void ReduceQKernel(const int8_t* input, int8_t* output, int64_t output_count, CudaShape input_shape,
                              CudaShape output_shape, int axis_mask, bool mean, int64_t reduce_count,
                              Int8View input_view, Int8View output_view) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) return;
    int64_t output_coordinates[kMaxRank]{};
    int64_t input_coordinates[kMaxRank]{};
    DecodeIndex(output_index, output_shape, output_coordinates);
    int output_axis = 0;
    for (int axis = 0; axis < input_shape.rank; ++axis) {
        if ((axis_mask & (1 << axis)) == 0) input_coordinates[axis] = output_coordinates[output_axis++];
        else input_coordinates[axis] = 0;
    }
    float sum = 0.0f;
    for (int64_t reduced = 0; reduced < reduce_count; ++reduced) {
        int64_t remainder = reduced;
        for (int axis = input_shape.rank - 1; axis >= 0; --axis) {
            if ((axis_mask & (1 << axis)) != 0) {
                input_coordinates[axis] = remainder % input_shape.dims[axis];
                remainder /= input_shape.dims[axis];
            }
        }
        sum += ReadQ(input, EncodeIndex(input_coordinates, input_shape), input_view);
    }
    output[output_index] = WriteQ(mean ? sum / static_cast<float>(reduce_count) : sum, output_view);
}

__global__ void SoftmaxQKernel(const int8_t* input, int8_t* output, int64_t rows, int64_t axis_dim, int64_t inner,
                               Int8View input_view, Int8View output_view) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int64_t outer_index = row / inner;
    const int64_t inner_index = row % inner;
    const int64_t base = outer_index * axis_dim * inner + inner_index;
    float maximum = -3.402823466e+38F;
    for (int64_t axis = 0; axis < axis_dim; ++axis) maximum = fmaxf(maximum, ReadQ(input, base + axis * inner, input_view));
    float sum = 0.0f;
    for (int64_t axis = 0; axis < axis_dim; ++axis) sum += expf(ReadQ(input, base + axis * inner, input_view) - maximum);
    for (int64_t axis = 0; axis < axis_dim; ++axis) output[base + axis * inner] = WriteQ(expf(ReadQ(input, base + axis * inner, input_view) - maximum) / sum, output_view);
}

__device__ inline int64_t ImageOffset(int64_t n, int64_t c, int64_t h, int64_t w, int64_t channels,
                                      int64_t height, int64_t width, bool channel_last) {
    return channel_last ? ((n * height + h) * width + w) * channels + c : ((n * channels + c) * height + h) * width + w;
}

__global__ void PoolQKernel(const int8_t* input, int8_t* output, int64_t output_count, int64_t channels,
                            int64_t input_height, int64_t input_width, int64_t output_height, int64_t output_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                            int64_t batches, bool channel_last, bool maximum, Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    int64_t n = 0, c = 0, oh = 0, ow = 0;
    if (channel_last) {
        ow = index % output_width;
        const int64_t t0 = index / output_width;
        c = t0 % channels;
        const int64_t t1 = t0 / channels;
        oh = t1 % output_height;
        n = t1 / output_height;
    } else {
        ow = index % output_width;
        const int64_t t0 = index / output_width;
        oh = t0 % output_height;
        const int64_t t1 = t0 / output_height;
        c = t1 % channels;
        n = t1 / channels;
    }
    if (n >= batches) return;
    float result = maximum ? -3.402823466e+38F : 0.0f;
    int64_t count = 0;
    for (int kh = 0; kh < kernel_h; ++kh) for (int kw = 0; kw < kernel_w; ++kw) {
        const int64_t ih = oh * stride_h + kh - pad_h;
        const int64_t iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= input_height || iw < 0 || iw >= input_width) continue;
        const float value = ReadQ(input, ImageOffset(n, c, ih, iw, channels, input_height, input_width, channel_last), input_view);
        if (maximum) result = fmaxf(result, value);
        else { result += value; ++count; }
    }
    if (!maximum && count != 0) result /= static_cast<float>(count);
    output[index] = WriteQ(result, output_view);
}

__global__ void GlobalAveragePoolQKernel(const int8_t* input, int8_t* output, int64_t count, int64_t channels,
                                         int64_t height, int64_t width, int64_t batches, bool channel_last,
                                         Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t n = index / channels;
    const int64_t c = index % channels;
    if (n >= batches) return;
    float sum = 0.0f;
    for (int64_t h = 0; h < height; ++h) for (int64_t w = 0; w < width; ++w)
        sum += ReadQ(input, ImageOffset(n, c, h, w, channels, height, width, channel_last), input_view);
    output[index] = WriteQ(sum / static_cast<float>(height * width), output_view);
}

__global__ void BatchNormQKernel(const int8_t* input, const float* scale, const float* bias, const float* mean,
                                 const float* variance, int8_t* output, int64_t count, int64_t channels,
                                 int64_t height, int64_t width, int64_t batches, bool channel_last, float epsilon,
                                 Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t n = 0, c = 0;
    if (channel_last) {
        const int64_t t0 = index / width;
        c = t0 % channels;
        const int64_t t1 = t0 / channels;
        n = t1 / height;
    } else {
        const int64_t t0 = index / width;
        const int64_t t1 = t0 / height;
        c = t1 % channels;
        n = t1 / channels;
    }
    if (n >= batches || c >= channels) return;
    const float normalized = (ReadQ(input, index, input_view) - mean[c]) / sqrtf(variance[c] + epsilon);
    output[index] = WriteQ(normalized * scale[c] + bias[c], output_view);
}

__global__ void FillQKernel(int8_t* output, int64_t count, float value, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(value, output_view);
}

__global__ void EqualQKernel(const int8_t* lhs, const int8_t* rhs, uint8_t* output, int64_t count,
                             CudaShape output_shape, CudaShape lhs_shape, CudaShape rhs_shape,
                             Int8View lhs_view, Int8View rhs_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float lhs_value = ReadQ(lhs, BroadcastIndex(index, output_shape, lhs_shape), lhs_view);
        const float rhs_value = ReadQ(rhs, BroadcastIndex(index, output_shape, rhs_shape), rhs_view);
        output[index] = lhs_value == rhs_value ? 1 : 0;
    }
}

__device__ inline uint16_t FloatToBFloat16Device(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t exponent = bits & 0x7f800000u;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0) return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__global__ void CastToFloatKernel(const int8_t* input, float* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = ReadQ(input, index, input_view);
}

__global__ void CastToHalfKernel(const int8_t* input, uint16_t* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) reinterpret_cast<__half*>(output)[index] = __float2half(ReadQ(input, index, input_view));
}

__global__ void CastToBFloat16Kernel(const int8_t* input, BFloat16* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index].bits = FloatToBFloat16Device(ReadQ(input, index, input_view));
}

__global__ void CastToInt8Kernel(const int8_t* input, int8_t* output, int64_t count, Int8View input_view, Int8View output_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = WriteQ(ReadQ(input, index, input_view), output_view);
}

__global__ void CastToInt32Kernel(const int8_t* input, int32_t* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<int32_t>(ReadQ(input, index, input_view));
}

__global__ void CastToInt64Kernel(const int8_t* input, int64_t* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<int64_t>(ReadQ(input, index, input_view));
}

__global__ void CastToUInt8Kernel(const int8_t* input, uint8_t* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<uint8_t>(ReadQ(input, index, input_view));
}

__global__ void CastToBoolKernel(const int8_t* input, uint8_t* output, int64_t count, Int8View input_view) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = ReadQ(input, index, input_view) != 0.0f;
}

inline int32_t RunUnary(operators::UnaryParam* param, int op, const char* name) {
    AutoTimer timer(name);
    if (param == nullptr) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) ||
        !BuildInt8OutputView(param->out.get(), param->input->dims().data(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    UnaryQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_view, output_view, op);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunBinary(operators::BinaryParam* param, int op, const char* name) {
    AutoTimer timer(name);
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) return -1;
    std::vector<int64_t> output_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        !SameDims(param->out.get(), output_dims)) return -1;
    CudaShape output_shape, lhs_shape, rhs_shape;
    if (!MakeShape(output_dims, &output_shape) || !MakeShape(param->lhs->dims().data(), &lhs_shape) ||
        !MakeShape(param->rhs->dims().data(), &rhs_shape)) return -1;
    Int8View lhs_view, rhs_view, output_view;
    if (!BuildInt8View(param->lhs.get(), &lhs_view) || !BuildInt8View(param->rhs.get(), &rhs_view) ||
        !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes lhs, rhs, output;
    if (!CopyToDevice(param->lhs.get(), &lhs) || !CopyToDevice(param->rhs.get(), &rhs) ||
        !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    BinaryQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(lhs.as<int8_t>(), rhs.as<int8_t>(), output.as<int8_t>(), param->out->numel(), output_shape, lhs_shape, rhs_shape, lhs_view, rhs_view, output_view, op);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunCopy(const Tensor* input_tensor, Tensor* output_tensor, const char* name) {
    AutoTimer timer(name);
    if (input_tensor == nullptr || output_tensor == nullptr || input_tensor->numel() != output_tensor->numel()) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(input_tensor, &input_view) || !BuildInt8View(output_tensor, &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(input_tensor, &input) || !output.allocate(static_cast<size_t>(output_tensor->numel()))) return -1;
    CopyQKernel<<<Blocks(output_tensor->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), output_tensor->numel(), input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, output_tensor) ? 0 : -1;
}

inline int32_t RunPow(operators::PowParam* param) {
    AutoTimer timer("CUDA::PowKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) ||
        !BuildInt8OutputView(param->out.get(), param->input->dims().data(), &output_view)) return -1;
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadHostScalar(param->exponent_tensor.get(), &exponent)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    PowQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_view, output_view, exponent);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunTranspose(operators::TransposeParam* param) {
    AutoTimer timer("CUDA::TransposeKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->dims().size() != param->perm.size()) return -1;
    CudaShape input_shape, output_shape;
    if (!MakeShape(param->input->dims().data(), &input_shape) || !MakeShape(param->out->dims().data(), &output_shape) ||
        input_shape.rank != output_shape.rank) return -1;
    std::vector<int> perm(static_cast<size_t>(input_shape.rank));
    std::vector<bool> seen(static_cast<size_t>(input_shape.rank), false);
    for (int axis = 0; axis < input_shape.rank; ++axis) {
        int64_t source = param->perm[static_cast<size_t>(axis)];
        if (source < 0) source += input_shape.rank;
        if (source < 0 || source >= input_shape.rank || seen[static_cast<size_t>(source)] ||
            output_shape.dims[axis] != input_shape.dims[source]) return -1;
        seen[static_cast<size_t>(source)] = true;
        perm[static_cast<size_t>(axis)] = static_cast<int>(source);
    }
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output, perm_device;
    if (!CopyToDevice(param->input.get(), &input) || !UploadVector(perm, &perm_device) ||
        !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    TransposeQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_shape, output_shape, perm_device.as<int>(), input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int64_t NormalizeSliceStart(int64_t value, int64_t dim, int64_t step) {
    if (step > 0) {
        if (value < 0) value += dim;
        return std::max<int64_t>(0, std::min<int64_t>(dim, value));
    }
    if (value == std::numeric_limits<int64_t>::max()) value = dim - 1;
    else if (value < 0) value += dim;
    return std::max<int64_t>(-1, std::min<int64_t>(dim - 1, value));
}

inline int64_t NormalizeSliceEnd(int64_t value, int64_t dim, int64_t step) {
    if (step > 0) {
        if (value < 0) value += dim;
        return std::max<int64_t>(0, std::min<int64_t>(dim, value));
    }
    if (value == std::numeric_limits<int64_t>::min()) value = -1;
    else if (value < 0) value += dim;
    return std::max<int64_t>(-1, std::min<int64_t>(dim - 1, value));
}

inline int32_t RunSlice(operators::SliceParam* param) {
    AutoTimer timer("CUDA::SliceKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    const auto& input_dims = param->input->dims().data();
    const int rank = static_cast<int>(input_dims.size());
    if (rank <= 0 || rank > kMaxRank) return -1;
    std::vector<int64_t> starts(static_cast<size_t>(rank), 0);
    std::vector<int64_t> ends = input_dims;
    std::vector<int64_t> steps(static_cast<size_t>(rank), 1);
    if (param->starts != nullptr && param->ends != nullptr) {
        std::vector<int64_t> raw_starts, raw_ends, axes, raw_steps;
        if (!ReadHostInt64(param->starts.get(), &raw_starts) || !ReadHostInt64(param->ends.get(), &raw_ends)) return -1;
        if (param->axes != nullptr && !ReadHostInt64(param->axes.get(), &axes)) return -1;
        if (param->steps != nullptr && !ReadHostInt64(param->steps.get(), &raw_steps)) return -1;
        if (axes.empty()) {
            axes.resize(raw_starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        if (raw_starts.size() != raw_ends.size() || axes.size() != raw_starts.size() ||
            (!raw_steps.empty() && raw_steps.size() != raw_starts.size())) return -1;
        for (size_t index = 0; index < axes.size(); ++index) {
            int64_t axis = axes[index] < 0 ? axes[index] + rank : axes[index];
            if (axis < 0 || axis >= rank) return -1;
            const int64_t step = raw_steps.empty() ? 1 : raw_steps[index];
            if (step == 0) return -1;
            steps[static_cast<size_t>(axis)] = step;
            starts[static_cast<size_t>(axis)] = NormalizeSliceStart(raw_starts[index], input_dims[static_cast<size_t>(axis)], step);
            ends[static_cast<size_t>(axis)] = NormalizeSliceEnd(raw_ends[index], input_dims[static_cast<size_t>(axis)], step);
        }
    } else {
        if (param->axis < 0 || param->axis >= rank) return -1;
        starts[static_cast<size_t>(param->axis)] = NormalizeSliceStart(param->start, input_dims[static_cast<size_t>(param->axis)], 1);
        ends[static_cast<size_t>(param->axis)] = NormalizeSliceEnd(param->end, input_dims[static_cast<size_t>(param->axis)], 1);
    }
    CudaShape input_shape, output_shape;
    if (!MakeShape(input_dims, &input_shape) || !MakeShape(param->out->dims().data(), &output_shape)) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output, starts_device, steps_device;
    if (!CopyToDevice(param->input.get(), &input) || !UploadVector(starts, &starts_device) || !UploadVector(steps, &steps_device) ||
        !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    SliceQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_shape, output_shape, starts_device.as<int64_t>(), steps_device.as<int64_t>(), input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunExpand(operators::ExpandParam* param) {
    AutoTimer timer("CUDA::ExpandKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->shape == nullptr || param->out == nullptr) return -1;
    std::vector<int64_t> shape;
    if (!ReadHostInt64(param->shape.get(), &shape) || !SameDims(param->out.get(), shape)) return -1;
    std::vector<int64_t> inferred;
    if (!InferBroadcastShape(param->input->dims().data(), shape, &inferred) || inferred != shape) return -1;
    CudaShape input_shape, output_shape;
    if (!MakeShape(param->input->dims().data(), &input_shape) || !MakeShape(shape, &output_shape)) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    ExpandQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_shape, output_shape, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunWhere(operators::WhereParam* param) {
    AutoTimer timer("CUDA::WhereKernel::INT8");
    if (param == nullptr || param->condition == nullptr || param->x == nullptr || param->y == nullptr || param->out == nullptr ||
        !ReadyStorage(param->condition.get(), DataType::BOOL)) return -1;
    std::vector<int64_t> xy_dims, output_dims;
    if (!InferBroadcastShape(param->x->dims().data(), param->y->dims().data(), &xy_dims) ||
        !InferBroadcastShape(xy_dims, param->condition->dims().data(), &output_dims) || !SameDims(param->out.get(), output_dims)) return -1;
    CudaShape output_shape, condition_shape, x_shape, y_shape;
    if (!MakeShape(output_dims, &output_shape) || !MakeShape(param->condition->dims().data(), &condition_shape) ||
        !MakeShape(param->x->dims().data(), &x_shape) || !MakeShape(param->y->dims().data(), &y_shape)) return -1;
    Int8View x_view, y_view, output_view;
    if (!BuildInt8View(param->x.get(), &x_view) || !BuildInt8View(param->y.get(), &y_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes condition, x, y, output;
    if (!CopyToDevice(param->condition.get(), &condition) || !CopyToDevice(param->x.get(), &x) || !CopyToDevice(param->y.get(), &y) ||
        !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    WhereQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(condition.as<uint8_t>(), x.as<int8_t>(), y.as<int8_t>(), output.as<int8_t>(), param->out->numel(), output_shape, condition_shape, x_shape, y_shape, x_view, y_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunGather(operators::GatherParam* param) {
    AutoTimer timer("CUDA::GatherKernel::INT8");
    if (param == nullptr || param->data == nullptr || param->indices == nullptr || param->out == nullptr) return -1;
    const int rank = static_cast<int>(param->data->dims().size());
    int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    std::vector<int64_t> expected;
    for (int index = 0; index < axis; ++index) expected.push_back(param->data->dims()[static_cast<size_t>(index)]);
    for (size_t index = 0; index < param->indices->dims().size(); ++index) expected.push_back(param->indices->dims()[index]);
    for (size_t index = static_cast<size_t>(axis + 1); index < param->data->dims().size(); ++index) expected.push_back(param->data->dims()[index]);
    if (!SameDims(param->out.get(), expected)) return -1;
    std::vector<int64_t> indices;
    if (!ReadHostInt64(param->indices.get(), &indices)) return -1;
    for (int64_t value : indices) if (value >= param->data->dims()[static_cast<size_t>(axis)] || value < -param->data->dims()[static_cast<size_t>(axis)]) return -1;
    CudaShape input_shape, indices_shape, output_shape;
    if (!MakeShape(param->data->dims().data(), &input_shape) || !MakeShape(param->indices->dims().data(), &indices_shape) || !MakeShape(expected, &output_shape)) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->data.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, index_device, output;
    if (!CopyToDevice(param->data.get(), &input) || !UploadVector(indices, &index_device) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    GatherQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), index_device.as<int64_t>(), output.as<int8_t>(), param->out->numel(), input_shape, indices_shape, output_shape, axis, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunResize(operators::ResizeParam* param) {
    AutoTimer timer("CUDA::ResizeKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->dims().size() != param->out->dims().size()) return -1;
    CudaShape input_shape, output_shape;
    if (!MakeShape(param->input->dims().data(), &input_shape) || !MakeShape(param->out->dims().data(), &output_shape)) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    ResizeQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_shape, output_shape, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunConcat(operators::ConcatParam* param) {
    AutoTimer timer("CUDA::ConcatKernel::INT8");
    if (param == nullptr || param->out == nullptr || param->inputs.empty()) return -1;
    const int rank = static_cast<int>(param->out->dims().size());
    int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    int64_t axis_total = 0;
    for (const auto& tensor : param->inputs) {
        if (!tensor || tensor->dims().size() != static_cast<size_t>(rank)) return -1;
        for (int index = 0; index < rank; ++index) if (index != axis && tensor->dims()[index] != param->out->dims()[index]) return -1;
        axis_total += tensor->dims()[static_cast<size_t>(axis)];
    }
    if (axis_total != param->out->dims()[static_cast<size_t>(axis)]) return -1;
    CudaShape output_shape;
    if (!MakeShape(param->out->dims().data(), &output_shape)) return -1;
    Int8View output_view;
    if (!BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes output;
    if (!output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    int64_t axis_offset = 0;
    for (const auto& tensor : param->inputs) {
        CudaShape input_shape;
        Int8View input_view;
        if (!MakeShape(tensor->dims().data(), &input_shape) || !BuildInt8View(tensor.get(), &input_view)) return -1;
        DeviceBytes input;
        if (!CopyToDevice(tensor.get(), &input)) return -1;
        ConcatQKernel<<<Blocks(tensor->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), tensor->numel(), input_shape, output_shape, axis, axis_offset, input_view, output_view);
        if (!LaunchSucceeded()) return -1;
        axis_offset += tensor->dims()[static_cast<size_t>(axis)];
    }
    return CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunSplit(operators::SplitParam* param) {
    AutoTimer timer("CUDA::SplitKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->outputs.empty()) return -1;
    const int rank = static_cast<int>(param->input->dims().size());
    int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    CudaShape input_shape;
    Int8View input_view;
    if (!MakeShape(param->input->dims().data(), &input_shape) || !BuildInt8View(param->input.get(), &input_view)) return -1;
    DeviceBytes input;
    if (!CopyToDevice(param->input.get(), &input)) return -1;
    int64_t axis_offset = 0;
    for (const auto& tensor : param->outputs) {
        if (!tensor || tensor->dims().size() != static_cast<size_t>(rank)) return -1;
        for (int index = 0; index < rank; ++index) if (index != axis && tensor->dims()[index] != param->input->dims()[index]) return -1;
        CudaShape output_shape;
        Int8View output_view;
        if (!MakeShape(tensor->dims().data(), &output_shape) || !BuildInt8View(tensor.get(), &output_view)) return -1;
        DeviceBytes output;
        if (!output.allocate(static_cast<size_t>(tensor->numel()))) return -1;
        SplitQKernel<<<Blocks(tensor->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), tensor->numel(), input_shape, output_shape, axis, axis_offset, input_view, output_view);
        if (!LaunchSucceeded() || !CopyFromDevice(output, tensor.get())) return -1;
        axis_offset += tensor->dims()[static_cast<size_t>(axis)];
    }
    return axis_offset == param->input->dims()[static_cast<size_t>(axis)] ? 0 : -1;
}

inline int32_t RunReduce(const Tensor* input_tensor, Tensor* output_tensor, const std::vector<int64_t>& axes,
                         bool keepdims, bool mean, const char* name) {
    AutoTimer timer(name);
    if (input_tensor == nullptr || output_tensor == nullptr) return -1;
    const int rank = static_cast<int>(input_tensor->dims().size());
    std::vector<int64_t> normalized;
    if (!NormalizeAxes(axes, rank, &normalized)) return -1;
    int axis_mask = 0;
    int64_t reduce_count = 1;
    for (int64_t axis : normalized) {
        axis_mask |= 1 << axis;
        if (reduce_count > std::numeric_limits<int64_t>::max() / input_tensor->dims()[static_cast<size_t>(axis)]) return -1;
        reduce_count *= input_tensor->dims()[static_cast<size_t>(axis)];
    }
    std::vector<int64_t> expected;
    if (keepdims) {
        expected = input_tensor->dims().data();
        for (int64_t axis : normalized) expected[static_cast<size_t>(axis)] = 1;
    } else {
        for (int axis = 0; axis < rank; ++axis) if ((axis_mask & (1 << axis)) == 0) expected.push_back(input_tensor->dims()[static_cast<size_t>(axis)]);
    }
    if (expected.empty()) expected.push_back(1);
    if (!SameDims(output_tensor, expected)) return -1;
    CudaShape input_shape, output_shape;
    Int8View input_view, output_view;
    if (!MakeShape(input_tensor->dims().data(), &input_shape) || !MakeShape(expected, &output_shape) ||
        !BuildInt8View(input_tensor, &input_view) || !BuildInt8View(output_tensor, &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(input_tensor, &input) || !output.allocate(static_cast<size_t>(output_tensor->numel()))) return -1;
    ReduceQKernel<<<Blocks(output_tensor->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), output_tensor->numel(), input_shape, output_shape, axis_mask, mean, reduce_count, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, output_tensor) ? 0 : -1;
}

inline int32_t RunSoftmax(operators::SoftmaxParam* param) {
    AutoTimer timer("CUDA::SoftmaxKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !SameDims(param->out.get(), param->input->dims().data())) return -1;
    const int rank = static_cast<int>(param->input->dims().size());
    int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    const int64_t outer = Product(param->input->dims().data(), 0, static_cast<size_t>(axis));
    const int64_t axis_dim = param->input->dims()[static_cast<size_t>(axis)];
    const int64_t inner = Product(param->input->dims().data(), static_cast<size_t>(axis + 1), param->input->dims().size());
    if (outer <= 0 || axis_dim <= 0 || inner <= 0) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    SoftmaxQKernel<<<Blocks(outer * inner), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), outer * inner, axis_dim, inner, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunPool(operators::PoolParam* param, bool maximum, const char* name) {
    AutoTimer timer(name);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    int64_t in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w;
    bool channel_last, output_channel_last;
    if (!ReadImageShape(param->input.get(), &in_n, &in_c, &in_h, &in_w, &channel_last) ||
        !ReadImageShape(param->out.get(), &out_n, &out_c, &out_h, &out_w, &output_channel_last) ||
        in_n != out_n || in_c != out_c || channel_last != output_channel_last) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    PoolQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), in_c, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w, param->stride_h, param->stride_w, param->pad_h, param->pad_w, in_n, channel_last, maximum, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunGlobalAveragePool(operators::GlobalAveragePoolParam* param) {
    AutoTimer timer("CUDA::GlobalAveragePoolKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr) return -1;
    int64_t in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w;
    bool channel_last, output_channel_last;
    if (!ReadImageShape(param->input.get(), &in_n, &in_c, &in_h, &in_w, &channel_last) ||
        !ReadImageShape(param->out.get(), &out_n, &out_c, &out_h, &out_w, &output_channel_last) ||
        in_n != out_n || in_c != out_c || out_h != 1 || out_w != 1 || channel_last != output_channel_last) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    GlobalAveragePoolQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), in_c, in_h, in_w, in_n, channel_last, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunBatchNorm(operators::BatchNormParam* param) {
    AutoTimer timer("CUDA::BatchNormalizationKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->dims().size() != 4 ||
        !SameDims(param->out.get(), param->input->dims().data())) return -1;
    int64_t n, c, h, w;
    bool channel_last;
    if (!ReadImageShape(param->input.get(), &n, &c, &h, &w, &channel_last) || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr || param->scale->numel() != c || param->bias->numel() != c ||
        param->mean->numel() != c || param->var->numel() != c) return -1;
    std::vector<float> scale(static_cast<size_t>(c)), bias(static_cast<size_t>(c)), mean(static_cast<size_t>(c)), variance(static_cast<size_t>(c));
    for (int64_t index = 0; index < c; ++index) {
        if (!ReadHostFloat(param->scale.get(), index, &scale[static_cast<size_t>(index)]) ||
            !ReadHostFloat(param->bias.get(), index, &bias[static_cast<size_t>(index)]) ||
            !ReadHostFloat(param->mean.get(), index, &mean[static_cast<size_t>(index)]) ||
            !ReadHostFloat(param->var.get(), index, &variance[static_cast<size_t>(index)]) ||
            variance[static_cast<size_t>(index)] + param->epsilon <= 0.0f) return -1;
    }
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view) || !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output, scale_device, bias_device, mean_device, variance_device;
    if (!CopyToDevice(param->input.get(), &input) || !UploadVector(scale, &scale_device) || !UploadVector(bias, &bias_device) ||
        !UploadVector(mean, &mean_device) || !UploadVector(variance, &variance_device) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    BatchNormQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(input.as<int8_t>(), scale_device.as<float>(), bias_device.as<float>(), mean_device.as<float>(), variance_device.as<float>(), output.as<int8_t>(), param->out->numel(), c, h, w, n, channel_last, param->epsilon, input_view, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunEqual(operators::EqualParam* param) {
    AutoTimer timer("CUDA::EqualKernel::INT8");
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        !ReadyStorage(param->out.get(), DataType::BOOL)) return -1;
    std::vector<int64_t> output_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) || !SameDims(param->out.get(), output_dims)) return -1;
    CudaShape output_shape, lhs_shape, rhs_shape;
    if (!MakeShape(output_dims, &output_shape) || !MakeShape(param->lhs->dims().data(), &lhs_shape) || !MakeShape(param->rhs->dims().data(), &rhs_shape)) return -1;
    Int8View lhs_view, rhs_view;
    if (!BuildInt8View(param->lhs.get(), &lhs_view) || !BuildInt8View(param->rhs.get(), &rhs_view)) return -1;
    DeviceBytes lhs, rhs, output;
    if (!CopyToDevice(param->lhs.get(), &lhs) || !CopyToDevice(param->rhs.get(), &rhs) || !output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    EqualQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(lhs.as<int8_t>(), rhs.as<int8_t>(), output.as<uint8_t>(), param->out->numel(), output_shape, lhs_shape, rhs_shape, lhs_view, rhs_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunCast(operators::CastParam* param) {
    AutoTimer timer("CUDA::CastKernel::INT8");
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        !SameDims(param->out.get(), param->input->dims().data()) || param->out->data_type() != param->to ||
        !ReadyStorage(param->input.get(), DataType::INT8)) return -1;
    Int8View input_view, output_view;
    if (!BuildInt8View(param->input.get(), &input_view)) return -1;
    if (param->to == DataType::INT8 && !BuildInt8View(param->out.get(), &output_view)) return -1;
    DeviceBytes input, output;
    if (!CopyToDevice(param->input.get(), &input) ||
        !output.allocate(static_cast<size_t>(param->out->numel()) * DataTypeBytes(param->to))) return -1;
    const int blocks = Blocks(param->out->numel());
    switch (param->to) {
        case DataType::FP32: CastToFloatKernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<float>(), param->out->numel(), input_view); break;
        case DataType::FP16: CastToHalfKernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<uint16_t>(), param->out->numel(), input_view); break;
        case DataType::BF16: CastToBFloat16Kernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<BFloat16>(), param->out->numel(), input_view); break;
        case DataType::INT8: CastToInt8Kernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<int8_t>(), param->out->numel(), input_view, output_view); break;
        case DataType::INT32: CastToInt32Kernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<int32_t>(), param->out->numel(), input_view); break;
        case DataType::INT64: CastToInt64Kernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<int64_t>(), param->out->numel(), input_view); break;
        case DataType::UINT8: CastToUInt8Kernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<uint8_t>(), param->out->numel(), input_view); break;
        case DataType::BOOL: CastToBoolKernel<<<blocks, kCudaThreads>>>(input.as<int8_t>(), output.as<uint8_t>(), param->out->numel(), input_view); break;
        default: return -1;
    }
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

inline int32_t RunConstantOfShape(operators::ConstantOfShapeParam* param) {
    AutoTimer timer("CUDA::ConstantOfShapeKernel::INT8");
    if (param == nullptr || param->shape == nullptr || param->out == nullptr || param->out->data_type() != DataType::INT8) return -1;
    std::vector<int64_t> shape;
    if (!ReadHostInt64(param->shape.get(), &shape) || !SameDims(param->out.get(), shape)) return -1;
    Int8View output_view;
    if (!BuildInt8View(param->out.get(), &output_view)) return -1;
    const float value = param->use_float_value ? param->float_value : static_cast<float>(param->int_value);
    DeviceBytes output;
    if (!output.allocate(static_cast<size_t>(param->out->numel()))) return -1;
    FillQKernel<<<Blocks(param->out->numel()), kCudaThreads>>>(output.as<int8_t>(), param->out->numel(), value, output_view);
    return LaunchSucceeded() && CopyFromDevice(output, param->out.get()) ? 0 : -1;
}

}  // namespace

#define CUDA_UNARY(KERNEL, OP) \
    template <> int32_t KERNEL<DeviceType::CUDA, DataType::INT8>::compute() { \
        return RunUnary(static_cast<operators::UnaryParam*>(param_), OP, "CUDA::" #KERNEL "::INT8"); \
    }
#define CUDA_BINARY(KERNEL, OP) \
    template <> int32_t KERNEL<DeviceType::CUDA, DataType::INT8>::compute() { \
        return RunBinary(static_cast<operators::BinaryParam*>(param_), OP, "CUDA::" #KERNEL "::INT8"); \
    }

CUDA_UNARY(ReluKernel, 0)
CUDA_UNARY(NegKernel, 1)
CUDA_UNARY(SigmoidKernel, 2)
CUDA_UNARY(SiluKernel, 3)
CUDA_UNARY(ExpKernel, 4)
CUDA_UNARY(SqrtKernel, 5)
CUDA_UNARY(TanhKernel, 6)
CUDA_UNARY(ErfKernel, 7)
CUDA_UNARY(SinKernel, 8)
CUDA_UNARY(CosKernel, 9)
CUDA_UNARY(SoftplusKernel, 10)
CUDA_BINARY(AddKernel, 0)
CUDA_BINARY(SubKernel, 1)
CUDA_BINARY(MulKernel, 2)
CUDA_BINARY(DivKernel, 3)

template <> int32_t PowKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunPow(static_cast<operators::PowParam*>(param_)); }
template <> int32_t BatchNormalizationKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunBatchNorm(static_cast<operators::BatchNormParam*>(param_)); }
template <> int32_t AvgPoolKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunPool(static_cast<operators::PoolParam*>(param_), false, "CUDA::AvgPoolKernel::INT8"); }
template <> int32_t MaxPoolKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunPool(static_cast<operators::PoolParam*>(param_), true, "CUDA::MaxPoolKernel::INT8"); }
template <> int32_t GlobalAveragePoolKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunGlobalAveragePool(static_cast<operators::GlobalAveragePoolParam*>(param_)); }
template <> int32_t ReduceSumKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::ReduceSumParam*>(param_); return p == nullptr ? -1 : RunReduce(p->input.get(), p->out.get(), p->axes, p->keepdims, false, "CUDA::ReduceSumKernel::INT8"); }
template <> int32_t ReduceMeanKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::ReduceMeanParam*>(param_); return p == nullptr ? -1 : RunReduce(p->input.get(), p->out.get(), p->axes, p->keepdims, true, "CUDA::ReduceMeanKernel::INT8"); }
template <> int32_t IdentityKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::UnaryParam*>(param_); return p == nullptr ? -1 : RunCopy(p->input.get(), p->out.get(), "CUDA::IdentityKernel::INT8"); }
template <> int32_t ReshapeKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::ReshapeParam*>(param_); return p == nullptr ? -1 : RunCopy(p->input.get(), p->out.get(), "CUDA::ReshapeKernel::INT8"); }
template <> int32_t FlattenKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::FlattenParam*>(param_); return p == nullptr ? -1 : RunCopy(p->input.get(), p->out.get(), "CUDA::FlattenKernel::INT8"); }
template <> int32_t TransposeKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunTranspose(static_cast<operators::TransposeParam*>(param_)); }
template <> int32_t SqueezeKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::AxesParam*>(param_); return p == nullptr ? -1 : RunCopy(p->input.get(), p->out.get(), "CUDA::SqueezeKernel::INT8"); }
template <> int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::INT8>::compute() { auto* p = static_cast<operators::AxesParam*>(param_); return p == nullptr ? -1 : RunCopy(p->input.get(), p->out.get(), "CUDA::UnsqueezeKernel::INT8"); }
template <> int32_t SliceKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunSlice(static_cast<operators::SliceParam*>(param_)); }
template <> int32_t SplitKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunSplit(static_cast<operators::SplitParam*>(param_)); }
template <> int32_t ConcatKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunConcat(static_cast<operators::ConcatParam*>(param_)); }
template <> int32_t ExpandKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunExpand(static_cast<operators::ExpandParam*>(param_)); }
template <> int32_t GatherKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunGather(static_cast<operators::GatherParam*>(param_)); }
template <> int32_t WhereKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunWhere(static_cast<operators::WhereParam*>(param_)); }
template <> int32_t ResizeKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunResize(static_cast<operators::ResizeParam*>(param_)); }
template <> int32_t SoftmaxKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunSoftmax(static_cast<operators::SoftmaxParam*>(param_)); }
template <> int32_t EqualKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunEqual(static_cast<operators::EqualParam*>(param_)); }
template <> int32_t CastKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunCast(static_cast<operators::CastParam*>(param_)); }
template <> int32_t ConstantOfShapeKernel<DeviceType::CUDA, DataType::INT8>::compute() { return RunConstantOfShape(static_cast<operators::ConstantOfShapeParam*>(param_)); }

#undef CUDA_UNARY
#undef CUDA_BINARY

void EnsureCudaStandardInt8KernelsRegistered() {
    static const bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
#define REG(OP, KERNEL) dispatcher.registerKernel(DeviceType::CUDA, DataType::INT8, OP, []() { return std::make_unique<KERNEL<DeviceType::CUDA, DataType::INT8>>(); })
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
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
