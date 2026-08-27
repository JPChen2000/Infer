#ifndef FEATHER_KERNEL_FP8_HOST_H
#define FEATHER_KERNEL_FP8_HOST_H

// Shared, scale-aware host implementations used by the Common and x86 FP8
// kernels.  FP8 storage is intentionally never reinterpreted as an integer:
// every read is decoded to FP32 and every write is quantized with the
// destination tensor's scale.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/common/tensor_op_utils.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {
namespace fp8_host {

inline std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<size_t>(axis)] =
            strides[static_cast<size_t>(axis + 1)] * dims[static_cast<size_t>(axis + 1)];
    }
    return strides;
}

inline bool HasValidShape(const std::vector<int64_t>& dims, bool allow_scalar = true) {
    if (!allow_scalar && dims.empty()) return false;
    for (const int64_t dim : dims) {
        if (dim <= 0) return false;
    }
    return true;
}

inline bool SafeProduct(const std::vector<int64_t>& dims, size_t begin, size_t end, int64_t* product) {
    if (product == nullptr || begin > end || end > dims.size()) return false;
    int64_t result = 1;
    for (size_t i = begin; i < end; ++i) {
        if (dims[i] <= 0 || result > std::numeric_limits<int64_t>::max() / dims[i]) return false;
        result *= dims[i];
    }
    *product = result;
    return true;
}

inline bool SafeRequiredBytes(int64_t numel, size_t element_bytes, size_t* bytes) {
    if (bytes == nullptr || numel < 0 || element_bytes == 0) return false;
    const auto count = static_cast<uint64_t>(numel);
    if (count > std::numeric_limits<size_t>::max() / element_bytes) return false;
    *bytes = static_cast<size_t>(count) * element_bytes;
    return true;
}

inline bool IsValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

inline bool HasValidFp8Quantization(const Tensor* tensor) {
    return tensor != nullptr && HasCompatiblePerTensorQuantization(tensor->quantization()) &&
           IsValidScale(tensor->quantization_scale());
}

inline bool TryScaledDimension(int64_t input_dim, float scale, int64_t* output_dim) {
    if (output_dim == nullptr || input_dim <= 0 || !IsValidScale(scale)) return false;
    const long double scaled = static_cast<long double>(input_dim) * static_cast<long double>(scale);
    const long double rounded = std::floor(scaled + 0.5L);
    if (!std::isfinite(static_cast<double>(scaled)) || rounded < 1.0L ||
        rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *output_dim = static_cast<int64_t>(rounded);
    return true;
}

inline int64_t ResizeCoordinate(int64_t output_coordinate, float scale, int64_t input_dim) {
    if (output_coordinate <= 0) return 0;
    const long double source = static_cast<long double>(output_coordinate) / static_cast<long double>(scale);
    if (!std::isfinite(static_cast<double>(source)) || source >= static_cast<long double>(input_dim - 1)) {
        return input_dim - 1;
    }
    return source <= 0.0L ? 0 : static_cast<int64_t>(source);
}

inline bool TryWindowOutputDimension(int64_t input_dim, int64_t kernel, int64_t stride, int64_t pad,
                                     int64_t dilation, int64_t* output_dim) {
    if (output_dim == nullptr || input_dim <= 0 || kernel <= 0 || stride <= 0 || pad < 0 || dilation <= 0 ||
        pad > std::numeric_limits<int64_t>::max() / 2 ||
        kernel - 1 > (std::numeric_limits<int64_t>::max() - 1) / dilation) return false;
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;
    if (input_dim > std::numeric_limits<int64_t>::max() - 2 * pad) return false;
    const int64_t padded = input_dim + 2 * pad;
    if (padded < effective_kernel) return false;
    const int64_t result = (padded - effective_kernel) / stride + 1;
    if (result <= 0) return false;
    *output_dim = result;
    return true;
}

inline bool IsInitialized(const Tensor* tensor) {
    return tensor != nullptr && tensor->IsInitialized();
}

inline bool SafeTensorNumel(const Tensor* tensor, int64_t* numel) {
    if (tensor == nullptr || numel == nullptr || !SafeProduct(tensor->dims().data(), 0, tensor->dims().size(), numel)) {
        return false;
    }
    return tensor->numel() == *numel;
}

inline bool IsIndexTensor(const Tensor* tensor) {
    if (!IsInitialized(tensor) || (tensor->data_type() != DataType::INT32 && tensor->data_type() != DataType::INT64)) {
        return false;
    }
    int64_t numel = 0;
    size_t required_bytes = 0;
    return HasValidShape(tensor->dims().data()) && SafeTensorNumel(tensor, &numel) &&
           SafeRequiredBytes(numel, DataTypeBytes(tensor->data_type()), &required_bytes) &&
           tensor->memory_size() >= required_bytes;
}

inline bool IsBoolTensor(const Tensor* tensor) {
    if (!IsInitialized(tensor) || tensor->data_type() != DataType::BOOL) return false;
    int64_t numel = 0;
    size_t required_bytes = 0;
    return HasValidShape(tensor->dims().data()) && SafeTensorNumel(tensor, &numel) &&
           SafeRequiredBytes(numel, sizeof(uint8_t), &required_bytes) && tensor->memory_size() >= required_bytes;
}

template <DataType dtype>
inline bool IsTensor(const Tensor* tensor) {
    int64_t numel = 0;
    size_t required_bytes = 0;
    return IsInitialized(tensor) && tensor->data_type() == dtype && HasValidShape(tensor->dims().data()) &&
           SafeTensorNumel(tensor, &numel) && SafeRequiredBytes(numel, DataTypeBytes(dtype), &required_bytes) &&
           tensor->memory_size() >= required_bytes && HasValidFp8Quantization(tensor);
}

template <DataType dtype>
inline bool PrepareOutput(Tensor* tensor, const std::vector<int64_t>* expected_dims = nullptr) {
    if (!IsInitialized(tensor) ||
        (tensor->data_type() != DataType::UNKNOWN && tensor->data_type() != dtype) ||
        !HasValidShape(tensor->dims().data()) ||
        (expected_dims != nullptr && tensor->dims().data() != *expected_dims)) {
        return false;
    }
    int64_t numel = 0;
    size_t required_bytes = 0;
    if (!SafeTensorNumel(tensor, &numel) || !SafeRequiredBytes(numel, DataTypeBytes(dtype), &required_bytes) ||
        tensor->memory_size() < required_bytes || !HasValidFp8Quantization(tensor)) {
        return false;
    }
    tensor->set_data_type(dtype);
    return true;
}

template <DataType dtype>
inline bool ValidateFc(operators::FcParam* param, int64_t* rows, int64_t* inner, int64_t* columns) {
    if (param == nullptr || rows == nullptr || inner == nullptr || columns == nullptr ||
        !IsTensor<dtype>(param->input.get()) || !IsTensor<dtype>(param->w.get()) ||
        param->input->dims().size() != 2 || param->w->dims().size() != 2) {
        return false;
    }
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    if (m <= 0 || k <= 0 || n <= 0 || param->w->dims()[0] != k ||
        m > std::numeric_limits<int64_t>::max() / n) {
        return false;
    }
    if (param->bias != nullptr &&
        (!IsTensor<dtype>(param->bias.get()) ||
         (param->bias->dims().data() != std::vector<int64_t>{n} &&
          param->bias->dims().data() != std::vector<int64_t>{m, n}))) {
        return false;
    }
    const std::vector<int64_t> expected_dims{m, n};
    if (!PrepareOutput<dtype>(param->out.get(), &expected_dims)) return false;
    *rows = m;
    *inner = k;
    *columns = n;
    return true;
}

inline bool InferBroadcast(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                           std::vector<int64_t>* output) {
    if (output == nullptr) {
        return false;
    }
    const size_t rank = std::max(lhs.size(), rhs.size());
    output->assign(rank, 1);
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t lhs_dim = axis < rank - lhs.size() ? 1 : lhs[axis - (rank - lhs.size())];
        const int64_t rhs_dim = axis < rank - rhs.size() ? 1 : rhs[axis - (rank - rhs.size())];
        if (lhs_dim <= 0 || rhs_dim <= 0 || (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1)) {
            return false;
        }
        (*output)[axis] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

inline int64_t BroadcastOffset(const std::vector<int64_t>& coordinates,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides) {
    if (coordinates.size() < input_dims.size()) {
        return -1;
    }
    const size_t gap = coordinates.size() - input_dims.size();
    int64_t offset = 0;
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        const int64_t coordinate = input_dims[axis] == 1 ? 0 : coordinates[axis + gap];
        offset += coordinate * input_strides[axis];
    }
    return offset;
}

template <DataType dtype, typename Fn>
int32_t Unary(operators::UnaryParam* param, Fn&& fn) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) ||
        !PrepareOutput<dtype>(param->out.get(), &param->input->dims().data())) {
        return -1;
    }
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, fn(TensorIO<dtype>::Read(param->input.get(), i)));
    }
    return 0;
}

enum class BinaryOp { kAdd, kMul, kSub, kDiv };

template <BinaryOp op>
inline float ApplyBinary(float lhs, float rhs) {
    if constexpr (op == BinaryOp::kAdd) return lhs + rhs;
    if constexpr (op == BinaryOp::kMul) return lhs * rhs;
    if constexpr (op == BinaryOp::kSub) return lhs - rhs;
    return lhs / rhs;
}

template <DataType dtype, BinaryOp op>
int32_t Binary(operators::BinaryParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->lhs.get()) || !IsTensor<dtype>(param->rhs.get())) {
        return -1;
    }
    std::vector<int64_t> output_dims;
    if (!InferBroadcast(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        !PrepareOutput<dtype>(param->out.get(), &output_dims)) {
        return -1;
    }
    const auto output_strides = Strides(output_dims);
    const auto lhs_strides = Strides(param->lhs->dims().data());
    const auto rhs_strides = Strides(param->rhs->dims().data());
    std::vector<int64_t> coordinates(output_dims.size(), 0);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < output_dims.size(); ++axis) {
            coordinates[axis] = remaining / output_strides[axis];
            remaining %= output_strides[axis];
        }
        const int64_t lhs_offset = BroadcastOffset(coordinates, param->lhs->dims().data(), lhs_strides);
        const int64_t rhs_offset = BroadcastOffset(coordinates, param->rhs->dims().data(), rhs_strides);
        if (lhs_offset < 0 || rhs_offset < 0) return -1;
        TensorIO<dtype>::Write(param->out.get(), linear,
                               ApplyBinary<op>(TensorIO<dtype>::Read(param->lhs.get(), lhs_offset),
                                               TensorIO<dtype>::Read(param->rhs.get(), rhs_offset)));
    }
    return 0;
}

template <DataType dtype>
int32_t Copy(operators::AxesParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) ||
        !PrepareOutput<dtype>(param->out.get()) ||
        param->input->numel() != param->out->numel()) {
        return -1;
    }
    if (param->input->raw_data() == param->out->raw_data()) {
        return 0;
    }
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, TensorIO<dtype>::Read(param->input.get(), i));
    }
    return 0;
}

template <DataType dtype>
int32_t Flatten(operators::FlattenParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) ||
        !PrepareOutput<dtype>(param->out.get()) || param->input->numel() != param->out->numel()) {
        return -1;
    }
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, TensorIO<dtype>::Read(param->input.get(), i));
    }
    return 0;
}

template <DataType dtype>
int32_t Transpose(operators::TransposeParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !PrepareOutput<dtype>(param->out.get())) {
        return -1;
    }
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (param->perm.size() != input_dims.size() || output_dims.size() != input_dims.size()) return -1;
    std::vector<bool> seen(input_dims.size(), false);
    for (size_t axis = 0; axis < param->perm.size(); ++axis) {
        const int64_t source = param->perm[axis];
        if (source < 0 || source >= static_cast<int64_t>(input_dims.size()) || seen[static_cast<size_t>(source)] ||
            output_dims[axis] != input_dims[static_cast<size_t>(source)]) return -1;
        seen[static_cast<size_t>(source)] = true;
    }
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < output_dims.size(); ++axis) {
            const int64_t coordinate = remaining / output_strides[axis];
            remaining %= output_strides[axis];
            input_offset += coordinate * input_strides[static_cast<size_t>(param->perm[axis])];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <DataType dtype>
int32_t Concat(operators::ConcatParam* param) {
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) return -1;
    const auto output_dims = param->out->dims().data();
    const int axis = param->axis < 0 ? param->axis + static_cast<int>(output_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(output_dims.size()) || !PrepareOutput<dtype>(param->out.get())) return -1;
    int64_t outer = 0;
    int64_t inner = 0;
    if (!SafeProduct(output_dims, 0, static_cast<size_t>(axis), &outer) ||
        !SafeProduct(output_dims, static_cast<size_t>(axis) + 1, output_dims.size(), &inner)) return -1;
    int64_t sum_axis = 0;
    for (const auto& input : param->inputs) {
        if (!IsTensor<dtype>(input.get()) || input->dims().size() != output_dims.size()) return -1;
        for (size_t i = 0; i < output_dims.size(); ++i) {
            if (static_cast<int>(i) != axis && input->dims()[i] != output_dims[i]) return -1;
        }
        if (sum_axis > std::numeric_limits<int64_t>::max() - input->dims()[axis]) return -1;
        sum_axis += input->dims()[axis];
    }
    if (sum_axis != output_dims[axis]) return -1;
    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (const auto& input : param->inputs) {
            const int64_t count = input->dims()[axis] * inner;
            const int64_t input_base = outer_idx * count;
            const int64_t output_base = (outer_idx * output_dims[axis] + axis_offset) * inner;
            for (int64_t i = 0; i < count; ++i) {
                TensorIO<dtype>::Write(param->out.get(), output_base + i,
                                       TensorIO<dtype>::Read(input.get(), input_base + i));
            }
            axis_offset += input->dims()[axis];
        }
    }
    return 0;
}

template <DataType dtype>
int32_t Split(operators::SplitParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || param->outputs.empty()) return -1;
    const auto input_dims = param->input->dims().data();
    const int axis = param->axis < 0 ? param->axis + static_cast<int>(input_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int>(input_dims.size())) return -1;
    int64_t outer = 0;
    int64_t inner = 0;
    if (!SafeProduct(input_dims, 0, static_cast<size_t>(axis), &outer) ||
        !SafeProduct(input_dims, static_cast<size_t>(axis) + 1, input_dims.size(), &inner)) return -1;
    int64_t axis_offset = 0;
    for (const auto& output : param->outputs) {
        if (output == nullptr || output->dims().size() != input_dims.size() ||
            !PrepareOutput<dtype>(output.get())) return -1;
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (static_cast<int>(i) != axis && output->dims()[i] != input_dims[i]) return -1;
        }
        const int64_t count = output->dims()[axis] * inner;
        for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            const int64_t input_base = (outer_idx * input_dims[axis] + axis_offset) * inner;
            const int64_t output_base = outer_idx * count;
            for (int64_t i = 0; i < count; ++i) {
                TensorIO<dtype>::Write(output.get(), output_base + i,
                                       TensorIO<dtype>::Read(param->input.get(), input_base + i));
            }
        }
        if (axis_offset > std::numeric_limits<int64_t>::max() - output->dims()[axis]) return -1;
        axis_offset += output->dims()[axis];
    }
    return axis_offset == input_dims[axis] ? 0 : -1;
}

template <DataType dtype>
int32_t Slice(operators::SliceParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !PrepareOutput<dtype>(param->out.get())) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    const int rank = static_cast<int>(input_dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank || output_dims.size() != input_dims.size()) return -1;
    int64_t start = param->start < 0 ? param->start + input_dims[axis] : param->start;
    int64_t end = param->end < 0 ? param->end + input_dims[axis] : param->end;
    start = std::max<int64_t>(0, std::min<int64_t>(input_dims[axis], start));
    end = std::max<int64_t>(start, std::min<int64_t>(input_dims[axis], end));
    if (output_dims[axis] != end - start) return -1;
    for (int i = 0; i < rank; ++i) if (i != axis && output_dims[i] != input_dims[i]) return -1;
    const auto in_strides = Strides(input_dims);
    const auto out_strides = Strides(output_dims);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        int64_t input_offset = 0;
        for (int i = 0; i < rank; ++i) {
            const int64_t coordinate = remaining / out_strides[i];
            remaining %= out_strides[i];
            input_offset += (coordinate + (i == axis ? start : 0)) * in_strides[i];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

inline bool NormalizeReductionAxes(const std::vector<int64_t>& axes, int rank, std::vector<int64_t>* normalized) {
    if (rank < 0 || normalized == nullptr) return false;
    normalized->clear();
    if (axes.empty()) {
        normalized->resize(static_cast<size_t>(rank));
        std::iota(normalized->begin(), normalized->end(), int64_t{0});
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

inline std::vector<int64_t> ReducedShape(const std::vector<int64_t>& input_dims,
                                         const std::vector<int64_t>& axes, bool keepdims) {
    std::set<int64_t> reduced(axes.begin(), axes.end());
    std::vector<int64_t> output;
    for (int64_t axis = 0; axis < static_cast<int64_t>(input_dims.size()); ++axis) {
        if (reduced.count(axis) != 0) {
            if (keepdims) output.push_back(1);
        } else {
            output.push_back(input_dims[static_cast<size_t>(axis)]);
        }
    }
    if (output.empty()) output.push_back(1);
    return output;
}

template <DataType dtype, bool mean>
int32_t Reduce(operators::ReduceMeanParam* mean_param) {
    if constexpr (!mean) return -1;
    (void)mean_param;
    return -1;
}

template <DataType dtype>
int32_t ReduceMean(operators::ReduceMeanParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get())) return -1;
    const auto input_dims = param->input->dims().data();
    std::vector<int64_t> axes;
    if (!NormalizeReductionAxes(param->axes, static_cast<int>(input_dims.size()), &axes)) return -1;
    const auto output_dims = ReducedShape(input_dims, axes, param->keepdims);
    if (!PrepareOutput<dtype>(param->out.get(), &output_dims)) return -1;
    std::vector<float> sums(static_cast<size_t>(param->out->numel()), 0.0f);
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    const std::set<int64_t> reduced(axes.begin(), axes.end());
    std::vector<int64_t> input_coords(input_dims.size(), 0);
    std::vector<int64_t> output_coords;
    for (int64_t linear = 0; linear < param->input->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            input_coords[axis] = remaining / input_strides[axis];
            remaining %= input_strides[axis];
        }
        output_coords.clear();
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            if (reduced.count(static_cast<int64_t>(axis)) != 0) {
                if (param->keepdims) output_coords.push_back(0);
            } else {
                output_coords.push_back(input_coords[axis]);
            }
        }
        if (output_coords.empty()) output_coords.push_back(0);
        int64_t output_offset = 0;
        for (size_t axis = 0; axis < output_coords.size(); ++axis) output_offset += output_coords[axis] * output_strides[axis];
        sums[static_cast<size_t>(output_offset)] += TensorIO<dtype>::Read(param->input.get(), linear);
    }
    int64_t reduction_count = 0;
    if (!SafeProduct(input_dims, 0, input_dims.size(), &reduction_count)) return -1;
    int64_t unreduced_count = 0;
    if (!SafeProduct(output_dims, 0, output_dims.size(), &unreduced_count) || unreduced_count <= 0 ||
        reduction_count <= 0 || reduction_count % unreduced_count != 0) return -1;
    reduction_count = reduction_count / unreduced_count;
    if (reduction_count <= 0) return -1;
    for (int64_t i = 0; i < param->out->numel(); ++i) TensorIO<dtype>::Write(param->out.get(), i, sums[static_cast<size_t>(i)] / reduction_count);
    return 0;
}

template <DataType dtype>
int32_t ReduceSum(operators::ReduceSumParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get())) return -1;
    const auto input_dims = param->input->dims().data();
    std::vector<int64_t> axes;
    if (!NormalizeReductionAxes(param->axes, static_cast<int>(input_dims.size()), &axes)) return -1;
    const auto output_dims = ReducedShape(input_dims, axes, param->keepdims);
    if (!PrepareOutput<dtype>(param->out.get(), &output_dims)) return -1;
    std::vector<float> sums(static_cast<size_t>(param->out->numel()), 0.0f);
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    const std::set<int64_t> reduced(axes.begin(), axes.end());
    std::vector<int64_t> input_coords(input_dims.size(), 0);
    std::vector<int64_t> output_coords;
    for (int64_t linear = 0; linear < param->input->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            input_coords[axis] = remaining / input_strides[axis];
            remaining %= input_strides[axis];
        }
        output_coords.clear();
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            if (reduced.count(static_cast<int64_t>(axis)) != 0) {
                if (param->keepdims) output_coords.push_back(0);
            } else {
                output_coords.push_back(input_coords[axis]);
            }
        }
        if (output_coords.empty()) output_coords.push_back(0);
        int64_t output_offset = 0;
        for (size_t axis = 0; axis < output_coords.size(); ++axis) output_offset += output_coords[axis] * output_strides[axis];
        sums[static_cast<size_t>(output_offset)] += TensorIO<dtype>::Read(param->input.get(), linear);
    }
    for (int64_t i = 0; i < param->out->numel(); ++i) TensorIO<dtype>::Write(param->out.get(), i, sums[static_cast<size_t>(i)]);
    return 0;
}

template <DataType dtype>
int32_t Softmax(operators::SoftmaxParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !HasValidShape(param->input->dims().data(), false) ||
        !PrepareOutput<dtype>(param->out.get(), &param->input->dims().data())) return -1;
    const auto dims = param->input->dims().data();
    const int rank = static_cast<int>(dims.size());
    int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) return -1;
    int64_t outer = 1;
    int64_t inner = 1;
    if (!SafeProduct(dims, 0, static_cast<size_t>(axis), &outer) ||
        !SafeProduct(dims, static_cast<size_t>(axis + 1), dims.size(), &inner)) return -1;
    const int64_t axis_dim = dims[static_cast<size_t>(axis)];
    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
            const int64_t base = outer_idx * axis_dim * inner + inner_idx;
            float max_value = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < axis_dim; ++i) max_value = std::max(max_value, TensorIO<dtype>::Read(param->input.get(), base + i * inner));
            float sum = 0.0f;
            for (int64_t i = 0; i < axis_dim; ++i) sum += std::exp(TensorIO<dtype>::Read(param->input.get(), base + i * inner) - max_value);
            if (!(sum > 0.0f)) return -1;
            for (int64_t i = 0; i < axis_dim; ++i) TensorIO<dtype>::Write(param->out.get(), base + i * inner, std::exp(TensorIO<dtype>::Read(param->input.get(), base + i * inner) - max_value) / sum);
        }
    }
    return 0;
}

template <DataType dtype, bool is_max>
int32_t Pool(operators::PoolParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || param->out == nullptr ||
        (param->input->dims().size() != 2 && param->input->dims().size() != 4) ||
        param->out->dims().size() != param->input->dims().size() || param->kernel_h <= 0 || param->kernel_w <= 0 ||
        param->stride_h <= 0 || param->stride_w <= 0 || param->pad_h < 0 || param->pad_w < 0) return -1;
    const bool is_4d = param->input->dims().size() == 4;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D input_shape{}, output_shape{};
    if (is_4d && (NormalizeDataLayout(param->out->layout()) != layout ||
                  !DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
                  !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape))) return -1;
    const int64_t batch = is_4d ? input_shape.n : 1;
    const int64_t channels = is_4d ? input_shape.c : 1;
    const int64_t in_h = is_4d ? input_shape.h : param->input->dims()[0];
    const int64_t in_w = is_4d ? input_shape.w : param->input->dims()[1];
    const int64_t out_h = is_4d ? output_shape.h : param->out->dims()[0];
    const int64_t out_w = is_4d ? output_shape.w : param->out->dims()[1];
    if (batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 ||
        in_h > (std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_h)) ||
        in_w > (std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_w))) return -1;
    int64_t expected_h = 0;
    int64_t expected_w = 0;
    if (!TryWindowOutputDimension(in_h, param->kernel_h, param->stride_h, param->pad_h, 1, &expected_h) ||
        !TryWindowOutputDimension(in_w, param->kernel_w, param->stride_w, param->pad_w, 1, &expected_w) ||
        expected_h != out_h || expected_w != out_w) return -1;
    const std::vector<int64_t> expected_dims = is_4d
        ? EncodeImageShape4D(ImageShape4D{batch, channels, out_h, out_w}, layout)
        : std::vector<int64_t>{out_h, out_w};
    if (!PrepareOutput<dtype>(param->out.get(), &expected_dims)) return -1;
    for (int64_t n = 0; n < batch; ++n) for (int64_t c = 0; c < channels; ++c)
        for (int64_t oh = 0; oh < out_h; ++oh) for (int64_t ow = 0; ow < out_w; ++ow) {
            float value = is_max ? -std::numeric_limits<float>::infinity() : 0.0f;
            int count = 0;
            for (int kh = 0; kh < param->kernel_h; ++kh) for (int kw = 0; kw < param->kernel_w; ++kw) {
                const int64_t ih = oh * param->stride_h + kh - param->pad_h;
                const int64_t iw = ow * param->stride_w + kw - param->pad_w;
                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                const int64_t input_offset = is_4d ? OffsetForImage4D(layout, n, c, ih, iw, channels, in_h, in_w) : ih * in_w + iw;
                const float x = TensorIO<dtype>::Read(param->input.get(), input_offset);
                if (is_max) value = std::max(value, x); else { value += x; ++count; }
            }
            if (!is_max) value = count == 0 ? 0.0f : value / count;
            const int64_t output_offset = is_4d ? OffsetForImage4D(layout, n, c, oh, ow, channels, out_h, out_w) : oh * out_w + ow;
            TensorIO<dtype>::Write(param->out.get(), output_offset, value);
        }
    return 0;
}

template <DataType dtype>
int32_t GlobalAveragePool(operators::GlobalAveragePoolParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || param->out == nullptr ||
        !HasValidShape(param->input->dims().data(), false) || param->input->dims().size() != 4 ||
        param->out->dims().size() != 4) return -1;
    ImageShape4D shape{};
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (NormalizeDataLayout(param->out->layout()) != layout ||
        !DecodeImageShape4D(param->input->dims().data(), layout, &shape)) return -1;
    const std::vector<int64_t> expected_dims = EncodeImageShape4D(ImageShape4D{shape.n, shape.c, 1, 1}, layout);
    if (!PrepareOutput<dtype>(param->out.get(), &expected_dims)) return -1;
    for (int64_t n = 0; n < shape.n; ++n) for (int64_t c = 0; c < shape.c; ++c) {
        float sum = 0.0f;
        for (int64_t h = 0; h < shape.h; ++h) for (int64_t w = 0; w < shape.w; ++w)
            sum += TensorIO<dtype>::Read(param->input.get(), OffsetForImage4D(layout, n, c, h, w, shape.c, shape.h, shape.w));
        TensorIO<dtype>::Write(param->out.get(), OffsetForImage4D(layout, n, c, 0, 0, shape.c, 1, 1), sum / static_cast<float>(shape.h * shape.w));
    }
    return 0;
}

template <DataType dtype>
int32_t Gather(operators::GatherParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->data.get()) || !IsIndexTensor(param->indices.get())) return -1;
    const auto data_dims = param->data->dims().data();
    const auto indices_dims = param->indices->dims().data();
    const int rank = static_cast<int>(data_dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (rank <= 0 || axis < 0 || axis >= rank) return -1;
    std::vector<int64_t> expected_output_dims;
    expected_output_dims.reserve(data_dims.size() - 1 + indices_dims.size());
    expected_output_dims.insert(expected_output_dims.end(), data_dims.begin(), data_dims.begin() + axis);
    expected_output_dims.insert(expected_output_dims.end(), indices_dims.begin(), indices_dims.end());
    expected_output_dims.insert(expected_output_dims.end(), data_dims.begin() + axis + 1, data_dims.end());
    if (!PrepareOutput<dtype>(param->out.get(), &expected_output_dims)) return -1;
    const auto output_dims = param->out->dims().data();
    const auto data_strides = Strides(data_dims);
    const auto indices_strides = Strides(indices_dims);
    const auto output_strides = Strides(output_dims);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        std::vector<int64_t> coords(output_dims.size(), 0);
        for (size_t i = 0; i < output_dims.size(); ++i) { coords[i] = remaining / output_strides[i]; remaining %= output_strides[i]; }
        int64_t index_offset = 0;
        for (size_t i = 0; i < indices_dims.size(); ++i) index_offset += coords[static_cast<size_t>(axis) + i] * indices_strides[i];
        int64_t index = common_tensor_detail::ReadIndex(param->indices.get(), index_offset);
        if (index < 0) index += data_dims[static_cast<size_t>(axis)];
        if (index < 0 || index >= data_dims[static_cast<size_t>(axis)]) return -1;
        int64_t data_offset = 0;
        for (int i = 0; i < axis; ++i) data_offset += coords[static_cast<size_t>(i)] * data_strides[static_cast<size_t>(i)];
        data_offset += index * data_strides[static_cast<size_t>(axis)];
        for (int i = axis + 1; i < rank; ++i) data_offset += coords[static_cast<size_t>(i - 1 + indices_dims.size())] * data_strides[static_cast<size_t>(i)];
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->data.get(), data_offset));
    }
    return 0;
}

template <DataType dtype>
int32_t Expand(operators::ExpandParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !IsIndexTensor(param->shape.get())) return -1;
    std::vector<int64_t> target;
    for (int64_t i = 0; i < param->shape->numel(); ++i) target.push_back(common_tensor_detail::ReadInteger(param->shape.get(), i));
    const auto input_dims = param->input->dims().data();
    if (target.size() < input_dims.size() || !HasValidShape(target)) return -1;
    if (!PrepareOutput<dtype>(param->out.get(), &target)) return -1;
    const auto output_dims = param->out->dims().data();
    const size_t gap = target.size() - input_dims.size();
    for (size_t i = 0; i < target.size(); ++i) {
        const int64_t input_dim = i < gap ? 1 : input_dims[i - gap];
        if (target[i] <= 0 || (input_dim != target[i] && input_dim != 1)) return -1;
    }
    const auto output_strides = Strides(output_dims);
    const auto input_strides = Strides(input_dims);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        std::vector<int64_t> coords(output_dims.size(), 0);
        for (size_t i = 0; i < output_dims.size(); ++i) { coords[i] = remaining / output_strides[i]; remaining %= output_strides[i]; }
        const int64_t input_offset = BroadcastOffset(coords, input_dims, input_strides);
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <DataType dtype>
int32_t Where(operators::WhereParam* param) {
    if (param == nullptr || !IsBoolTensor(param->condition.get()) ||
        !IsTensor<dtype>(param->x.get()) || !IsTensor<dtype>(param->y.get())) return -1;
    std::vector<int64_t> output_dims;
    std::vector<int64_t> xy_dims;
    if (!InferBroadcast(param->condition->dims().data(), param->x->dims().data(), &xy_dims) ||
        !InferBroadcast(xy_dims, param->y->dims().data(), &output_dims) ||
        !PrepareOutput<dtype>(param->out.get(), &output_dims)) return -1;
    const auto output_strides = Strides(output_dims);
    const auto condition_strides = Strides(param->condition->dims().data());
    const auto x_strides = Strides(param->x->dims().data());
    const auto y_strides = Strides(param->y->dims().data());
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        std::vector<int64_t> coords(output_dims.size(), 0);
        for (size_t i = 0; i < output_dims.size(); ++i) { coords[i] = remaining / output_strides[i]; remaining %= output_strides[i]; }
        const bool condition = common_tensor_detail::ReadBool(param->condition.get(), BroadcastOffset(coords, param->condition->dims().data(), condition_strides));
        const Tensor* source = condition ? param->x.get() : param->y.get();
        const auto& dims = condition ? param->x->dims().data() : param->y->dims().data();
        const auto& strides = condition ? x_strides : y_strides;
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(source, BroadcastOffset(coords, dims, strides)));
    }
    return 0;
}

template <DataType dtype>
int32_t Equal(operators::EqualParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->lhs.get()) || !IsTensor<dtype>(param->rhs.get()) ||
        !IsInitialized(param->out.get())) return -1;
    std::vector<int64_t> output_dims;
    int64_t output_numel = 0;
    size_t output_bytes = 0;
    if (!InferBroadcast(param->lhs->dims().data(), param->rhs->dims().data(), &output_dims) ||
        param->out->dims().data() != output_dims || !SafeTensorNumel(param->out.get(), &output_numel) ||
        !SafeRequiredBytes(output_numel, sizeof(uint8_t), &output_bytes) || param->out->memory_size() < output_bytes) return -1;
    param->out->set_data_type(DataType::BOOL);
    auto* output = static_cast<uint8_t*>(param->out->raw_data());
    const auto output_strides = Strides(output_dims);
    const auto lhs_strides = Strides(param->lhs->dims().data());
    const auto rhs_strides = Strides(param->rhs->dims().data());
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        std::vector<int64_t> coords(output_dims.size(), 0);
        for (size_t i = 0; i < output_dims.size(); ++i) { coords[i] = remaining / output_strides[i]; remaining %= output_strides[i]; }
        const float lhs = TensorIO<dtype>::Read(param->lhs.get(), BroadcastOffset(coords, param->lhs->dims().data(), lhs_strides));
        const float rhs = TensorIO<dtype>::Read(param->rhs.get(), BroadcastOffset(coords, param->rhs->dims().data(), rhs_strides));
        output[linear] = lhs == rhs ? 1 : 0;
    }
    return 0;
}

template <DataType dtype>
int32_t BatchNormalization(operators::BatchNormParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !IsTensor<dtype>(param->scale.get()) ||
        !IsTensor<dtype>(param->bias.get()) || !IsTensor<dtype>(param->mean.get()) || !IsTensor<dtype>(param->var.get()) ||
        param->out == nullptr || !std::isfinite(param->epsilon) || param->epsilon < 0.0f) return -1;
    ImageShape4D shape{};
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (param->input->dims().size() != 4 || param->out->dims().size() != 4 ||
        !HasValidShape(param->input->dims().data(), false) ||
        NormalizeDataLayout(param->out->layout()) != layout ||
        !DecodeImageShape4D(param->input->dims().data(), layout, &shape) ||
        param->out->dims().data() != param->input->dims().data() ||
        param->scale->dims().data() != std::vector<int64_t>{shape.c} ||
        param->bias->dims().data() != std::vector<int64_t>{shape.c} ||
        param->mean->dims().data() != std::vector<int64_t>{shape.c} ||
        param->var->dims().data() != std::vector<int64_t>{shape.c}) return -1;
    const std::vector<int64_t> input_dims = param->input->dims().data();
    if (!PrepareOutput<dtype>(param->out.get(), &input_dims)) return -1;
    for (int64_t n = 0; n < shape.n; ++n) for (int64_t c = 0; c < shape.c; ++c) {
        const float scale = TensorIO<dtype>::Read(param->scale.get(), c);
        const float bias = TensorIO<dtype>::Read(param->bias.get(), c);
        const float mean = TensorIO<dtype>::Read(param->mean.get(), c);
        const float variance = TensorIO<dtype>::Read(param->var.get(), c);
        const float denominator = variance + param->epsilon;
        if (!std::isfinite(scale) || !std::isfinite(bias) || !std::isfinite(mean) || !std::isfinite(variance) ||
            !std::isfinite(denominator) || denominator <= 0.0f) return -1;
        const float inv_std = 1.0f / std::sqrt(denominator);
        for (int64_t h = 0; h < shape.h; ++h) for (int64_t w = 0; w < shape.w; ++w) {
            const int64_t offset = OffsetForImage4D(layout, n, c, h, w, shape.c, shape.h, shape.w);
            TensorIO<dtype>::Write(param->out.get(), offset, (TensorIO<dtype>::Read(param->input.get(), offset) - mean) * inv_std * scale + bias);
        }
    }
    return 0;
}

template <DataType dtype>
int32_t Conv2D(operators::Conv2dParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !IsTensor<dtype>(param->w.get()) ||
        param->out == nullptr || param->stride_h <= 0 || param->stride_w <= 0 || param->pad_h < 0 || param->pad_w < 0 ||
        param->dilation_h <= 0 || param->dilation_w <= 0 || param->group <= 0) return -1;
    if (param->input->dims().size() == 2 && param->w->dims().size() == 2) {
        const auto in_dims = param->input->dims().data();
        const auto out_dims = param->out->dims().data();
        if (out_dims.size() != 2 || !HasValidShape(in_dims, false) || param->w->dims()[0] <= 0 || param->w->dims()[1] <= 0 ||
            in_dims[0] > std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_h) ||
            in_dims[1] > std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_w)) return -1;
        int64_t expected_h = 0;
        int64_t expected_w = 0;
        if (!TryWindowOutputDimension(in_dims[0], param->w->dims()[0], param->stride_h, param->pad_h,
                                      param->dilation_h, &expected_h) ||
            !TryWindowOutputDimension(in_dims[1], param->w->dims()[1], param->stride_w, param->pad_w,
                                      param->dilation_w, &expected_w) ||
            out_dims[0] != expected_h || out_dims[1] != expected_w) return -1;
        if (param->bias != nullptr && !IsTensor<dtype>(param->bias.get())) return -1;
        if (param->bias != nullptr && param->bias->dims().data() != out_dims) return -1;
        const std::vector<int64_t> expected_dims{expected_h, expected_w};
        if (!PrepareOutput<dtype>(param->out.get(), &expected_dims)) return -1;
        for (int64_t oh = 0; oh < out_dims[0]; ++oh) for (int64_t ow = 0; ow < out_dims[1]; ++ow) {
            float sum = 0.0f;
            for (int64_t kh = 0; kh < param->w->dims()[0]; ++kh) for (int64_t kw = 0; kw < param->w->dims()[1]; ++kw) {
                const int64_t ih = oh * param->stride_h + kh * param->dilation_h - param->pad_h;
                const int64_t iw = ow * param->stride_w + kw * param->dilation_w - param->pad_w;
                if (ih >= 0 && ih < in_dims[0] && iw >= 0 && iw < in_dims[1]) sum += TensorIO<dtype>::Read(param->input.get(), ih * in_dims[1] + iw) * TensorIO<dtype>::Read(param->w.get(), kh * param->w->dims()[1] + kw);
            }
            if (param->bias != nullptr && IsTensor<dtype>(param->bias.get())) sum += TensorIO<dtype>::Read(param->bias.get(), oh * out_dims[1] + ow);
            TensorIO<dtype>::Write(param->out.get(), oh * out_dims[1] + ow, sum);
        }
        return 0;
    }
    ImageShape4D input_shape{}, output_shape{};
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    if (param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4 ||
        NormalizeDataLayout(param->out->layout()) != layout || !HasValidShape(param->input->dims().data(), false) ||
        !HasValidShape(param->w->dims().data(), false) ||
        !DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), layout, &output_shape)) return -1;
    const int64_t out_channels = param->w->dims()[0];
    const int64_t kernel_channels = param->w->dims()[1];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    if (out_channels <= 0 || kernel_channels <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        input_shape.h > std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_h) ||
        input_shape.w > std::numeric_limits<int64_t>::max() - 2 * static_cast<int64_t>(param->pad_w)) return -1;
    const int64_t group = param->group;
    if (group <= 0 || group > input_shape.c || group > out_channels || input_shape.c % group != 0 ||
        out_channels % group != 0 || kernel_channels != input_shape.c / group) return -1;
    int64_t expected_h = 0;
    int64_t expected_w = 0;
    if (!TryWindowOutputDimension(input_shape.h, kernel_h, param->stride_h, param->pad_h, param->dilation_h, &expected_h) ||
        !TryWindowOutputDimension(input_shape.w, kernel_w, param->stride_w, param->pad_w, param->dilation_w, &expected_w) ||
        output_shape.n != input_shape.n || output_shape.c != out_channels ||
        output_shape.h != expected_h || output_shape.w != expected_w) return -1;
    if (param->bias != nullptr && !IsTensor<dtype>(param->bias.get())) return -1;
    if (param->bias != nullptr && param->bias->numel() != out_channels) return -1;
    const std::vector<int64_t> expected_dims = EncodeImageShape4D(
        ImageShape4D{input_shape.n, out_channels, expected_h, expected_w}, layout);
    if (!PrepareOutput<dtype>(param->out.get(), &expected_dims)) return -1;
    for (int64_t n = 0; n < input_shape.n; ++n) for (int64_t g = 0; g < group; ++g) for (int64_t oc = 0; oc < out_channels / group; ++oc) {
        const int64_t global_oc = g * (out_channels / group) + oc;
        for (int64_t oh = 0; oh < output_shape.h; ++oh) for (int64_t ow = 0; ow < output_shape.w; ++ow) {
            float sum = 0.0f;
            for (int64_t ic = 0; ic < kernel_channels; ++ic) for (int64_t kh = 0; kh < kernel_h; ++kh) for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t ih = oh * param->stride_h + kh * param->dilation_h - param->pad_h;
                const int64_t iw = ow * param->stride_w + kw * param->dilation_w - param->pad_w;
                if (ih < 0 || ih >= input_shape.h || iw < 0 || iw >= input_shape.w) continue;
                const int64_t input_offset = OffsetForImage4D(layout, n, g * kernel_channels + ic, ih, iw, input_shape.c, input_shape.h, input_shape.w);
                const int64_t weight_offset = ((global_oc * kernel_channels + ic) * kernel_h + kh) * kernel_w + kw;
                sum += TensorIO<dtype>::Read(param->input.get(), input_offset) * TensorIO<dtype>::Read(param->w.get(), weight_offset);
            }
            if (param->bias != nullptr && IsTensor<dtype>(param->bias.get()) && param->bias->numel() == out_channels) sum += TensorIO<dtype>::Read(param->bias.get(), global_oc);
            const int64_t output_offset = OffsetForImage4D(param->out->layout(), n, global_oc, oh, ow, out_channels, output_shape.h, output_shape.w);
            TensorIO<dtype>::Write(param->out.get(), output_offset, sum);
        }
    }
    return 0;
}

template <DataType dtype>
int32_t Resize(operators::ResizeParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->input.get()) || !PrepareOutput<dtype>(param->out.get()) ||
        param->input->dims().size() != param->out->dims().size() || param->scales.size() != param->input->dims().size()) return -1;
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (!HasValidShape(input_dims) || !HasValidShape(output_dims)) return -1;
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        int64_t expected_dim = 0;
        if (!TryScaledDimension(input_dims[axis], param->scales[axis], &expected_dim) ||
            output_dims[axis] != expected_dim) {
            return -1;
        }
    }
    const auto input_strides = Strides(input_dims);
    const auto output_strides = Strides(output_dims);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear, input_offset = 0;
        for (size_t axis = 0; axis < output_dims.size(); ++axis) {
            const int64_t coordinate = remaining / output_strides[axis];
            remaining %= output_strides[axis];
            const int64_t input_coordinate = ResizeCoordinate(coordinate, param->scales[axis], input_dims[axis]);
            input_offset += input_coordinate * input_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <DataType dtype>
int32_t ResizeConcat(operators::ResizeConcatParam* param) {
    if (param == nullptr || !IsTensor<dtype>(param->resize_input.get()) || !IsTensor<dtype>(param->concat_input.get()) ||
        !PrepareOutput<dtype>(param->out.get()) || param->resize_input->dims().size() != 4 || param->concat_input->dims().size() != 4 ||
        param->out->dims().size() != 4 || param->scales.size() != 4 || param->resize_input_index < 0 ||
        param->resize_input_index > 1) return -1;
    const DataLayout resize_layout = NormalizeDataLayout(param->resize_input->layout());
    const DataLayout concat_layout = NormalizeDataLayout(param->concat_input->layout());
    const DataLayout output_layout = NormalizeDataLayout(param->out->layout());
    if (resize_layout != concat_layout || resize_layout != output_layout) return -1;
    const int axis = param->axis < 0 ? param->axis + 4 : param->axis;
    if (axis != ChannelAxisForLayout(output_layout)) return -1;
    for (const float scale : param->scales) if (!IsValidScale(scale)) return -1;
    if (param->scales[0] != 1.0f || param->scales[static_cast<size_t>(axis)] != 1.0f) return -1;
    ImageShape4D resize_shape{}, concat_shape{}, output_shape{};
    if (!DecodeImageShape4D(param->resize_input->dims().data(), resize_layout, &resize_shape) ||
        !DecodeImageShape4D(param->concat_input->dims().data(), concat_layout, &concat_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), output_layout, &output_shape) ||
        resize_shape.n != concat_shape.n || output_shape.n != resize_shape.n ||
        output_shape.c != resize_shape.c + concat_shape.c) return -1;
    const auto resize_dims = param->resize_input->dims().data();
    const auto concat_dims = param->concat_input->dims().data();
    const auto output_dims = param->out->dims().data();
    std::vector<int64_t> expected_resize_dims(4);
    for (size_t i = 0; i < 4; ++i) {
        if (!TryScaledDimension(resize_dims[i], param->scales[i], &expected_resize_dims[i])) return -1;
        if (static_cast<int>(i) != axis && expected_resize_dims[i] != concat_dims[i]) return -1;
        if (static_cast<int>(i) != axis && output_dims[i] != expected_resize_dims[i]) return -1;
    }
    if (expected_resize_dims[static_cast<size_t>(axis)] + concat_dims[static_cast<size_t>(axis)] !=
            output_dims[static_cast<size_t>(axis)] ||
        output_dims[static_cast<size_t>(axis)] <= 0) return -1;
    const bool channel_last = IsChannelLastLayout(output_layout);
    const float scale_h = param->scales[channel_last ? 1 : 2];
    const float scale_w = param->scales[channel_last ? 2 : 3];
    const DataLayout out_layout = output_layout;
    for (int64_t n = 0; n < output_shape.n; ++n) for (int64_t oc = 0; oc < output_shape.c; ++oc)
        for (int64_t oh = 0; oh < output_shape.h; ++oh) for (int64_t ow = 0; ow < output_shape.w; ++ow) {
            const bool use_resize = param->resize_input_index == 0 ? oc < resize_shape.c : oc >= concat_shape.c;
            float value = 0.0f;
            if (use_resize) {
                const int64_t rc = param->resize_input_index == 0 ? oc : oc - concat_shape.c;
                const int64_t ih = ResizeCoordinate(oh, scale_h, resize_shape.h);
                const int64_t iw = ResizeCoordinate(ow, scale_w, resize_shape.w);
                value = TensorIO<dtype>::Read(param->resize_input.get(), OffsetForImage4D(param->resize_input->layout(), n, rc, ih, iw, resize_shape.c, resize_shape.h, resize_shape.w));
            } else {
                const int64_t cc = param->resize_input_index == 0 ? oc - resize_shape.c : oc;
                value = TensorIO<dtype>::Read(param->concat_input.get(), OffsetForImage4D(param->concat_input->layout(), n, cc, oh, ow, concat_shape.c, output_shape.h, output_shape.w));
            }
            TensorIO<dtype>::Write(param->out.get(), OffsetForImage4D(out_layout, n, oc, oh, ow, output_shape.c, output_shape.h, output_shape.w), value);
        }
    return 0;
}

}  // namespace fp8_host
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_FP8_HOST_H
