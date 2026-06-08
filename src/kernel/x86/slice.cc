#include "src/kernel/slice.h"

#include <immintrin.h>

#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"
#include "util/types.h"

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

template <typename T>
void CopyRawBlock(const T* src, T* dst, int64_t count) {
    std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(T));
}

template <>
void CopyRawBlock<float>(const float* src, float* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 vec = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, vec);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(float));
    }
}

template <>
void CopyRawBlock<uint16_t>(const uint16_t* src, uint16_t* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vec);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(uint16_t));
    }
}

template <DataType dtype>
int32_t ComputeSliceFallback(feather::operators::SliceParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const int32_t rank = static_cast<int32_t>(in_dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }

    const int64_t dim = in_dims[axis];
    int64_t start = param->start < 0 ? param->start + dim : param->start;
    int64_t end = param->end < 0 ? param->end + dim : param->end;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);

    param->out->set_data_type(dtype);
    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);

    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t i = 0; i < out_dims.size(); ++i) {
            out_coords[i] = remaining / out_strides[i];
            remaining %= out_strides[i];
        }
        for (size_t i = 0; i < in_dims.size(); ++i) {
            in_coords[i] = out_coords[i];
        }
        in_coords[axis] += start;

        int64_t input_offset = 0;
        for (size_t i = 0; i < in_dims.size(); ++i) {
            input_offset += in_coords[i] * in_strides[i];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <typename T>
int32_t ComputeSliceRaw(feather::operators::SliceParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const int32_t rank = static_cast<int32_t>(in_dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }

    const int64_t dim = in_dims[axis];
    int64_t start = param->start < 0 ? param->start + dim : param->start;
    int64_t end = param->end < 0 ? param->end + dim : param->end;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);
    if (end < start) {
        return -1;
    }

    int64_t outer = 1;
    int64_t inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        outer *= in_dims[i];
    }
    for (int32_t i = axis + 1; i < rank; ++i) {
        inner *= in_dims[i];
    }

    const int64_t slice_axis = end - start;
    const int64_t copy_count = slice_axis * inner;
    const int64_t input_axis = in_dims[axis];
    const T* input = param->input->data<T>();
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        const int64_t input_base = (outer_idx * input_axis + start) * inner;
        const int64_t output_base = outer_idx * copy_count;
        CopyRawBlock(input + input_base, output + output_base, copy_count);
    }
    return 0;
}

}  // namespace

template <>
int32_t SliceKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Slice::FP32");
    auto* param = static_cast<feather::operators::SliceParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeSliceFallback<DataType::FP32>(param);
    }
    return ComputeSliceRaw<float>(param, DataType::FP32);
}

template <>
int32_t SliceKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Slice::FP16");
    auto* param = static_cast<feather::operators::SliceParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeSliceFallback<DataType::FP16>(param);
    }
    return ComputeSliceRaw<uint16_t>(param, DataType::FP16);
}

void EnsureX86SliceKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Slice",
            []() { return std::make_unique<SliceKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Slice",
            []() { return std::make_unique<SliceKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
