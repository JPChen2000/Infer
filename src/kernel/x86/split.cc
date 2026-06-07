#include "src/kernel/split.h"

#include <immintrin.h>

#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

int64_t ComputeProduct(const std::vector<int64_t>& dims, size_t begin, size_t end) {
    int64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        product *= dims[i];
    }
    return product;
}

inline void CopyFloatBlock(const float* src, float* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 value = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, value);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(float));
    }
}

inline void CopyHalfBlock(const uint16_t* src, uint16_t* dst, int64_t count) {
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), value);
    }
    if (i < count) {
        std::memcpy(dst + i, src + i, static_cast<size_t>(count - i) * sizeof(uint16_t));
    }
}

template <DataType dtype>
int32_t ComputeSplitFallback(feather::operators::SplitParam* param) {
    if (param == nullptr || param->input == nullptr || param->outputs.empty()) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(in_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(in_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(in_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(in_dims, static_cast<size_t>(axis) + 1, in_dims.size());
    const int64_t input_axis = in_dims[axis];

    for (auto& output : param->outputs) {
        if (output == nullptr) {
            return -1;
        }
        output->set_data_type(dtype);
    }

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (size_t i = 0; i < param->outputs.size(); ++i) {
            const int64_t output_axis = param->outputs[i]->dims()[axis];
            const int64_t copy_count = output_axis * inner;
            const int64_t input_base = (outer_idx * input_axis + axis_offset) * inner;
            const int64_t output_base = outer_idx * copy_count;
            for (int64_t j = 0; j < copy_count; ++j) {
                TensorIO<dtype>::Write(param->outputs[i].get(), output_base + j,
                                       TensorIO<dtype>::Read(param->input.get(), input_base + j));
            }
            axis_offset += output_axis;
        }
    }
    return 0;
}

template <typename T>
int32_t ComputeSplitRaw(feather::operators::SplitParam* param, DataType dtype,
                        void (*copy_fn)(const T*, T*, int64_t)) {
    if (param == nullptr || param->input == nullptr || param->outputs.empty()) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(in_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(in_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(in_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(in_dims, static_cast<size_t>(axis) + 1, in_dims.size());
    const int64_t input_axis = in_dims[axis];
    const T* input = param->input->data<T>();

    for (auto& output : param->outputs) {
        if (output == nullptr) {
            return -1;
        }
        output->set_data_type(dtype);
        output->mutable_data<T>();
    }

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (size_t i = 0; i < param->outputs.size(); ++i) {
            const int64_t output_axis = param->outputs[i]->dims()[axis];
            const int64_t copy_count = output_axis * inner;
            const int64_t input_base = (outer_idx * input_axis + axis_offset) * inner;
            const int64_t output_base = outer_idx * copy_count;
            copy_fn(input + input_base, param->outputs[i]->mutable_data<T>() + output_base, copy_count);
            axis_offset += output_axis;
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t SplitKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Split::FP32");
    auto* param = static_cast<feather::operators::SplitParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeSplitFallback<DataType::FP32>(param);
    }
    return ComputeSplitRaw<float>(param, DataType::FP32, CopyFloatBlock);
}

template <>
int32_t SplitKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Split::FP16");
    auto* param = static_cast<feather::operators::SplitParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeSplitFallback<DataType::FP16>(param);
    }
    return ComputeSplitRaw<uint16_t>(param, DataType::FP16, CopyHalfBlock);
}

void EnsureX86SplitKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Split",
            []() { return std::make_unique<SplitKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Split",
            []() { return std::make_unique<SplitKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
