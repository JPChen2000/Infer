#include "src/kernel/concat.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "util/bf16.h"
#include "util/thread_pool_nv.h"
#include "util/threading.h"
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

size_t GetConcatThreadCount(int64_t total_work_items) {
    return ThreadCountForWorkItems(total_work_items);
}

ThreadPoolNv& GetConcatThreadPool() {
    static ThreadPoolNv pool(DefaultThreadCount());
    return pool;
}

template <typename Fn>
void ParallelForConcat(int64_t total_work_items, int64_t total_scalar_work_items, Fn&& fn) {
    // The runtime executes graph nodes serially; keep small copies local instead of
    // paying for a worker-pool rendezvous on every Concat node.
    if (total_work_items <= 1 || total_scalar_work_items < 262144) {
        fn(0, total_work_items);
        return;
    }

    const size_t thread_count = GetConcatThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetConcatThreadPool();
    std::vector<std::future<int>> futures;
    futures.reserve(thread_count);

    const int64_t chunk_size =
        (total_work_items + static_cast<int64_t>(thread_count) - 1) / static_cast<int64_t>(thread_count);
    for (size_t tid = 0; tid < thread_count; ++tid) {
        const int64_t begin = static_cast<int64_t>(tid) * chunk_size;
        const int64_t end = std::min(total_work_items, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.emplace_back(pool.enqueue([begin, end, &fn](int) {
            fn(begin, end);
            return 0;
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

template <DataType dtype>
int32_t ComputeConcatFallback(feather::operators::ConcatParam* param) {
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) {
        return -1;
    }

    const auto& out_dims = param->out->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(out_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(out_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t out_axis = out_dims[axis];

    param->out->set_data_type(dtype);
    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (const auto& input : param->inputs) {
            if (input == nullptr) {
                return -1;
            }
            const int64_t input_axis = input->dims()[axis];
            const int64_t copy_count = input_axis * inner;
            const int64_t input_base = outer_idx * copy_count;
            const int64_t output_base = (outer_idx * out_axis + axis_offset) * inner;
            for (int64_t i = 0; i < copy_count; ++i) {
                TensorIO<dtype>::Write(param->out.get(), output_base + i, TensorIO<dtype>::Read(input.get(), input_base + i));
            }
            axis_offset += input_axis;
        }
    }
    return 0;
}

template <typename T>
bool AllInputsMatchType(const feather::operators::ConcatParam* param, DataType dtype) {
    if (param == nullptr || param->out == nullptr) {
        return false;
    }
    for (const auto& input : param->inputs) {
        if (input == nullptr || input->data_type() != dtype) {
            return false;
        }
    }
    return true;
}

template <typename T>
int32_t ComputeConcatRaw(feather::operators::ConcatParam* param, DataType dtype,
                         void (*copy_fn)(const T*, T*, int64_t)) {
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) {
        return -1;
    }

    const auto& out_dims = param->out->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(out_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(out_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t out_axis = out_dims[axis];

    param->out->set_data_type(dtype);
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);
    ParallelForConcat(outer, outer * out_axis * inner, [&](int64_t begin, int64_t end) {
        for (int64_t outer_idx = begin; outer_idx < end; ++outer_idx) {
            int64_t axis_offset = 0;
            for (const auto& input : param->inputs) {
                const int64_t input_axis = input->dims()[axis];
                const int64_t copy_count = input_axis * inner;
                const int64_t input_base = outer_idx * copy_count;
                const int64_t output_base = (outer_idx * out_axis + axis_offset) * inner;
                copy_fn(input->data<T>() + input_base, output + output_base, copy_count);
                axis_offset += input_axis;
            }
        }
    });
    return 0;
}

}  // namespace

template <>
int32_t ConcatKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Concat::FP32");
    auto* param = static_cast<feather::operators::ConcatParam*>(param_);
    if (!AllInputsMatchType<float>(param, DataType::FP32)) {
        return ComputeConcatFallback<DataType::FP32>(param);
    }
    return ComputeConcatRaw<float>(param, DataType::FP32, CopyFloatBlock);
}

template <>
int32_t ConcatKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Concat::FP16");
    auto* param = static_cast<feather::operators::ConcatParam*>(param_);
    if (!AllInputsMatchType<uint16_t>(param, DataType::FP16)) {
        return ComputeConcatFallback<DataType::FP16>(param);
    }
    return ComputeConcatRaw<uint16_t>(param, DataType::FP16, CopyHalfBlock);
}

template <>
int32_t ConcatKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Concat::BF16");
    auto* param = static_cast<feather::operators::ConcatParam*>(param_);
    if (!AllInputsMatchType<BFloat16>(param, DataType::BF16)) {
        return ComputeConcatFallback<DataType::BF16>(param);
    }
    return ComputeConcatRaw<BFloat16>(param, DataType::BF16,
                                      [](const BFloat16* src, BFloat16* dst, int64_t count) {
                                          std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(BFloat16));
                                      });
}

template <>
int32_t ConcatKernel<DeviceType::X86, DataType::INT8>::compute() {
    AutoTimer timer("X86::Concat::INT8");
    auto* param = static_cast<feather::operators::ConcatParam*>(param_);
    if (!AllInputsMatchType<int8_t>(param, DataType::INT8) || param->out == nullptr ||
        param->out->data_type() != DataType::INT8 || param->inputs.size() < 2) return -1;

    const auto& out_dims = param->out->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(out_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(out_dims.size())) return -1;
    const auto& output_quantization = param->out->quantization();
    if (!output_quantization.enabled || output_quantization.granularity != QuantizationGranularity::kPerTensor ||
        !std::isfinite(output_quantization.scale_at(0)) || output_quantization.scale_at(0) <= 0.0f) return -1;

    const size_t input_count = param->inputs.size();
    std::vector<float> input_scales(input_count);
    std::vector<int32_t> input_zero_points(input_count);
    std::vector<bool> input_matches_output(input_count, false);
    bool same_quantization = true;
    for (size_t input_index = 0; input_index < input_count; ++input_index) {
        const auto& input = param->inputs[input_index];
        if (input == nullptr || input->dims().size() != out_dims.size()) return -1;
        for (size_t dim = 0; dim < out_dims.size(); ++dim) {
            if (static_cast<int32_t>(dim) != axis && input->dims()[dim] != out_dims[dim]) return -1;
        }
        const auto& input_quantization = input->quantization();
        if (!input_quantization.enabled || input_quantization.granularity != QuantizationGranularity::kPerTensor ||
            !std::isfinite(input_quantization.scale_at(0)) || input_quantization.scale_at(0) <= 0.0f) return -1;
        input_scales[input_index] = input_quantization.scale_at(0);
        input_zero_points[input_index] = input_quantization.zero_point_at(0);
        input_matches_output[input_index] =
            input_scales[input_index] == output_quantization.scale_at(0) &&
            input_zero_points[input_index] == output_quantization.zero_point_at(0);
        same_quantization = same_quantization && input_matches_output[input_index];
    }
    if (same_quantization) {
        return ComputeConcatRaw<int8_t>(param, DataType::INT8,
                                         [](const int8_t* src, int8_t* dst, int64_t count) {
                                             std::memcpy(dst, src, static_cast<size_t>(count));
                                         });
    }

    // Concat only changes placement. Precompute one byte-to-byte table per
    // input so differing calibration scales do not trigger floating-point
    // division and rounding for every copied element.
    std::vector<std::array<int32_t, 256>> lookup(input_count);
    const double output_scale = static_cast<double>(output_quantization.scale_at(0));
    const int32_t output_zero_point = output_quantization.zero_point_at(0);
    for (size_t input_index = 0; input_index < input_count; ++input_index) {
        for (int32_t value = -128; value <= 127; ++value) {
            const double transformed =
                (static_cast<double>(value - input_zero_points[input_index]) *
                 static_cast<double>(input_scales[input_index])) /
                    output_scale + static_cast<double>(output_zero_point);
            if (!std::isfinite(transformed)) return -1;
            const double rounded = std::round(transformed);
            lookup[input_index][static_cast<size_t>(value + 128)] =
                static_cast<int32_t>(std::max(-128.0, std::min(127.0, rounded)));
        }
    }

    const int64_t outer = ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t output_axis = out_dims[static_cast<size_t>(axis)];
    int8_t* output = param->out->mutable_data<int8_t>();
    param->out->set_data_type(DataType::INT8);
    ParallelForConcat(outer, outer * output_axis * inner, [&](int64_t begin, int64_t end) {
        for (int64_t outer_index = begin; outer_index < end; ++outer_index) {
            int64_t axis_offset = 0;
            for (size_t input_index = 0; input_index < input_count; ++input_index) {
                const auto& input = param->inputs[input_index];
                const int64_t input_axis = input->dims()[static_cast<size_t>(axis)];
                const int64_t count = input_axis * inner;
                const int64_t input_base = outer_index * count;
                const int64_t output_base = (outer_index * output_axis + axis_offset) * inner;
                const int8_t* input_data = input->data<int8_t>();
                if (input_matches_output[input_index]) {
                    std::memcpy(output + output_base, input_data + input_base, static_cast<size_t>(count));
                } else {
                    const auto& input_lookup = lookup[input_index];
                    int64_t index = 0;
                    for (; index + 8 <= count; index += 8) {
                        const __m128i input_bytes = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(input_data + input_base + index));
                        const __m256i indices = _mm256_add_epi32(
                            _mm256_cvtepi8_epi32(input_bytes), _mm256_set1_epi32(128));
                        const __m256i gathered = _mm256_i32gather_epi32(input_lookup.data(), indices, 4);
                        const __m128i low = _mm256_castsi256_si128(gathered);
                        const __m128i high = _mm256_extracti128_si256(gathered, 1);
                        const __m128i packed16 = _mm_packs_epi32(low, high);
                        const __m128i packed8 = _mm_packs_epi16(packed16, packed16);
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + output_base + index), packed8);
                    }
                    for (; index < count; ++index) {
                        output[output_base + index] = input_lookup[static_cast<size_t>(
                            static_cast<int32_t>(input_data[input_base + index]) + 128)];
                    }
                }
                axis_offset += input_axis;
            }
        }
    });
    return 0;
}

void EnsureX86ConcatKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::INT8, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::INT8>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
