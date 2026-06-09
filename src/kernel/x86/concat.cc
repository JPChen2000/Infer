#include "src/kernel/concat.h"

#include <immintrin.h>

#include <cstring>
#include <future>
#include <vector>

#include "src/kernel/common/kernel_io.h"
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
    if (total_work_items <= 1 || total_scalar_work_items < 16384) {
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

void EnsureX86ConcatKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
