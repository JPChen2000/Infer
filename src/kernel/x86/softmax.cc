#include "src/kernel/softmax.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "util/fp16.h"
#include "util/thread_pool_nv.h"
#include "util/threading.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

struct SoftmaxShape {
    int64_t outer{0};
    int64_t inner{0};
    int64_t axis_dim{0};
};

bool PrepareSoftmaxShape(const feather::operators::SoftmaxParam* param, SoftmaxShape* shape) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || shape == nullptr) {
        return false;
    }

    const auto& dims = param->input->dims().data();
    const int32_t rank = static_cast<int32_t>(dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return false;
    }

    shape->outer = 1;
    shape->inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        shape->outer *= dims[i];
    }
    for (int32_t i = axis + 1; i < rank; ++i) {
        shape->inner *= dims[i];
    }
    shape->axis_dim = dims[axis];
    return shape->axis_dim > 0;
}

float HorizontalMax(__m256 vec) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vec);
    float max_value = tmp[0];
    for (int i = 1; i < 8; ++i) {
        max_value = std::max(max_value, tmp[i]);
    }
    return max_value;
}

float FindMaxAvx(const float* data, int64_t count) {
    if (count <= 0) {
        return -std::numeric_limits<float>::infinity();
    }

    int64_t idx = 0;
    float max_value = data[0];
    if (count >= 8) {
        __m256 max_vec = _mm256_loadu_ps(data);
        idx = 8;
        for (; idx + 8 <= count; idx += 8) {
            max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(data + idx));
        }
        max_value = HorizontalMax(max_vec);
    }

    for (; idx < count; ++idx) {
        max_value = std::max(max_value, data[idx]);
    }
    return max_value;
}

inline void ConvertHalfArrayToFloat(const uint16_t* input, int64_t count, float* output) {
    int64_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        const __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + idx));
        const __m256 value = _mm256_cvtph_ps(half);
        _mm256_storeu_ps(output + idx, value);
    }
    for (; idx < count; ++idx) {
        output[idx] = HalfToFloat(input[idx]);
    }
}

inline void StoreFloatArrayToHalf(const float* input, int64_t count, uint16_t* output) {
    int64_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        const __m256 value = _mm256_loadu_ps(input + idx);
        const __m128i half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + idx), half);
    }
    for (; idx < count; ++idx) {
        output[idx] = FloatToHalf(input[idx]);
    }
}

size_t GetSoftmaxThreadCount(int64_t total_work_items) {
    return ThreadCountForWorkItems(total_work_items);
}

ThreadPoolNv& GetSoftmaxThreadPool() {
    static ThreadPoolNv pool(DefaultThreadCount());
    return pool;
}

template <typename Fn>
void ParallelForSoftmaxWorkItems(int64_t total_work_items, int64_t total_scalar_work_items, Fn&& fn) {
    if (total_work_items <= 1 || total_scalar_work_items < 16384) {
        fn(0, total_work_items);
        return;
    }

    const size_t thread_count = GetSoftmaxThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetSoftmaxThreadPool();
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

void ComputeExpAndNormalize(float* values, int64_t count) {
    const float max_value = FindMaxAvx(values, count);
    float sum = 0.0f;
    for (int64_t idx = 0; idx < count; ++idx) {
        values[idx] = std::exp(values[idx] - max_value);
        sum += values[idx];
    }

    const float inv_sum = 1.0f / sum;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    int64_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        const __m256 value = _mm256_loadu_ps(values + idx);
        _mm256_storeu_ps(values + idx, _mm256_mul_ps(value, inv_sum_vec));
    }
    for (; idx < count; ++idx) {
        values[idx] *= inv_sum;
    }
}

template <DataType dtype>
int32_t ComputeSoftmaxFallback(feather::operators::SoftmaxParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    SoftmaxShape shape;
    if (!PrepareSoftmaxShape(param, &shape)) {
        return -1;
    }
    param->out->set_data_type(dtype);

    for (int64_t outer_idx = 0; outer_idx < shape.outer; ++outer_idx) {
        for (int64_t inner_idx = 0; inner_idx < shape.inner; ++inner_idx) {
            const int64_t base = outer_idx * shape.axis_dim * shape.inner + inner_idx;
            float max_value = TensorIO<dtype>::Read(param->input.get(), base);
            for (int64_t axis_idx = 1; axis_idx < shape.axis_dim; ++axis_idx) {
                const float value = TensorIO<dtype>::Read(param->input.get(), base + axis_idx * shape.inner);
                if (value > max_value) {
                    max_value = value;
                }
            }

            float sum = 0.0f;
            for (int64_t axis_idx = 0; axis_idx < shape.axis_dim; ++axis_idx) {
                const float value =
                    std::exp(TensorIO<dtype>::Read(param->input.get(), base + axis_idx * shape.inner) - max_value);
                TensorIO<dtype>::Write(param->out.get(), base + axis_idx * shape.inner, value);
                sum += value;
            }

            for (int64_t axis_idx = 0; axis_idx < shape.axis_dim; ++axis_idx) {
                const float normalized =
                    TensorIO<dtype>::Read(param->out.get(), base + axis_idx * shape.inner) / sum;
                TensorIO<dtype>::Write(param->out.get(), base + axis_idx * shape.inner, normalized);
            }
        }
    }
    return 0;
}

int32_t ComputeSoftmaxFp32X86(feather::operators::SoftmaxParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != DataType::FP32)) {
        return -1;
    }

    SoftmaxShape shape;
    if (!PrepareSoftmaxShape(param, &shape)) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t total_work_items = shape.outer * shape.inner;
    const int64_t total_scalar_work_items = total_work_items * shape.axis_dim;

    ParallelForSoftmaxWorkItems(total_work_items, total_scalar_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(shape.axis_dim), 0.0f);
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t outer_index = work_index / shape.inner;
            const int64_t inner_index = work_index % shape.inner;
            const int64_t base = outer_index * shape.axis_dim * shape.inner + inner_index;
            if (shape.inner == 1) {
                std::memcpy(scratch.data(), input + base, static_cast<size_t>(shape.axis_dim) * sizeof(float));
                ComputeExpAndNormalize(scratch.data(), shape.axis_dim);
                std::memcpy(output + base, scratch.data(), static_cast<size_t>(shape.axis_dim) * sizeof(float));
                continue;
            }

            for (int64_t axis_index = 0; axis_index < shape.axis_dim; ++axis_index) {
                scratch[static_cast<size_t>(axis_index)] = input[base + axis_index * shape.inner];
            }
            ComputeExpAndNormalize(scratch.data(), shape.axis_dim);
            for (int64_t axis_index = 0; axis_index < shape.axis_dim; ++axis_index) {
                output[base + axis_index * shape.inner] = scratch[static_cast<size_t>(axis_index)];
            }
        }
    });
    return 0;
}

int32_t ComputeSoftmaxFp16X86(feather::operators::SoftmaxParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    SoftmaxShape shape;
    if (!PrepareSoftmaxShape(param, &shape)) {
        return -1;
    }

    param->out->set_data_type(DataType::FP16);
    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    const int64_t total_work_items = shape.outer * shape.inner;
    const int64_t total_scalar_work_items = total_work_items * shape.axis_dim;

    if (shape.inner == 1) {
        ParallelForSoftmaxWorkItems(total_work_items, total_scalar_work_items, [&](int64_t begin, int64_t end) {
            std::vector<float> scratch(static_cast<size_t>(shape.axis_dim), 0.0f);
            for (int64_t work_index = begin; work_index < end; ++work_index) {
                const uint16_t* input_row = input + work_index * shape.axis_dim;
                uint16_t* output_row = output + work_index * shape.axis_dim;
                ConvertHalfArrayToFloat(input_row, shape.axis_dim, scratch.data());
                ComputeExpAndNormalize(scratch.data(), shape.axis_dim);
                StoreFloatArrayToHalf(scratch.data(), shape.axis_dim, output_row);
            }
        });
        return 0;
    }

    ParallelForSoftmaxWorkItems(total_work_items, total_scalar_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(shape.axis_dim), 0.0f);
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t outer_idx = work_index / shape.inner;
            const int64_t inner_idx = work_index % shape.inner;
            const int64_t base = outer_idx * shape.axis_dim * shape.inner + inner_idx;

            for (int64_t axis_idx = 0; axis_idx < shape.axis_dim; ++axis_idx) {
                scratch[static_cast<size_t>(axis_idx)] = HalfToFloat(input[base + axis_idx * shape.inner]);
            }

            ComputeExpAndNormalize(scratch.data(), shape.axis_dim);

            for (int64_t axis_idx = 0; axis_idx < shape.axis_dim; ++axis_idx) {
                output[base + axis_idx * shape.inner] = FloatToHalf(scratch[static_cast<size_t>(axis_idx)]);
            }
        }
    });

    return 0;
}

}  // namespace

template <>
int32_t SoftmaxKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Softmax::FP32");
    auto* param = static_cast<feather::operators::SoftmaxParam*>(param_);
    return ComputeSoftmaxFp32X86(param);
}

template <>
int32_t SoftmaxKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Softmax::FP16");
    auto* param = static_cast<feather::operators::SoftmaxParam*>(param_);
    return ComputeSoftmaxFp16X86(param);
}

void EnsureX86SoftmaxKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
