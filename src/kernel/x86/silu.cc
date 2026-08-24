#include "src/kernel/silu.h"

#include <immintrin.h>

#include <array>
#include <cmath>
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

inline float SiluScalar(float value) {
    return value / (1.0f + std::exp(-value));
}

template <DataType dtype>
int32_t ComputeSiluFallback(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, SiluScalar(TensorIO<dtype>::Read(param->input.get(), i)));
    }
    return 0;
}

const std::array<uint16_t, 1u << 16>& GetSiluFp16Lut() {
    static const std::array<uint16_t, 1u << 16> lut = []() {
        std::array<uint16_t, 1u << 16> table{};
        for (size_t i = 0; i < table.size(); ++i) {
            table[i] = FloatToHalf(SiluScalar(HalfToFloat(static_cast<uint16_t>(i))));
        }
        return table;
    }();
    return lut;
}

const std::array<uint16_t, 1u << 16>& GetSiluBf16Lut() {
    static const std::array<uint16_t, 1u << 16> lut = []() {
        std::array<uint16_t, 1u << 16> table{};
        for (size_t i = 0; i < table.size(); ++i) {
            table[i] = FloatToBFloat16(SiluScalar(BFloat16ToFloat(static_cast<uint16_t>(i))));
        }
        return table;
    }();
    return lut;
}

size_t GetSiluThreadCount(int64_t total_work_items) {
    return ThreadCountForWorkItems(total_work_items);
}

ThreadPoolNv& GetSiluThreadPool() {
    static ThreadPoolNv pool(DefaultThreadCount());
    return pool;
}

template <typename Fn>
void ParallelForSilu(int64_t total_work_items, Fn&& fn) {
    if (total_work_items <= 1 || total_work_items < 16384) {
        fn(0, total_work_items);
        return;
    }

    const size_t thread_count = GetSiluThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetSiluThreadPool();
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

}  // namespace

template <>
int32_t SiluKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::SiLU::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP32) {
        return ComputeSiluFallback<DataType::FP32>(param);
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();

    ParallelForSilu(numel, [&](int64_t begin, int64_t end) {
        alignas(32) float values[8];
        int64_t i = begin;
        for (; i + 8 <= end; i += 8) {
            const __m256 input_vec = _mm256_loadu_ps(input + i);
            _mm256_store_ps(values, input_vec);
            for (float& value : values) {
                value = SiluScalar(value);
            }
            const __m256 output_vec = _mm256_load_ps(values);
            _mm256_storeu_ps(output + i, output_vec);
        }
        for (; i < end; ++i) {
            output[i] = SiluScalar(input[i]);
        }
    });
    return 0;
}

template <>
int32_t SiluKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::SiLU::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP16) {
        return ComputeSiluFallback<DataType::FP16>(param);
    }

    param->out->set_data_type(DataType::FP16);
    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    const int64_t numel = param->input->numel();

    const auto& lut = GetSiluFp16Lut();
    ParallelForSilu(numel, [&](int64_t begin, int64_t end) {
        int64_t i = begin;
        for (; i + 8 <= end; i += 8) {
            output[i + 0] = lut[input[i + 0]];
            output[i + 1] = lut[input[i + 1]];
            output[i + 2] = lut[input[i + 2]];
            output[i + 3] = lut[input[i + 3]];
            output[i + 4] = lut[input[i + 4]];
            output[i + 5] = lut[input[i + 5]];
            output[i + 6] = lut[input[i + 6]];
            output[i + 7] = lut[input[i + 7]];
        }
        for (; i < end; ++i) {
            output[i] = lut[input[i]];
        }
    });
    return 0;
}

template <>
int32_t SiluKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::SiLU::BF16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::BF16) {
        return ComputeSiluFallback<DataType::BF16>(param);
    }

    const auto& lut = GetSiluBf16Lut();
    const BFloat16* input = param->input->data<BFloat16>();
    BFloat16* output = param->out->mutable_data<BFloat16>();
    const int64_t numel = param->input->numel();
    ParallelForSilu(numel, [&](int64_t begin, int64_t end) {
        int64_t i = begin;
        for (; i + 8 <= end; i += 8) {
            output[i + 0].bits = lut[input[i + 0].bits];
            output[i + 1].bits = lut[input[i + 1].bits];
            output[i + 2].bits = lut[input[i + 2].bits];
            output[i + 3].bits = lut[input[i + 3].bits];
            output[i + 4].bits = lut[input[i + 4].bits];
            output[i + 5].bits = lut[input[i + 5].bits];
            output[i + 6].bits = lut[input[i + 6].bits];
            output[i + 7].bits = lut[input[i + 7].bits];
        }
        for (; i < end; ++i) {
            output[i].bits = lut[input[i].bits];
        }
    });
    return 0;
}

void EnsureX86SiluKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
