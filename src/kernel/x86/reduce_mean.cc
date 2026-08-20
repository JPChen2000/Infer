#include "src/kernel/reduce_mean.h"

#include <immintrin.h>

#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

float HorizontalSum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

bool TryComputeLastAxisMeanFp32(feather::operators::ReduceMeanParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !param->input->IsInitialized() ||
        !param->out->IsInitialized() || param->input->data_type() != DataType::FP32) {
        return false;
    }

    const auto& input_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (input_dims.empty()) {
        return false;
    }
    const int64_t rank = static_cast<int64_t>(input_dims.size());
    const auto axes = common_tensor_detail::NormalizeAxes(param->axes, rank);
    if (axes.size() != 1 || axes.front() != rank - 1) {
        return false;
    }

    const int64_t columns = input_dims.back();
    if (columns <= 0 || param->input->numel() % columns != 0) {
        return false;
    }
    const int64_t rows = param->input->numel() / columns;
    if (param->out->numel() != rows) {
        return false;
    }
    if (param->keepdims) {
        if (out_dims.size() != input_dims.size() || out_dims.back() != 1) {
            return false;
        }
        for (size_t axis = 0; axis + 1 < input_dims.size(); ++axis) {
            if (out_dims[axis] != input_dims[axis]) {
                return false;
            }
        }
    } else if (rank == 1) {
        if (out_dims != std::vector<int64_t>{1}) {
            return false;
        }
    } else {
        if (out_dims.size() + 1 != input_dims.size()) {
            return false;
        }
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            if (out_dims[axis] != input_dims[axis]) {
                return false;
            }
        }
    }

    const float* input = param->input->data<float>();
    float* out = static_cast<float*>(param->out->raw_data());
    for (int64_t row = 0; row < rows; ++row) {
        const float* input_row = input + row * columns;
        __m256 sum = _mm256_setzero_ps();
        int64_t column = 0;
        for (; column + 8 <= columns; column += 8) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(input_row + column));
        }
        float mean = HorizontalSum(sum);
        for (; column < columns; ++column) {
            mean += input_row[column];
        }
        out[row] = mean / static_cast<float>(columns);
    }
    param->out->set_data_type(DataType::FP32);
    return true;
}

}  // namespace

template <>
int32_t ReduceMeanKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::ReduceMean::FP32");
    auto* param = static_cast<feather::operators::ReduceMeanParam*>(param_);
    if (TryComputeLastAxisMeanFp32(param)) {
        return 0;
    }

    ReduceMeanKernel<DeviceType::COMMON, DataType::FP32> fallback;
    fallback.SetParam(param);
    return fallback.compute();
}

void EnsureX86ReduceMeanKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReduceMean",
            []() { return std::make_unique<ReduceMeanKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
