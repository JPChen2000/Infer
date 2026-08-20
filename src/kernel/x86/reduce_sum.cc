#include "src/kernel/reduce_sum.h"

#include <algorithm>
#include <immintrin.h>
#include <numeric>
#include <set>
#include <vector>

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<size_t>(axis)] =
            strides[static_cast<size_t>(axis + 1)] * dims[static_cast<size_t>(axis + 1)];
    }
    return strides;
}

std::vector<int64_t> NormalizeAxes(const std::vector<int64_t>& axes, int64_t rank) {
    std::vector<int64_t> normalized;
    normalized.reserve(axes.size());
    for (int64_t axis : axes) {
        normalized.push_back(axis < 0 ? axis + rank : axis);
    }
    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

float HorizontalSum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

int32_t ComputeReduceSumFallback(feather::operators::ReduceSumParam* param,
                                 const std::vector<int64_t>& normalized_axes) {
    const auto input_dims = param->input->dims().data();
    const auto out_dims = param->out->dims().data();
    const std::set<int64_t> axis_set(normalized_axes.begin(), normalized_axes.end());
    const auto input_strides = ComputeStrides(input_dims);
    const auto out_strides = ComputeStrides(out_dims);
    std::vector<float> sums(static_cast<size_t>(param->out->numel()), 0.0f);
    std::vector<int64_t> input_coords(input_dims.size(), 0);
    std::vector<int64_t> out_coords;
    const float* input = param->input->data<float>();

    for (int64_t linear = 0; linear < param->input->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < input_dims.size(); ++axis) {
            input_coords[axis] = remaining / input_strides[axis];
            remaining %= input_strides[axis];
        }
        out_coords.clear();
        for (int64_t axis = 0; axis < static_cast<int64_t>(input_dims.size()); ++axis) {
            if (axis_set.count(axis) != 0) {
                if (param->keepdims) out_coords.push_back(0);
            } else {
                out_coords.push_back(input_coords[static_cast<size_t>(axis)]);
            }
        }
        if (out_coords.empty()) out_coords.push_back(0);
        if (out_coords.size() != out_strides.size()) return -1;
        int64_t out_offset = 0;
        for (size_t axis = 0; axis < out_coords.size(); ++axis) {
            out_offset += out_coords[axis] * out_strides[axis];
        }
        sums[static_cast<size_t>(out_offset)] += input[linear];
    }
    std::copy(sums.begin(), sums.end(), static_cast<float*>(param->out->raw_data()));
    return 0;
}

int32_t ComputeReduceSumFp32(feather::operators::ReduceSumParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !param->input->IsInitialized() ||
        !param->out->IsInitialized() || param->input->data_type() != DataType::FP32) {
        return -1;
    }
    const auto& input_dims = param->input->dims().data();
    if (input_dims.empty()) return -1;
    auto axes = NormalizeAxes(param->axes, static_cast<int64_t>(input_dims.size()));
    if (axes.empty()) {
        axes.resize(input_dims.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    if (std::set<int64_t>(axes.begin(), axes.end()).size() != axes.size() ||
        std::any_of(axes.begin(), axes.end(), [&input_dims](int64_t axis) {
            return axis < 0 || axis >= static_cast<int64_t>(input_dims.size());
        })) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* out = static_cast<float*>(param->out->raw_data());
    if (axes.size() == input_dims.size()) {
        if (param->out->numel() != 1) return -1;
        __m256 acc = _mm256_setzero_ps();
        int64_t index = 0;
        for (; index + 8 <= param->input->numel(); index += 8) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(input + index));
        }
        float sum = HorizontalSum(acc);
        for (; index < param->input->numel(); ++index) sum += input[index];
        out[0] = sum;
        return 0;
    }
    if (axes.size() != 1) {
        return ComputeReduceSumFallback(param, axes);
    }

    const int64_t axis = axes.front();
    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + axis, int64_t{1},
                                          std::multiplies<int64_t>());
    const int64_t reduce = input_dims[static_cast<size_t>(axis)];
    const int64_t inner = std::accumulate(input_dims.begin() + axis + 1, input_dims.end(), int64_t{1},
                                          std::multiplies<int64_t>());
    if (outer <= 0 || reduce <= 0 || inner <= 0 || param->out->numel() != outer * inner) return -1;

    if (inner == 1) {
        for (int64_t group = 0; group < outer; ++group) {
            const float* input_group = input + group * reduce;
            __m256 acc = _mm256_setzero_ps();
            int64_t index = 0;
            for (; index + 8 <= reduce; index += 8) {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(input_group + index));
            }
            float sum = HorizontalSum(acc);
            for (; index < reduce; ++index) sum += input_group[index];
            out[group] = sum;
        }
        return 0;
    }

    for (int64_t group = 0; group < outer; ++group) {
        float* out_group = out + group * inner;
        std::fill(out_group, out_group + inner, 0.0f);
        for (int64_t reduction = 0; reduction < reduce; ++reduction) {
            const float* input_row = input + (group * reduce + reduction) * inner;
            int64_t index = 0;
            for (; index + 8 <= inner; index += 8) {
                const __m256 previous = _mm256_loadu_ps(out_group + index);
                _mm256_storeu_ps(out_group + index, _mm256_add_ps(previous, _mm256_loadu_ps(input_row + index)));
            }
            for (; index < inner; ++index) out_group[index] += input_row[index];
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t ReduceSumKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::ReduceSum::FP32");
    return ComputeReduceSumFp32(static_cast<feather::operators::ReduceSumParam*>(param_));
}

void EnsureX86ReduceSumKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReduceSum",
            []() { return std::make_unique<ReduceSumKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
