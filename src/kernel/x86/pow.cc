#include "src/kernel/pow.h"

#include <immintrin.h>

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

template <DataType dtype>
int32_t ComputePowFallback(feather::operators::PowParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        TensorIO<dtype>::Write(param->out.get(), i, std::pow(TensorIO<dtype>::Read(param->input.get(), i), exponent));
    }
    return 0;
}

inline float PowScalar(float value, float exponent) { return std::pow(value, exponent); }

}  // namespace

template <>
int32_t PowKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Pow::FP32");
    auto* param = static_cast<feather::operators::PowParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP32) {
        return ComputePowFallback<DataType::FP32>(param);
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) {
        return -1;
    }

    if (exponent == 2.0f) {
        int64_t index = 0;
        for (; index + 8 <= numel; index += 8) {
            const __m256 value = _mm256_loadu_ps(input + index);
            _mm256_storeu_ps(output + index, _mm256_mul_ps(value, value));
        }
        for (; index < numel; ++index) {
            output[index] = input[index] * input[index];
        }
        return 0;
    }

    alignas(32) float values[8];
    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input + i);
        _mm256_store_ps(values, input_vec);
        for (float& value : values) {
            value = PowScalar(value, exponent);
        }
        _mm256_storeu_ps(output + i, _mm256_load_ps(values));
    }
    for (; i < numel; ++i) {
        output[i] = PowScalar(input[i], exponent);
    }
    return 0;
}

template <>
int32_t PowKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Pow::FP16");
    auto* param = static_cast<feather::operators::PowParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP16) {
        return ComputePowFallback<DataType::FP16>(param);
    }

    param->out->set_data_type(DataType::FP16);
    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    const int64_t numel = param->input->numel();
    float exponent = param->exponent;
    if (param->exponent_tensor != nullptr && !ReadScalarFloatTensor(param->exponent_tensor.get(), &exponent)) {
        return -1;
    }

    if (exponent == 2.0f) {
        int64_t index = 0;
        for (; index + 8 <= numel; index += 8) {
            const __m128i value_bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + index));
            const __m256 value = _mm256_cvtph_ps(value_bits);
            const __m128i squared =
                _mm256_cvtps_ph(_mm256_mul_ps(value, value), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + index), squared);
        }
        for (; index < numel; ++index) {
            const float value = HalfToFloat(input[index]);
            output[index] = FloatToHalf(value * value);
        }
        return 0;
    }

    alignas(32) float values[8];
    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        const __m128i input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m256 float_vec = _mm256_cvtph_ps(input_vec);
        _mm256_store_ps(values, float_vec);
        for (float& value : values) {
            value = PowScalar(value, exponent);
        }
        const __m128i half_vec =
            _mm256_cvtps_ph(_mm256_load_ps(values), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), half_vec);
    }
    for (; i < numel; ++i) {
        output[i] = FloatToHalf(PowScalar(HalfToFloat(input[i]), exponent));
    }
    return 0;
}

void EnsureX86PowKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Pow",
            []() { return std::make_unique<PowKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Pow",
            []() { return std::make_unique<PowKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
