#include "src/kernel/sigmoid.h"

#include <immintrin.h>

#include <array>
#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

template <DataType dtype>
int32_t ComputeSigmoidFallback(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    for (int64_t i = 0; i < param->input->numel(); ++i) {
        const float value = TensorIO<dtype>::Read(param->input.get(), i);
        TensorIO<dtype>::Write(param->out.get(), i, 1.0f / (1.0f + std::exp(-value)));
    }
    return 0;
}

inline float SigmoidScalar(float value) { return 1.0f / (1.0f + std::exp(-value)); }

const std::array<uint16_t, 1u << 16>& GetSigmoidFp16Lut() {
    static const std::array<uint16_t, 1u << 16> lut = []() {
        std::array<uint16_t, 1u << 16> table{};
        for (size_t i = 0; i < table.size(); ++i) {
            table[i] = FloatToHalf(SigmoidScalar(HalfToFloat(static_cast<uint16_t>(i))));
        }
        return table;
    }();
    return lut;
}

}  // namespace

template <>
int32_t SigmoidKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Sigmoid::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP32) {
        return ComputeSigmoidFallback<DataType::FP32>(param);
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();

    alignas(32) float values[8];
    for (int64_t i = 0; i + 8 <= numel; i += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input + i);
        _mm256_store_ps(values, input_vec);
        for (float& value : values) {
            value = SigmoidScalar(value);
        }
        const __m256 output_vec = _mm256_load_ps(values);
        _mm256_storeu_ps(output + i, output_vec);
    }
    for (int64_t i = (numel / 8) * 8; i < numel; ++i) {
        output[i] = SigmoidScalar(input[i]);
    }
    return 0;
}

template <>
int32_t SigmoidKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Sigmoid::FP16");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->data_type() != DataType::FP16) {
        return ComputeSigmoidFallback<DataType::FP16>(param);
    }

    param->out->set_data_type(DataType::FP16);
    const uint16_t* input = param->input->data<uint16_t>();
    uint16_t* output = param->out->mutable_data<uint16_t>();
    const int64_t numel = param->input->numel();

    const auto& lut = GetSigmoidFp16Lut();
    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        output[i + 0] = lut[input[i + 0]];
        output[i + 1] = lut[input[i + 1]];
        output[i + 2] = lut[input[i + 2]];
        output[i + 3] = lut[input[i + 3]];
        output[i + 4] = lut[input[i + 4]];
        output[i + 5] = lut[input[i + 5]];
        output[i + 6] = lut[input[i + 6]];
        output[i + 7] = lut[input[i + 7]];
    }
    for (; i < numel; ++i) {
        output[i] = lut[input[i]];
    }
    return 0;
}

void EnsureX86SigmoidKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sigmoid",
            []() { return std::make_unique<SigmoidKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Sigmoid",
            []() { return std::make_unique<SigmoidKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
