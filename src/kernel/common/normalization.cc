// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/kernel/normalization.h"

#include <cmath>

#include "util/bf16.h"
#include "util/fp16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

float Read(const void* data, DataType type, int64_t index) {
    switch (type) {
        case DataType::FP16: return HalfToFloat(static_cast<const uint16_t*>(data)[index]);
        case DataType::BF16: return BFloat16ToFloat(static_cast<const BFloat16*>(data)[index].bits);
        default: return static_cast<const float*>(data)[index];
    }
}

void Write(void* data, DataType type, int64_t index, float value) {
    switch (type) {
        case DataType::FP16: static_cast<uint16_t*>(data)[index] = FloatToHalf(value); break;
        case DataType::BF16: static_cast<BFloat16*>(data)[index].bits = FloatToBFloat16(value); break;
        default: static_cast<float*>(data)[index] = value; break;
    }
}

template <DataType dtype>
int32_t Compute(operators::NormalizationParam* param) {
    operators::normalization_detail::Shape shape;
    if (!param || !operators::normalization_detail::ValidateStorage(*param, &shape) ||
        param->input->data_type() != dtype) return -1;
    const void* input = param->input->raw_data();
    const void* weight = param->weight->raw_data();
    const void* bias = param->bias ? param->bias->raw_data() : nullptr;
    const DataType weight_type = param->weight->data_type();
    const DataType bias_type = param->bias ? param->bias->data_type() : dtype;
    void* output = param->out->mutable_data(param->out->memory_size());
    float* means = param->mean ? param->mean->mutable_data<float>() : nullptr;
    float* inv_stds = param->inv_std_dev ? param->inv_std_dev->mutable_data<float>() : nullptr;
    const auto& dims = param->input->dims().data();
    const bool flat_weight = param->weight->dims().data() ==
                             std::vector<int64_t>(dims.begin() + shape.axis, dims.end());
    const bool flat_bias = param->bias && param->bias->dims().data() ==
                           std::vector<int64_t>(dims.begin() + shape.axis, dims.end());
    for (int64_t row = 0; row < shape.rows; ++row) {
        const int64_t base = row * shape.width;
        float mean = 0.0f, sum = 0.0f;
        if (param->layer_norm) {
            // Welford avoids cancellation for nearly constant large inputs.
            for (int64_t col = 0; col < shape.width; ++col) {
                const float value = Read(input, dtype, base + col);
                const float delta = value - mean;
                mean += delta / static_cast<float>(col + 1);
                sum += delta * (value - mean);
            }
        } else {
            for (int64_t col = 0; col < shape.width; ++col) {
                const float value = Read(input, dtype, base + col);
                sum += value * value;
            }
        }
        const float inv_std = 1.0f / std::sqrt(sum / static_cast<float>(shape.width) + param->epsilon);
        if (means) means[row] = mean;
        if (inv_stds) inv_stds[row] = inv_std;
        for (int64_t col = 0; col < shape.width; ++col) {
            const int64_t index = base + col;
            const int64_t wi = flat_weight ? col :
                operators::normalization_detail::AffineOffset(index, shape, shape.weight_strides);
            const float scale = Read(weight, weight_type, wi) + param->weight_offset;
            const int64_t bi = flat_bias ? col :
                operators::normalization_detail::AffineOffset(index, shape, shape.bias_strides);
            const float shift = bias ? Read(bias, bias_type, bi) : 0.0f;
            Write(output, dtype, index, (Read(input, dtype, index) - mean) * inv_std * scale + shift);
        }
    }
    return 0;
}

}  // namespace

template <> int32_t NormalizationKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Normalization::FP32");
    return Compute<DataType::FP32>(ParamAs<operators::NormalizationParam>());
}
template <> int32_t NormalizationKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Normalization::FP16");
    return Compute<DataType::FP16>(ParamAs<operators::NormalizationParam>());
}
template <> int32_t NormalizationKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Normalization::BF16");
    return Compute<DataType::BF16>(ParamAs<operators::NormalizationParam>());
}

namespace {
template <DataType dtype> void Register() {
    for (const char* name : {"RMSNormalization", "LayerNormalization"}) {
        KernelDispatcher::instance().registerKernel(DeviceType::COMMON, dtype, name,
            [] { return std::make_unique<NormalizationKernel<DeviceType::COMMON, dtype>>(); });
    }
}
bool registered = [] {
    Register<DataType::FP32>(); Register<DataType::FP16>(); Register<DataType::BF16>();
    return true;
}();
}  // namespace

void EnsureCommonNormalizationKernelsRegistered() { (void)registered; }
void EnsureNormalizationKernelsRegistered() {
    EnsureCommonNormalizationKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaNormalizationKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
