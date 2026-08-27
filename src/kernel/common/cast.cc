#include "src/kernel/cast.h"

#include <cmath>
#include <limits>
#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/bf16.h"
#include "util/fp8.h"
#include "util/fp16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool IsFp8DataType(DataType data_type) {
    return data_type == DataType::FP8E4M3 || data_type == DataType::FP8E5M2;
}

bool HasValidFp8Quantization(const Tensor* tensor) {
    return tensor != nullptr && HasCompatiblePerTensorQuantization(tensor->quantization()) &&
           std::isfinite(tensor->quantization_scale()) && tensor->quantization_scale() > 0.0f;
}

bool g_common_cast_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP8E4M3, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::COMMON, DataType::FP8E4M3>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP8E5M2, "Cast", []() {
        return std::make_unique<CastKernel<DeviceType::COMMON, DataType::FP8E5M2>>();
    });
    return true;
}();

int32_t ComputeCast(feather::operators::CastParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !param->input->IsInitialized() ||
        !param->out->IsInitialized()) {
        return -1;
    }
    const int64_t input_numel = param->input->numel();
    const size_t input_bytes = DataTypeBytes(param->input->data_type());
    const size_t output_bytes = DataTypeBytes(param->to);
    if (input_numel < 0 || input_bytes == 0 || output_bytes == 0 ||
        param->out->dims().data() != param->input->dims().data() || param->out->numel() != input_numel ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != param->to) ||
        input_numel > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(input_bytes) ||
        static_cast<uint64_t>(input_numel) > std::numeric_limits<size_t>::max() / input_bytes ||
        static_cast<uint64_t>(input_numel) > std::numeric_limits<size_t>::max() / output_bytes ||
        param->input->memory_size() < static_cast<size_t>(input_numel) * input_bytes ||
        param->out->memory_size() < static_cast<size_t>(input_numel) * output_bytes ||
        (IsFp8DataType(param->input->data_type()) && !HasValidFp8Quantization(param->input.get())) ||
        (IsFp8DataType(param->to) && !HasValidFp8Quantization(param->out.get()))) {
        return -1;
    }

    for (int64_t i = 0; i < input_numel; ++i) {
        const float value = common_tensor_detail::ReadFloat(param->input.get(), i);
        switch (param->to) {
            case DataType::BOOL:
                static_cast<uint8_t*>(param->out->raw_data())[i] =
                    common_tensor_detail::ReadBool(param->input.get(), i) ? 1 : 0;
                break;
            case DataType::UINT8:
                static_cast<uint8_t*>(param->out->raw_data())[i] = static_cast<uint8_t>(value);
                break;
            case DataType::INT8:
                static_cast<int8_t*>(param->out->raw_data())[i] = static_cast<int8_t>(value);
                break;
            case DataType::FP16:
                static_cast<uint16_t*>(param->out->raw_data())[i] = FloatToHalf(value);
                break;
            case DataType::BF16:
                static_cast<BFloat16*>(param->out->raw_data())[i].bits = FloatToBFloat16(value);
                break;
            case DataType::FP8E4M3:
                static_cast<Fp8E4M3*>(param->out->raw_data())[i].bits =
                    FloatToFp8E4M3(value / param->out->quantization_scale());
                break;
            case DataType::FP8E5M2:
                static_cast<Fp8E5M2*>(param->out->raw_data())[i].bits =
                    FloatToFp8E5M2(value / param->out->quantization_scale());
                break;
            case DataType::INT32:
                static_cast<int32_t*>(param->out->raw_data())[i] = static_cast<int32_t>(value);
                break;
            case DataType::INT64:
                static_cast<int64_t*>(param->out->raw_data())[i] = static_cast<int64_t>(value);
                break;
            case DataType::FP32:
                static_cast<float*>(param->out->raw_data())[i] = value;
                break;
            default:
                return -1;
        }
    }
    param->out->set_data_type(param->to);
    return 0;
}

}  // namespace

template <>
int32_t CastKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Cast::FP32");
    return ComputeCast(static_cast<feather::operators::CastParam*>(param_));
}

template <>
int32_t CastKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Cast::FP16");
    return ComputeCast(static_cast<feather::operators::CastParam*>(param_));
}

template <>
int32_t CastKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Cast::BF16");
    return ComputeCast(static_cast<feather::operators::CastParam*>(param_));
}

template <DataType dtype>
int32_t ComputeFp8Cast(feather::operators::CastParam* param) {
    if (param == nullptr || param->input == nullptr || param->input->data_type() != dtype) return -1;
    return ComputeCast(param);
}

template <>
int32_t CastKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::Cast::FP8E4M3");
    return ComputeFp8Cast<DataType::FP8E4M3>(static_cast<feather::operators::CastParam*>(param_));
}

template <>
int32_t CastKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::Cast::FP8E5M2");
    return ComputeFp8Cast<DataType::FP8E5M2>(static_cast<feather::operators::CastParam*>(param_));
}

void EnsureCommonCastKernelsRegistered() { (void)g_common_cast_kernels_registered; }

void EnsureCastKernelsRegistered() {
    EnsureCommonCastKernelsRegistered();
    EnsureX86CastKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
