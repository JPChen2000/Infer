#include "src/kernel/cast.h"

#include <memory>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/bf16.h"
#include "util/fp16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

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
    return true;
}();

int32_t ComputeCast(feather::operators::CastParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !param->input->IsInitialized() ||
        !param->out->IsInitialized()) {
        return -1;
    }
    const size_t element_bytes = DataTypeBytes(param->to);
    const size_t required_bytes = static_cast<size_t>(param->input->numel()) * element_bytes;
    if (element_bytes == 0 || param->out->memory_size() < required_bytes) {
        return -1;
    }

    for (int64_t i = 0; i < param->input->numel(); ++i) {
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

void EnsureCommonCastKernelsRegistered() { (void)g_common_cast_kernels_registered; }

void EnsureCastKernelsRegistered() {
    EnsureCommonCastKernelsRegistered();
    EnsureX86CastKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
