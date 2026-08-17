#include "src/kernel/expand.h"

#include <cstring>
#include <memory>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_expand_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT32, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::COMMON, DataType::INT32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Expand", []() {
        return std::make_unique<ExpandKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeExpand(feather::operators::ExpandParam* param) {
    if (param == nullptr || param->input == nullptr || param->shape == nullptr || param->out == nullptr ||
        param->input->data_type() != dtype || !param->input->IsInitialized() || !param->out->IsInitialized() ||
        (param->shape->data_type() != DataType::INT64 && param->shape->data_type() != DataType::INT32)) {
        return -1;
    }
    std::vector<int64_t> target_shape;
    target_shape.reserve(static_cast<size_t>(param->shape->numel()));
    for (int64_t index = 0; index < param->shape->numel(); ++index) {
        target_shape.push_back(common_tensor_detail::ReadInteger(param->shape.get(), index));
    }
    const auto input_dims = param->input->dims().data();
    const auto output_dims = param->out->dims().data();
    if (target_shape != output_dims) {
        return -1;
    }
    const auto output_strides = common_tensor_detail::ComputeStrides(output_dims);
    const auto input_strides = common_tensor_detail::ComputeStrides(input_dims);
    std::vector<int64_t> coordinates;
    const size_t element_bytes = DataTypeBytes(dtype);
    if (element_bytes == 0 || param->out->memory_size() < static_cast<size_t>(param->out->numel()) * element_bytes) {
        return -1;
    }
    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, output_dims, output_strides, &coordinates);
        const int64_t input_offset = common_tensor_detail::BroadcastOffset(coordinates, input_dims, input_strides);
        std::memcpy(static_cast<char*>(param->out->raw_data()) + linear * element_bytes,
                    static_cast<const char*>(param->input->raw_data()) + input_offset * element_bytes, element_bytes);
    }
    return 0;
}

}  // namespace

template <>
int32_t ExpandKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Expand::FP32");
    return ComputeExpand<DataType::FP32>(static_cast<feather::operators::ExpandParam*>(param_));
}

template <>
int32_t ExpandKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Expand::FP16");
    return ComputeExpand<DataType::FP16>(static_cast<feather::operators::ExpandParam*>(param_));
}

template <>
int32_t ExpandKernel<DeviceType::COMMON, DataType::INT32>::compute() {
    AutoTimer timer("Common::Expand::INT32");
    return ComputeExpand<DataType::INT32>(static_cast<feather::operators::ExpandParam*>(param_));
}

template <>
int32_t ExpandKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Expand::INT64");
    return ComputeExpand<DataType::INT64>(static_cast<feather::operators::ExpandParam*>(param_));
}

template <>
int32_t ExpandKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Expand::BOOL");
    return ComputeExpand<DataType::BOOL>(static_cast<feather::operators::ExpandParam*>(param_));
}

void EnsureCommonExpandKernelsRegistered() { (void)g_common_expand_kernels_registered; }

void EnsureExpandKernelsRegistered() { EnsureCommonExpandKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
