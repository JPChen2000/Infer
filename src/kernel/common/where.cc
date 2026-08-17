#include "src/kernel/where.h"

#include <cstring>
#include <memory>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_where_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT32, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::INT32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Where", []() {
        return std::make_unique<WhereKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeWhere(feather::operators::WhereParam* param) {
    if (param == nullptr || param->condition == nullptr || param->x == nullptr || param->y == nullptr ||
        param->out == nullptr || param->condition->data_type() != DataType::BOOL || param->x->data_type() != dtype ||
        param->y->data_type() != dtype || !param->condition->IsInitialized() || !param->x->IsInitialized() ||
        !param->y->IsInitialized() || !param->out->IsInitialized()) {
        return -1;
    }
    const auto condition_dims = param->condition->dims().data();
    const auto x_dims = param->x->dims().data();
    const auto y_dims = param->y->dims().data();
    const auto output_dims = param->out->dims().data();
    const auto condition_strides = common_tensor_detail::ComputeStrides(condition_dims);
    const auto x_strides = common_tensor_detail::ComputeStrides(x_dims);
    const auto y_strides = common_tensor_detail::ComputeStrides(y_dims);
    const auto output_strides = common_tensor_detail::ComputeStrides(output_dims);
    std::vector<int64_t> coordinates;
    const size_t element_bytes = DataTypeBytes(dtype);
    if (element_bytes == 0 || param->out->memory_size() < static_cast<size_t>(param->out->numel()) * element_bytes) {
        return -1;
    }
    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, output_dims, output_strides, &coordinates);
        const int64_t condition_offset =
            common_tensor_detail::BroadcastOffset(coordinates, condition_dims, condition_strides);
        const int64_t x_offset = common_tensor_detail::BroadcastOffset(coordinates, x_dims, x_strides);
        const int64_t y_offset = common_tensor_detail::BroadcastOffset(coordinates, y_dims, y_strides);
        const Tensor* source = common_tensor_detail::ReadBool(param->condition.get(), condition_offset) ? param->x.get()
                                                                                                         : param->y.get();
        const int64_t source_offset = source == param->x.get() ? x_offset : y_offset;
        std::memcpy(static_cast<char*>(param->out->raw_data()) + linear * element_bytes,
                    static_cast<const char*>(source->raw_data()) + source_offset * element_bytes, element_bytes);
    }
    return 0;
}

}  // namespace

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Where::FP32");
    return ComputeWhere<DataType::FP32>(static_cast<feather::operators::WhereParam*>(param_));
}

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Where::FP16");
    return ComputeWhere<DataType::FP16>(static_cast<feather::operators::WhereParam*>(param_));
}

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Where::BF16");
    return ComputeWhere<DataType::BF16>(static_cast<feather::operators::WhereParam*>(param_));
}

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::INT32>::compute() {
    AutoTimer timer("Common::Where::INT32");
    return ComputeWhere<DataType::INT32>(static_cast<feather::operators::WhereParam*>(param_));
}

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Where::INT64");
    return ComputeWhere<DataType::INT64>(static_cast<feather::operators::WhereParam*>(param_));
}

template <>
int32_t WhereKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Where::BOOL");
    return ComputeWhere<DataType::BOOL>(static_cast<feather::operators::WhereParam*>(param_));
}

void EnsureCommonWhereKernelsRegistered() { (void)g_common_where_kernels_registered; }

void EnsureWhereKernelsRegistered() { EnsureCommonWhereKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
