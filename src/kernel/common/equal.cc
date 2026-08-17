#include "src/kernel/equal.h"

#include <memory>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_equal_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT32, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::INT32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Equal", []() {
        return std::make_unique<EqualKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

int32_t ComputeEqual(feather::operators::EqualParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        param->lhs->data_type() != param->rhs->data_type() || !param->out->IsInitialized()) {
        return -1;
    }
    const auto lhs_dims = param->lhs->dims().data();
    const auto rhs_dims = param->rhs->dims().data();
    const auto out_dims = param->out->dims().data();
    const auto out_strides = common_tensor_detail::ComputeStrides(out_dims);
    const auto lhs_strides = common_tensor_detail::ComputeStrides(lhs_dims);
    const auto rhs_strides = common_tensor_detail::ComputeStrides(rhs_dims);
    std::vector<int64_t> coordinates;
    auto* output = static_cast<uint8_t*>(param->out->raw_data());
    param->out->set_data_type(DataType::BOOL);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, out_dims, out_strides, &coordinates);
        const int64_t lhs_offset =
            common_tensor_detail::BroadcastOffset(coordinates, lhs_dims, lhs_strides);
        const int64_t rhs_offset =
            common_tensor_detail::BroadcastOffset(coordinates, rhs_dims, rhs_strides);
        output[linear] = common_tensor_detail::SameValue(param->lhs.get(), lhs_offset, param->rhs.get(), rhs_offset) ? 1 : 0;
    }
    return 0;
}

}  // namespace

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Equal::FP32");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Equal::FP16");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Equal::BF16");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::INT32>::compute() {
    AutoTimer timer("Common::Equal::INT32");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Equal::INT64");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

template <>
int32_t EqualKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Equal::BOOL");
    return ComputeEqual(static_cast<feather::operators::EqualParam*>(param_));
}

void EnsureCommonEqualKernelsRegistered() { (void)g_common_equal_kernels_registered; }

void EnsureEqualKernelsRegistered() { EnsureCommonEqualKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
