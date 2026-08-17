#include "src/kernel/gather.h"

#include <memory>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_gather_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Gather", []() {
        return std::make_unique<GatherKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Gather", []() {
        return std::make_unique<GatherKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeGather(feather::operators::GatherParam* param) {
    if (param == nullptr || param->data == nullptr || param->indices == nullptr || param->out == nullptr ||
        param->data->data_type() != dtype ||
        (param->indices->data_type() != DataType::INT32 && param->indices->data_type() != DataType::INT64)) {
        return -1;
    }
    const auto data_dims = param->data->dims().data();
    const auto indices_dims = param->indices->dims().data();
    const auto out_dims = param->out->dims().data();
    const int64_t data_rank = static_cast<int64_t>(data_dims.size());
    const int64_t axis = param->axis < 0 ? param->axis + data_rank : param->axis;
    if (axis < 0 || axis >= data_rank) {
        return -1;
    }
    const auto data_strides = common_tensor_detail::ComputeStrides(data_dims);
    const auto indices_strides = common_tensor_detail::ComputeStrides(indices_dims);
    const auto out_strides = common_tensor_detail::ComputeStrides(out_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);

    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, out_dims, out_strides, &out_coords);
        int64_t indices_offset = 0;
        for (size_t index_axis = 0; index_axis < indices_dims.size(); ++index_axis) {
            indices_offset += out_coords[static_cast<size_t>(axis) + index_axis] * indices_strides[index_axis];
        }
        int64_t index = common_tensor_detail::ReadIndex(param->indices.get(), indices_offset);
        if (index < 0) {
            index += data_dims[static_cast<size_t>(axis)];
        }
        if (index < 0 || index >= data_dims[static_cast<size_t>(axis)]) {
            return -1;
        }
        int64_t data_offset = 0;
        for (int64_t data_axis = 0; data_axis < axis; ++data_axis) {
            data_offset += out_coords[static_cast<size_t>(data_axis)] * data_strides[static_cast<size_t>(data_axis)];
        }
        data_offset += index * data_strides[static_cast<size_t>(axis)];
        for (int64_t data_axis = axis + 1; data_axis < data_rank; ++data_axis) {
            const size_t coord_index = static_cast<size_t>(data_axis - 1 + static_cast<int64_t>(indices_dims.size()));
            data_offset += out_coords[coord_index] * data_strides[static_cast<size_t>(data_axis)];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->data.get(), data_offset));
    }
    param->out->set_data_type(dtype);
    return 0;
}

}  // namespace

template <>
int32_t GatherKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Gather::FP32");
    return ComputeGather<DataType::FP32>(static_cast<feather::operators::GatherParam*>(param_));
}

template <>
int32_t GatherKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Gather::FP16");
    return ComputeGather<DataType::FP16>(static_cast<feather::operators::GatherParam*>(param_));
}

void EnsureCommonGatherKernelsRegistered() { (void)g_common_gather_kernels_registered; }

void EnsureGatherKernelsRegistered() { EnsureCommonGatherKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
