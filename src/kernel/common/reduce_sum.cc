#include "src/kernel/reduce_sum.h"

#include <numeric>
#include <set>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_reduce_sum_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "ReduceSum", []() {
        return std::make_unique<ReduceSumKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "ReduceSum", []() {
        return std::make_unique<ReduceSumKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "ReduceSum", []() {
        return std::make_unique<ReduceSumKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeReduceSum(feather::operators::ReduceSumParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->data_type() != dtype ||
        param->out->data_type() != dtype) {
        return -1;
    }
    const auto input_dims = param->input->dims().data();
    auto axes = common_tensor_detail::NormalizeAxes(param->axes, static_cast<int64_t>(input_dims.size()));
    if (axes.empty()) {
        axes.resize(input_dims.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    const std::set<int64_t> axis_set(axes.begin(), axes.end());
    if (axis_set.size() != axes.size()) return -1;
    for (int64_t axis : axes) {
        if (axis < 0 || axis >= static_cast<int64_t>(input_dims.size())) return -1;
    }

    const auto out_dims = param->out->dims().data();
    const auto input_strides = common_tensor_detail::ComputeStrides(input_dims);
    const auto out_strides = common_tensor_detail::ComputeStrides(out_dims);
    std::vector<float> sums(static_cast<size_t>(param->out->numel()), 0.0f);
    std::vector<int64_t> input_coords(input_dims.size(), 0);
    std::vector<int64_t> out_coords;

    for (int64_t linear = 0; linear < param->input->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, input_dims, input_strides, &input_coords);
        out_coords.clear();
        for (int64_t axis = 0; axis < static_cast<int64_t>(input_dims.size()); ++axis) {
            if (axis_set.count(axis) != 0) {
                if (param->keepdims) out_coords.push_back(0);
            } else {
                out_coords.push_back(input_coords[static_cast<size_t>(axis)]);
            }
        }
        if (out_coords.empty()) out_coords.push_back(0);
        int64_t out_offset = 0;
        for (size_t axis = 0; axis < out_coords.size(); ++axis) {
            out_offset += out_coords[axis] * out_strides[axis];
        }
        sums[static_cast<size_t>(out_offset)] += TensorIO<dtype>::Read(param->input.get(), linear);
    }
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        TensorIO<dtype>::Write(param->out.get(), index, sums[static_cast<size_t>(index)]);
    }
    return 0;
}

}  // namespace

template <> int32_t ReduceSumKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::ReduceSum::FP32");
    return ComputeReduceSum<DataType::FP32>(static_cast<feather::operators::ReduceSumParam*>(param_));
}
template <> int32_t ReduceSumKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::ReduceSum::FP16");
    return ComputeReduceSum<DataType::FP16>(static_cast<feather::operators::ReduceSumParam*>(param_));
}
template <> int32_t ReduceSumKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::ReduceSum::BF16");
    return ComputeReduceSum<DataType::BF16>(static_cast<feather::operators::ReduceSumParam*>(param_));
}

void EnsureReduceSumKernelsRegistered() { (void)g_reduce_sum_kernels_registered; }

}  // namespace kernel
}  // namespace feather
