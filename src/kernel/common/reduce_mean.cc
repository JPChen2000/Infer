#include "src/kernel/reduce_mean.h"

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_reduce_mean_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeReduceMean(feather::operators::ReduceMeanParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->data_type() != dtype) {
        return -1;
    }
    const auto input_dims = param->input->dims().data();
    auto axes = common_tensor_detail::NormalizeAxes(param->axes, static_cast<int64_t>(input_dims.size()));
    if (axes.empty()) {
        axes.resize(input_dims.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    for (const int64_t axis : axes) {
        if (axis < 0 || axis >= static_cast<int64_t>(input_dims.size())) {
            return -1;
        }
    }
    const std::set<int64_t> axis_set(axes.begin(), axes.end());
    if (axis_set.size() != axes.size()) {
        return -1;
    }

    const auto out_dims = param->out->dims().data();
    const auto input_strides = common_tensor_detail::ComputeStrides(input_dims);
    const auto out_strides = common_tensor_detail::ComputeStrides(out_dims);
    std::vector<int64_t> input_coords(input_dims.size(), 0);
    std::vector<int64_t> out_coords;
    std::vector<float> sums(static_cast<size_t>(param->out->numel()), 0.0f);

    float reduce_count = 1.0f;
    for (const int64_t axis : axes) {
        reduce_count *= static_cast<float>(input_dims[axis]);
    }
    for (int64_t linear = 0; linear < param->input->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, input_dims, input_strides, &input_coords);
        out_coords.clear();
        for (int64_t axis = 0; axis < static_cast<int64_t>(input_dims.size()); ++axis) {
            if (axis_set.count(axis) != 0) {
                if (param->keepdims) {
                    out_coords.push_back(0);
                }
            } else {
                out_coords.push_back(input_coords[axis]);
            }
        }
        if (out_coords.empty()) {
            out_coords.push_back(0);
        }
        int64_t out_offset = 0;
        for (size_t axis = 0; axis < out_coords.size(); ++axis) {
            out_offset += out_coords[axis] * out_strides[axis];
        }
        sums[static_cast<size_t>(out_offset)] += TensorIO<dtype>::Read(param->input.get(), linear);
    }
    for (int64_t index = 0; index < param->out->numel(); ++index) {
        TensorIO<dtype>::Write(param->out.get(), index, sums[static_cast<size_t>(index)] / reduce_count);
    }
    param->out->set_data_type(dtype);
    return 0;
}

}  // namespace

template <>
int32_t ReduceMeanKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::ReduceMean::FP32");
    return ComputeReduceMean<DataType::FP32>(static_cast<feather::operators::ReduceMeanParam*>(param_));
}

template <>
int32_t ReduceMeanKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::ReduceMean::FP16");
    return ComputeReduceMean<DataType::FP16>(static_cast<feather::operators::ReduceMeanParam*>(param_));
}

template <>
int32_t ReduceMeanKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::ReduceMean::BF16");
    return ComputeReduceMean<DataType::BF16>(static_cast<feather::operators::ReduceMeanParam*>(param_));
}

void EnsureCommonReduceMeanKernelsRegistered() { (void)g_common_reduce_mean_kernels_registered; }

void EnsureReduceMeanKernelsRegistered() { EnsureCommonReduceMeanKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
