#include "src/kernel/split.h"

#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_split_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Split",
                                                []() { return std::make_unique<SplitKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Split",
                                                []() { return std::make_unique<SplitKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Split",
                                                []() { return std::make_unique<SplitKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

int64_t ComputeProduct(const std::vector<int64_t>& dims, size_t begin, size_t end) {
    int64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        product *= dims[i];
    }
    return product;
}

}  // namespace

template <DataType dtype>
int32_t ComputeSplitKernel(feather::operators::SplitParam* param) {
    if (param == nullptr || param->input == nullptr || param->outputs.empty()) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(in_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(in_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(in_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(in_dims, static_cast<size_t>(axis) + 1, in_dims.size());
    const int64_t input_axis = in_dims[axis];

    for (auto& output : param->outputs) {
        if (output == nullptr) {
            return -1;
        }
        output->set_data_type(dtype);
    }

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (size_t i = 0; i < param->outputs.size(); ++i) {
            const int64_t output_axis = param->outputs[i]->dims()[axis];
            const int64_t copy_count = output_axis * inner;
            const int64_t input_base = (outer_idx * input_axis + axis_offset) * inner;
            const int64_t output_base = outer_idx * copy_count;
            for (int64_t j = 0; j < copy_count; ++j) {
                TensorIO<dtype>::Write(param->outputs[i].get(), output_base + j,
                                       TensorIO<dtype>::Read(param->input.get(), input_base + j));
            }
            axis_offset += output_axis;
        }
    }
    return 0;
}

template <>
int32_t SplitKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Split::FP32");
    return ComputeSplitKernel<DataType::FP32>(static_cast<feather::operators::SplitParam*>(param_));
}

template <>
int32_t SplitKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Split::FP16");
    return ComputeSplitKernel<DataType::FP16>(static_cast<feather::operators::SplitParam*>(param_));
}

template <>
int32_t SplitKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Split::BF16");
    return ComputeSplitKernel<DataType::BF16>(static_cast<feather::operators::SplitParam*>(param_));
}

typedef feather::kernel::SplitKernel<DeviceType::COMMON, DataType::FP32> SplitCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Split, SplitCommonFP32Kernel);

typedef feather::kernel::SplitKernel<DeviceType::COMMON, DataType::FP16> SplitCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Split, SplitCommonFP16Kernel);

typedef feather::kernel::SplitKernel<DeviceType::COMMON, DataType::BF16> SplitCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Split, SplitCommonBF16Kernel);

void EnsureCommonSplitKernelsRegistered() { (void)g_split_kernels_registered; }

void EnsureSplitKernelsRegistered() {
    EnsureCommonSplitKernelsRegistered();
    EnsureX86SplitKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
