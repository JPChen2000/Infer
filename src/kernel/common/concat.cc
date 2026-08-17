#include "src/kernel/concat.h"

#include <cstring>
#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_concat_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Concat",
                                                []() { return std::make_unique<ConcatKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Concat",
                                                []() { return std::make_unique<ConcatKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Concat",
                                                []() { return std::make_unique<ConcatKernel<DeviceType::COMMON, DataType::BF16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT64, "Concat",
                                                []() { return std::make_unique<ConcatKernel<DeviceType::COMMON, DataType::INT64>>(); });
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
int32_t ComputeConcatKernel(feather::operators::ConcatParam* param) {
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) {
        return -1;
    }

    const auto& out_dims = param->out->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(out_dims.size()) : param->axis;
    if (axis < 0 || axis >= static_cast<int32_t>(out_dims.size())) {
        return -1;
    }

    const int64_t outer = ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t out_axis = out_dims[axis];

    param->out->set_data_type(dtype);
    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (const auto& input : param->inputs) {
            if (input == nullptr) {
                return -1;
            }
            const int64_t input_axis = input->dims()[axis];
            const int64_t copy_count = input_axis * inner;
            const int64_t input_base = outer_idx * input_axis * inner;
            const int64_t output_base = (outer_idx * out_axis + axis_offset) * inner;
            for (int64_t i = 0; i < copy_count; ++i) {
                TensorIO<dtype>::Write(param->out.get(), output_base + i, TensorIO<dtype>::Read(input.get(), input_base + i));
            }
            axis_offset += input_axis;
        }
    }
    return 0;
}

template <>
int32_t ConcatKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Concat::FP32");
    return ComputeConcatKernel<DataType::FP32>(static_cast<feather::operators::ConcatParam*>(param_));
}

template <>
int32_t ConcatKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Concat::FP16");
    return ComputeConcatKernel<DataType::FP16>(static_cast<feather::operators::ConcatParam*>(param_));
}

template <>
int32_t ConcatKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Concat::BF16");
    return ComputeConcatKernel<DataType::BF16>(static_cast<feather::operators::ConcatParam*>(param_));
}

int32_t ConcatKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Concat::INT64");
    auto* param = static_cast<feather::operators::ConcatParam*>(param_);
    if (param == nullptr || param->out == nullptr || param->inputs.size() < 2) {
        return -1;
    }
    const auto& out_dims = param->out->dims().data();
    int32_t axis = param->axis < 0 ? param->axis + static_cast<int32_t>(out_dims.size()) : param->axis;
    const int64_t outer = ComputeProduct(out_dims, 0, static_cast<size_t>(axis));
    const int64_t inner = ComputeProduct(out_dims, static_cast<size_t>(axis) + 1, out_dims.size());
    const int64_t out_axis = out_dims[axis];
    std::vector<int64_t> out_data(static_cast<size_t>(param->out->numel()), 0);
    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        int64_t axis_offset = 0;
        for (const auto& input : param->inputs) {
            if (input == nullptr || input->data_type() != DataType::INT64 || !input->IsInitialized() ||
                input->memory_size() < static_cast<size_t>(input->numel()) * sizeof(int64_t)) {
                return -1;
            }
            const int64_t input_axis = input->dims()[axis];
            const int64_t copy_count = input_axis * inner;
            const int64_t input_base = outer_idx * copy_count;
            const int64_t output_base = (outer_idx * out_axis + axis_offset) * inner;
            if (input_base + copy_count > input->numel() || output_base + copy_count > param->out->numel()) {
                return -1;
            }
            const auto* input_data = input->data<int64_t>();
            for (int64_t i = 0; i < copy_count; ++i) {
                out_data[static_cast<size_t>(output_base + i)] = input_data[input_base + i];
            }
            axis_offset += input_axis;
        }
    }
    param->out->Assign<int64_t>(out_data, out_dims);
    return 0;
}

typedef feather::kernel::ConcatKernel<DeviceType::COMMON, DataType::FP32> ConcatCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Concat, ConcatCommonFP32Kernel);

typedef feather::kernel::ConcatKernel<DeviceType::COMMON, DataType::FP16> ConcatCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Concat, ConcatCommonFP16Kernel);

typedef feather::kernel::ConcatKernel<DeviceType::COMMON, DataType::BF16> ConcatCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Concat, ConcatCommonBF16Kernel);

void EnsureCommonConcatKernelsRegistered() { (void)g_concat_kernels_registered; }

void EnsureConcatKernelsRegistered() {
    EnsureCommonConcatKernelsRegistered();
    EnsureX86ConcatKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
