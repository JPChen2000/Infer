#include "src/kernel/transpose.h"

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

bool g_transpose_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Transpose",
                                                []() { return std::make_unique<TransposeKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Transpose",
                                                []() { return std::make_unique<TransposeKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Transpose",
                                                []() { return std::make_unique<TransposeKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeTransposeKernel(feather::operators::TransposeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();

    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);

    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        std::fill(in_coords.begin(), in_coords.end(), 0);
        for (size_t axis = 0; axis < param->perm.size(); ++axis) {
            in_coords[param->perm[axis]] = out_coords[axis];
        }
        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            input_offset += in_coords[axis] * in_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <>
int32_t TransposeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Transpose::FP32");
    return ComputeTransposeKernel<DataType::FP32>(static_cast<feather::operators::TransposeParam*>(param_));
}

template <>
int32_t TransposeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Transpose::FP16");
    return ComputeTransposeKernel<DataType::FP16>(static_cast<feather::operators::TransposeParam*>(param_));
}

template <>
int32_t TransposeKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Transpose::BF16");
    return ComputeTransposeKernel<DataType::BF16>(static_cast<feather::operators::TransposeParam*>(param_));
}

typedef feather::kernel::TransposeKernel<DeviceType::COMMON, DataType::FP32> TransposeCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Transpose, TransposeCommonFP32Kernel);

typedef feather::kernel::TransposeKernel<DeviceType::COMMON, DataType::FP16> TransposeCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Transpose, TransposeCommonFP16Kernel);

typedef feather::kernel::TransposeKernel<DeviceType::COMMON, DataType::BF16> TransposeCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Transpose, TransposeCommonBF16Kernel);

void EnsureCommonTransposeKernelsRegistered() { (void)g_transpose_kernels_registered; }

void EnsureTransposeKernelsRegistered() {
    EnsureCommonTransposeKernelsRegistered();
    EnsureX86TransposeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
