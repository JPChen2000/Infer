#include "src/kernel/resize.h"

#include <algorithm>

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

bool g_resize_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Resize",
                                                []() { return std::make_unique<ResizeKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Resize",
                                                []() { return std::make_unique<ResizeKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeResizeKernel(feather::operators::ResizeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size()) {
        return -1;
    }

    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);

    param->out->set_data_type(dtype);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }

        int64_t input_offset = 0;
        for (size_t axis = 0; axis < in_dims.size(); ++axis) {
            const float scale = param->scales[axis];
            int64_t coord = static_cast<int64_t>(static_cast<double>(out_coords[axis]) / scale);
            coord = std::max<int64_t>(0, std::min<int64_t>(coord, in_dims[axis] - 1));
            in_coords[axis] = coord;
            input_offset += coord * in_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <>
int32_t ResizeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Resize::FP32");
    return ComputeResizeKernel<DataType::FP32>(static_cast<feather::operators::ResizeParam*>(param_));
}

template <>
int32_t ResizeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Resize::FP16");
    return ComputeResizeKernel<DataType::FP16>(static_cast<feather::operators::ResizeParam*>(param_));
}

typedef feather::kernel::ResizeKernel<DeviceType::COMMON, DataType::FP32> ResizeCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Resize, ResizeCommonFP32Kernel);

typedef feather::kernel::ResizeKernel<DeviceType::COMMON, DataType::FP16> ResizeCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Resize, ResizeCommonFP16Kernel);

void EnsureCommonResizeKernelsRegistered() { (void)g_resize_kernels_registered; }

void EnsureResizeKernelsRegistered() {
    EnsureCommonResizeKernelsRegistered();
    EnsureX86ResizeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
