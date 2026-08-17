#include "src/kernel/slice.h"

#include <cstring>

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

bool g_slice_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Slice",
                                                []() { return std::make_unique<SliceKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Slice",
                                                []() { return std::make_unique<SliceKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT64, "Slice",
                                                []() { return std::make_unique<SliceKernel<DeviceType::COMMON, DataType::INT64>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeSliceCommon(feather::operators::SliceParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const int32_t rank = static_cast<int32_t>(in_dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }

    const int64_t dim = in_dims[axis];
    int64_t start = param->start < 0 ? param->start + dim : param->start;
    int64_t end = param->end < 0 ? param->end + dim : param->end;
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(dim, end);

    param->out->set_data_type(dtype);
    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    std::vector<int64_t> in_coords(in_dims.size(), 0);

    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t i = 0; i < out_dims.size(); ++i) {
            out_coords[i] = remaining / out_strides[i];
            remaining %= out_strides[i];
        }
        for (size_t i = 0; i < in_dims.size(); ++i) {
            in_coords[i] = out_coords[i];
        }
        in_coords[axis] += start;

        int64_t input_offset = 0;
        for (size_t i = 0; i < in_dims.size(); ++i) {
            input_offset += in_coords[i] * in_strides[i];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <>
int32_t SliceKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Slice::FP32");
    auto* param = static_cast<feather::operators::SliceParam*>(param_);
    return ComputeSliceCommon<DataType::FP32>(param);
}

template <>
int32_t SliceKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Slice::FP16");
    auto* param = static_cast<feather::operators::SliceParam*>(param_);
    return ComputeSliceCommon<DataType::FP16>(param);
}

int32_t SliceKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Slice::INT64");
    auto* param = static_cast<feather::operators::SliceParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    const int32_t rank = static_cast<int32_t>(in_dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    const int64_t dim = in_dims[axis];
    int64_t start = param->start < 0 ? param->start + dim : param->start;
    start = std::max<int64_t>(0, start);
    param->out->set_data_type(DataType::INT64);
    auto* out = param->out->mutable_data<int64_t>();
    const auto out_strides = ComputeStrides(out_dims);
    const auto in_strides = ComputeStrides(in_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t i = 0; i < out_dims.size(); ++i) {
            out_coords[i] = remaining / out_strides[i];
            remaining %= out_strides[i];
        }
        int64_t input_offset = 0;
        for (size_t i = 0; i < in_dims.size(); ++i) {
            const int64_t coord = static_cast<int32_t>(i) == axis ? out_coords[i] + start : out_coords[i];
            input_offset += coord * in_strides[i];
        }
        out[linear] = param->input->data<int64_t>()[input_offset];
    }
    return 0;
}

typedef feather::kernel::SliceKernel<DeviceType::COMMON, DataType::FP32> SliceCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Slice, SliceCommonFP32Kernel);

typedef feather::kernel::SliceKernel<DeviceType::COMMON, DataType::FP16> SliceCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Slice, SliceCommonFP16Kernel);

void EnsureSliceKernelsRegistered() {
    (void)g_slice_kernels_registered;
    EnsureX86SliceKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
