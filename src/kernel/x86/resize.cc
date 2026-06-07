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

template <DataType dtype>
int32_t ComputeResizeFallback(feather::operators::ResizeParam* param) {
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
            input_offset += coord * in_strides[axis];
        }
        TensorIO<dtype>::Write(param->out.get(), linear, TensorIO<dtype>::Read(param->input.get(), input_offset));
    }
    return 0;
}

template <typename T>
int32_t ComputeResizeRaw(feather::operators::ResizeParam* param, DataType dtype) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() != out_dims.size() || in_dims.size() != param->scales.size()) {
        return -1;
    }

    const T* input = param->input->data<T>();
    T* output = param->out->mutable_data<T>();
    param->out->set_data_type(dtype);

    if (in_dims.size() == 4) {
        const int64_t batch = in_dims[0];
        const int64_t channels = in_dims[1];
        const int64_t in_h = in_dims[2];
        const int64_t in_w = in_dims[3];
        const int64_t out_h = out_dims[2];
        const int64_t out_w = out_dims[3];
        const float scale_h = param->scales[2];
        const float scale_w = param->scales[3];
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    const int64_t ih = std::max<int64_t>(
                        0, std::min<int64_t>(static_cast<int64_t>(static_cast<double>(oh) / scale_h), in_h - 1));
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        const int64_t iw = std::max<int64_t>(
                            0, std::min<int64_t>(static_cast<int64_t>(static_cast<double>(ow) / scale_w), in_w - 1));
                        const int64_t input_offset = ((n * channels + c) * in_h + ih) * in_w + iw;
                        const int64_t output_offset = ((n * channels + c) * out_h + oh) * out_w + ow;
                        output[output_offset] = input[input_offset];
                    }
                }
            }
        }
        return 0;
    }

    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    std::vector<int64_t> out_coords(out_dims.size(), 0);
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
            input_offset += coord * in_strides[axis];
        }
        output[linear] = input[input_offset];
    }
    return 0;
}

}  // namespace

template <>
int32_t ResizeKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Resize::FP32");
    auto* param = static_cast<feather::operators::ResizeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP32) {
        return ComputeResizeFallback<DataType::FP32>(param);
    }
    return ComputeResizeRaw<float>(param, DataType::FP32);
}

template <>
int32_t ResizeKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Resize::FP16");
    auto* param = static_cast<feather::operators::ResizeParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::FP16) {
        return ComputeResizeFallback<DataType::FP16>(param);
    }
    return ComputeResizeRaw<uint16_t>(param, DataType::FP16);
}

void EnsureX86ResizeKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
