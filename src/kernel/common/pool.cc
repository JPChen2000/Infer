#include "src/kernel/pool.h"

#include <algorithm>
#include <limits>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_pool_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "AvgPool",
                                                []() { return std::make_unique<AvgPoolKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "MaxPool",
                                                []() { return std::make_unique<MaxPoolKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "AvgPool",
                                                []() { return std::make_unique<AvgPoolKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "MaxPool",
                                                []() { return std::make_unique<MaxPoolKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype, bool is_max>
int32_t ComputePoolKernel(feather::operators::PoolParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const bool is_4d = param->input->dims().size() == 4;
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (is_4d &&
        (!DecodeImageShape4D(param->input->dims().data(), layout, &input_shape) ||
         !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape))) {
        return -1;
    }
    const int batch = is_4d ? static_cast<int>(input_shape.n) : 1;
    const int channels = is_4d ? static_cast<int>(input_shape.c) : 1;
    const int in_h = is_4d ? static_cast<int>(input_shape.h) : static_cast<int>(param->input->dims()[0]);
    const int in_w = is_4d ? static_cast<int>(input_shape.w) : static_cast<int>(param->input->dims()[1]);
    const int out_h = is_4d ? static_cast<int>(output_shape.h) : static_cast<int>(param->out->dims()[0]);
    const int out_w = is_4d ? static_cast<int>(output_shape.w) : static_cast<int>(param->out->dims()[1]);

    param->out->set_data_type(dtype);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float value = is_max ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < param->kernel_h; ++kh) {
                        for (int kw = 0; kw < param->kernel_w; ++kw) {
                            const int ih = oh * param->stride_h + kh - param->pad_h;
                            const int iw = ow * param->stride_w + kw - param->pad_w;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                continue;
                            }
                            const int64_t input_offset = is_4d
                                ? OffsetForImage4D(layout, n, c, ih, iw, channels, in_h, in_w)
                                : static_cast<int64_t>(ih) * in_w + iw;
                            const float input_value = TensorIO<dtype>::Read(param->input.get(), input_offset);
                            if (is_max) {
                                value = std::max(value, input_value);
                            } else {
                                value += input_value;
                                ++count;
                            }
                        }
                    }
                    if (!is_max) {
                        value = count == 0 ? 0.0f : value / static_cast<float>(count);
                    }
                    const int64_t output_offset = is_4d
                        ? OffsetForImage4D(layout, n, c, oh, ow, channels, out_h, out_w)
                        : static_cast<int64_t>(oh) * out_w + ow;
                    TensorIO<dtype>::Write(param->out.get(), output_offset, value);
                }
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t AvgPoolKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::AvgPool::FP32");
    return ComputePoolKernel<DataType::FP32, false>(static_cast<feather::operators::PoolParam*>(param_));
}

template <>
int32_t AvgPoolKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::AvgPool::FP16");
    return ComputePoolKernel<DataType::FP16, false>(static_cast<feather::operators::PoolParam*>(param_));
}

template <>
int32_t MaxPoolKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::MaxPool::FP32");
    return ComputePoolKernel<DataType::FP32, true>(static_cast<feather::operators::PoolParam*>(param_));
}

template <>
int32_t MaxPoolKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::MaxPool::FP16");
    return ComputePoolKernel<DataType::FP16, true>(static_cast<feather::operators::PoolParam*>(param_));
}

typedef feather::kernel::AvgPoolKernel<DeviceType::COMMON, DataType::FP32> AvgPoolCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, AvgPool, AvgPoolCommonFP32Kernel);

typedef feather::kernel::AvgPoolKernel<DeviceType::COMMON, DataType::FP16> AvgPoolCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, AvgPool, AvgPoolCommonFP16Kernel);

typedef feather::kernel::MaxPoolKernel<DeviceType::COMMON, DataType::FP32> MaxPoolCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, MaxPool, MaxPoolCommonFP32Kernel);

typedef feather::kernel::MaxPoolKernel<DeviceType::COMMON, DataType::FP16> MaxPoolCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, MaxPool, MaxPoolCommonFP16Kernel);

void EnsureCommonPoolKernelsRegistered() { (void)g_pool_kernels_registered; }

void EnsurePoolKernelsRegistered() {
    EnsureCommonPoolKernelsRegistered();
    EnsureX86PoolKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
