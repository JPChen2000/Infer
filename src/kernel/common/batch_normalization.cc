#include "src/kernel/batch_normalization.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_batch_norm_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "BatchNormalization",
                                                []() { return std::make_unique<BatchNormalizationKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "BatchNormalization",
                                                []() { return std::make_unique<BatchNormalizationKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

int32_t ComputeBatchNormFp32(feather::operators::BatchNormParam* param) {
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->dims().size() != 4 || param->out->dims().size() != 4) {
        return -1;
    }

    ImageShape4D shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), &shape)) {
        return -1;
    }
    const int64_t channels = shape.c;
    if (param->scale->numel() != channels || param->bias->numel() != channels || param->mean->numel() != channels ||
        param->var->numel() != channels) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    param->out->set_layout(param->input->layout());
    const float epsilon = param->epsilon;
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float scale = TensorIO<DataType::FP32>::Read(param->scale.get(), c);
            const float bias = TensorIO<DataType::FP32>::Read(param->bias.get(), c);
            const float mean = TensorIO<DataType::FP32>::Read(param->mean.get(), c);
            const float var = TensorIO<DataType::FP32>::Read(param->var.get(), c);
            const float inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int64_t h = 0; h < shape.h; ++h) {
                for (int64_t w = 0; w < shape.w; ++w) {
                    const int64_t in_offset = OffsetForImage4D(param->input->layout(), n, c, h, w, channels, shape.h, shape.w);
                    const int64_t out_offset = OffsetForImage4D(param->out->layout(), n, c, h, w, channels, shape.h, shape.w);
                    const float x = TensorIO<DataType::FP32>::Read(param->input.get(), in_offset);
                    TensorIO<DataType::FP32>::Write(param->out.get(), out_offset, (x - mean) * inv_std * scale + bias);
                }
            }
        }
    }
    return 0;
}

int32_t ComputeBatchNormFp16(feather::operators::BatchNormParam* param) {
    if (param == nullptr || param->input == nullptr || param->scale == nullptr || param->bias == nullptr ||
        param->mean == nullptr || param->var == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->dims().size() != 4 || param->out->dims().size() != 4) {
        return -1;
    }

    ImageShape4D shape{};
    if (!DecodeImageShape4D(param->input->dims().data(), NormalizeDataLayout(param->input->layout()), &shape)) {
        return -1;
    }
    const int64_t channels = shape.c;
    if (param->scale->numel() != channels || param->bias->numel() != channels || param->mean->numel() != channels ||
        param->var->numel() != channels) {
        return -1;
    }

    param->out->set_data_type(DataType::FP16);
    param->out->set_layout(param->input->layout());
    const float epsilon = param->epsilon;
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float scale = TensorIO<DataType::FP16>::Read(param->scale.get(), c);
            const float bias = TensorIO<DataType::FP16>::Read(param->bias.get(), c);
            const float mean = TensorIO<DataType::FP16>::Read(param->mean.get(), c);
            const float var = TensorIO<DataType::FP16>::Read(param->var.get(), c);
            const float inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int64_t h = 0; h < shape.h; ++h) {
                for (int64_t w = 0; w < shape.w; ++w) {
                    const int64_t in_offset = OffsetForImage4D(param->input->layout(), n, c, h, w, channels, shape.h, shape.w);
                    const int64_t out_offset = OffsetForImage4D(param->out->layout(), n, c, h, w, channels, shape.h, shape.w);
                    const float x = TensorIO<DataType::FP16>::Read(param->input.get(), in_offset);
                    TensorIO<DataType::FP16>::Write(param->out.get(), out_offset, (x - mean) * inv_std * scale + bias);
                }
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t BatchNormalizationKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::BatchNormalization::FP32");
    return ComputeBatchNormFp32(static_cast<feather::operators::BatchNormParam*>(param_));
}

template <>
int32_t BatchNormalizationKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::BatchNormalization::FP16");
    return ComputeBatchNormFp16(static_cast<feather::operators::BatchNormParam*>(param_));
}

typedef feather::kernel::BatchNormalizationKernel<DeviceType::COMMON, DataType::FP32> BatchNormalizationCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, BatchNormalization, BatchNormalizationCommonFP32Kernel);

typedef feather::kernel::BatchNormalizationKernel<DeviceType::COMMON, DataType::FP16> BatchNormalizationCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, BatchNormalization, BatchNormalizationCommonFP16Kernel);

void EnsureCommonBatchNormalizationKernelsRegistered() { (void)g_batch_norm_kernels_registered; }

void EnsureBatchNormalizationKernelsRegistered() { EnsureCommonBatchNormalizationKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
