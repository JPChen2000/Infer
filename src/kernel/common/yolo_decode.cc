#include "src/kernel/yolo_decode.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_yolo_decode_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(
        DeviceType::COMMON, DataType::FP32, "YoloDecode",
        []() { return std::make_unique<YoloDecodeKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::COMMON, DataType::FP16, "YoloDecode",
        []() { return std::make_unique<YoloDecodeKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

inline float Sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

template <DataType dtype>
int32_t ComputeYoloDecodeKernel(feather::operators::YoloDecodeParam* param) {
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr ||
        param->grid == nullptr || param->stride == nullptr || param->wh_scale == nullptr ||
        param->anchor_grid == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& grid_dims = param->grid->dims().data();
    if (in_dims.size() != 4 || grid_dims.size() != 5) {
        return -1;
    }

    ImageShape4D input_shape;
    if (!DecodeImageShape4D(in_dims, param->input->layout(), &input_shape)) {
        return -1;
    }
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    const int64_t batch = input_shape.n;
    const int64_t channels = input_shape.c;
    const int64_t height = input_shape.h;
    const int64_t width = input_shape.w;
    const int64_t anchors = grid_dims[1];
    const int64_t attrs = channels / anchors;
    if (anchors <= 0 || attrs < 5 || channels % anchors != 0) {
        return -1;
    }

    const float xy_scale = TensorIO<dtype>::Read(param->xy_scale.get(), 0);
    const float stride = TensorIO<dtype>::Read(param->stride.get(), 0);
    const float wh_scale = TensorIO<dtype>::Read(param->wh_scale.get(), 0);
    const int64_t grid_batch = grid_dims[0];
    param->out->set_data_type(dtype);

    for (int64_t n = 0; n < batch; ++n) {
        const int64_t grid_n = grid_batch == 1 ? 0 : n;
        for (int64_t anchor = 0; anchor < anchors; ++anchor) {
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t point = ((anchor * height + y) * width + x);
                    const int64_t out_base = (n * anchors * height * width + point) * attrs;
                    for (int64_t attr = 0; attr < attrs; ++attr) {
                        const int64_t channel = anchor * attrs + attr;
                        const int64_t input_offset =
                            OffsetForImage4D(layout, n, channel, y, x, channels, height, width);
                        const float value = Sigmoid(TensorIO<dtype>::Read(param->input.get(), input_offset));
                        float decoded = value;
                        if (attr < 2) {
                            const int64_t grid_offset =
                                ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + attr;
                            decoded = (value * xy_scale + TensorIO<dtype>::Read(param->grid.get(), grid_offset)) *
                                      stride;
                        } else if (attr < 4) {
                            const int64_t anchor_offset =
                                ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + (attr - 2);
                            const float scaled = value * wh_scale;
                            decoded = scaled * scaled * TensorIO<dtype>::Read(param->anchor_grid.get(), anchor_offset);
                        }
                        TensorIO<dtype>::Write(param->out.get(), out_base + attr, decoded);
                    }
                }
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t YoloDecodeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::YoloDecode::FP32");
    return ComputeYoloDecodeKernel<DataType::FP32>(static_cast<feather::operators::YoloDecodeParam*>(param_));
}

template <>
int32_t YoloDecodeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::YoloDecode::FP16");
    return ComputeYoloDecodeKernel<DataType::FP16>(static_cast<feather::operators::YoloDecodeParam*>(param_));
}

typedef feather::kernel::YoloDecodeKernel<DeviceType::COMMON, DataType::FP32> YoloDecodeCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, YoloDecode, YoloDecodeCommonFP32Kernel);

typedef feather::kernel::YoloDecodeKernel<DeviceType::COMMON, DataType::FP16> YoloDecodeCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, YoloDecode, YoloDecodeCommonFP16Kernel);

void EnsureCommonYoloDecodeKernelsRegistered() { (void)g_yolo_decode_kernels_registered; }

void EnsureYoloDecodeKernelsRegistered() {
    EnsureCommonYoloDecodeKernelsRegistered();
    EnsureX86YoloDecodeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
