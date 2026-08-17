#include "src/kernel/global_average_pool.h"

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_gap_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "GlobalAveragePool",
                                                []() { return std::make_unique<GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "GlobalAveragePool",
                                                []() { return std::make_unique<GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype>
int32_t ComputeGlobalAveragePool(feather::operators::GlobalAveragePoolParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
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
    param->out->set_data_type(dtype);
    param->out->set_layout(param->input->layout());
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int64_t h = 0; h < shape.h; ++h) {
                for (int64_t w = 0; w < shape.w; ++w) {
                    const int64_t in_offset = OffsetForImage4D(param->input->layout(), n, c, h, w, channels, shape.h, shape.w);
                    sum += TensorIO<dtype>::Read(param->input.get(), in_offset);
                }
            }
            const float mean = sum / static_cast<float>(shape.h * shape.w);
            const int64_t out_offset = OffsetForImage4D(param->out->layout(), n, c, 0, 0, channels, 1, 1);
            TensorIO<dtype>::Write(param->out.get(), out_offset, mean);
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::GlobalAveragePool::FP32");
    return ComputeGlobalAveragePool<DataType::FP32>(static_cast<feather::operators::GlobalAveragePoolParam*>(param_));
}

template <>
int32_t GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::GlobalAveragePool::FP16");
    return ComputeGlobalAveragePool<DataType::FP16>(static_cast<feather::operators::GlobalAveragePoolParam*>(param_));
}

typedef feather::kernel::GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP32> GlobalAveragePoolCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, GlobalAveragePool, GlobalAveragePoolCommonFP32Kernel);

typedef feather::kernel::GlobalAveragePoolKernel<DeviceType::COMMON, DataType::FP16> GlobalAveragePoolCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, GlobalAveragePool, GlobalAveragePoolCommonFP16Kernel);

void EnsureCommonGlobalAveragePoolKernelsRegistered() { (void)g_gap_kernels_registered; }

void EnsureGlobalAveragePoolKernelsRegistered() { EnsureCommonGlobalAveragePoolKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
