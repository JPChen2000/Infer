#include "src/kernel/softmax.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_softmax_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Softmax",
                                                []() { return std::make_unique<SoftmaxKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Softmax",
                                                []() { return std::make_unique<SoftmaxKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Softmax",
                                                []() { return std::make_unique<SoftmaxKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeSoftmaxCommon(feather::operators::SoftmaxParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& dims = param->input->dims().data();
    const int32_t rank = static_cast<int32_t>(dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }
    param->out->set_data_type(dtype);

    int64_t outer = 1;
    int64_t inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        outer *= dims[i];
    }
    for (int32_t i = axis + 1; i < rank; ++i) {
        inner *= dims[i];
    }
    const int64_t axis_dim = dims[axis];

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
            const int64_t base = outer_idx * axis_dim * inner + inner_idx;
            float max_value = TensorIO<dtype>::Read(param->input.get(), base);
            for (int64_t axis_idx = 1; axis_idx < axis_dim; ++axis_idx) {
                const float value = TensorIO<dtype>::Read(param->input.get(), base + axis_idx * inner);
                if (value > max_value) {
                    max_value = value;
                }
            }

            float sum = 0.0f;
            for (int64_t axis_idx = 0; axis_idx < axis_dim; ++axis_idx) {
                const float value = std::exp(TensorIO<dtype>::Read(param->input.get(), base + axis_idx * inner) -
                                             max_value);
                TensorIO<dtype>::Write(param->out.get(), base + axis_idx * inner, value);
                sum += value;
            }
            for (int64_t axis_idx = 0; axis_idx < axis_dim; ++axis_idx) {
                const float normalized =
                    TensorIO<dtype>::Read(param->out.get(), base + axis_idx * inner) / sum;
                TensorIO<dtype>::Write(param->out.get(), base + axis_idx * inner, normalized);
            }
        }
    }

    return 0;
}

template <>
int32_t SoftmaxKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Softmax::FP32");
    return ComputeSoftmaxCommon<DataType::FP32>(static_cast<feather::operators::SoftmaxParam*>(param_));
}

template <>
int32_t SoftmaxKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Softmax::FP16");
    return ComputeSoftmaxCommon<DataType::FP16>(static_cast<feather::operators::SoftmaxParam*>(param_));
}

template <>
int32_t SoftmaxKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Softmax::BF16");
    return ComputeSoftmaxCommon<DataType::BF16>(static_cast<feather::operators::SoftmaxParam*>(param_));
}

typedef feather::kernel::SoftmaxKernel<DeviceType::COMMON, DataType::FP32> SoftmaxCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Softmax, SoftmaxCommonFP32Kernel);

typedef feather::kernel::SoftmaxKernel<DeviceType::COMMON, DataType::FP16> SoftmaxCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Softmax, SoftmaxCommonFP16Kernel);

void EnsureSoftmaxKernelsRegistered() {
    (void)g_softmax_kernels_registered;
    EnsureX86SoftmaxKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
