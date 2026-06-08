#include "src/kernel/softmax.h"

#include <cmath>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

template <DataType dtype>
int32_t ComputeSoftmaxFallback(feather::operators::SoftmaxParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& dims = param->input->dims().data();
    const int32_t rank = static_cast<int32_t>(dims.size());
    const int32_t axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }

    int64_t outer = 1;
    int64_t inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        outer *= dims[i];
    }
    for (int32_t i = axis + 1; i < rank; ++i) {
        inner *= dims[i];
    }
    const int64_t axis_dim = dims[axis];
    param->out->set_data_type(dtype);

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
                const float value =
                    std::exp(TensorIO<dtype>::Read(param->input.get(), base + axis_idx * inner) - max_value);
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

}  // namespace

template <>
int32_t SoftmaxKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Softmax::FP32");
    auto* param = static_cast<feather::operators::SoftmaxParam*>(param_);
    return ComputeSoftmaxFallback<DataType::FP32>(param);
}

template <>
int32_t SoftmaxKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Softmax::FP16");
    auto* param = static_cast<feather::operators::SoftmaxParam*>(param_);
    return ComputeSoftmaxFallback<DataType::FP16>(param);
}

void EnsureX86SoftmaxKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
