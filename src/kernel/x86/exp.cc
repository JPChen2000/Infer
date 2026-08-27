#include "src/kernel/exp.h"

#include <cmath>

#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t ExpKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Exp::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != DataType::FP32) ||
        param->input->dims().data() != param->out->dims().data()) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    for (int64_t index = 0; index < param->input->numel(); ++index) {
        output[index] = std::exp(input[index]);
    }
    return 0;
}

void EnsureX86ExpKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Exp",
            []() { return std::make_unique<ExpKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
