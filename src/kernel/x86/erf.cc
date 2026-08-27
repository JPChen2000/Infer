#include "src/kernel/erf.h"

#include <cmath>
#include <memory>

#include "util/threading.h"
#include "util/timer.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

constexpr int64_t kErfParallelThreshold = 1 << 15;

}  // namespace

template <>
int32_t ErfKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Erf::FP32");
    auto* param = static_cast<feather::operators::UnaryParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 || param->out->data_type() != DataType::FP32 ||
        param->input->dims().data() != param->out->dims().data()) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();

#if defined(FEATHER_WITH_OPENMP)
    if (numel >= kErfParallelThreshold && !omp_in_parallel()) {
        const size_t worker_count = ThreadCountForWorkItems(numel);
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int64_t index = 0; index < numel; ++index) {
            output[index] = std::erf(input[index]);
        }
        return 0;
    }
#endif
    for (int64_t index = 0; index < numel; ++index) {
        output[index] = std::erf(input[index]);
    }
    return 0;
}

void EnsureX86ErfKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Erf",
            []() { return std::make_unique<ErfKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
