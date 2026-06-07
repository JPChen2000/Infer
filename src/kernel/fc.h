#ifndef FEATHER_KERNEL_FC_COMPUTE_H
#define FEATHER_KERNEL_FC_COMPUTE_H
#include "core/kernel.h"
#include "core/tensor.h"
#include "util/logger.h"
using feather::Tensor;
namespace feather {
namespace kernel {

void EnsureFcKernelsRegistered();

template <DeviceType dev, DataType dtype>
class FcKernel : public KernelBase {
   public:
    int32_t compute();
};
}  // namespace kernel
}  // namespace feather
// template class feather::kernel::FcCUDAKernel<DataType::FP32>;
#endif  // FEATHER_KERNEL_FC_COMPUTE_H
