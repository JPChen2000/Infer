#ifndef FEATHER_KERNEL_ERF_H
#define FEATHER_KERNEL_ERF_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureErfKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ErfKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_ERF_H
