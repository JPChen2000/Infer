#ifndef FEATHER_KERNEL_NEG_H
#define FEATHER_KERNEL_NEG_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureNegKernelsRegistered();

template <DeviceType dev, DataType dtype>
class NegKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_NEG_H
