#ifndef FEATHER_KERNEL_SIN_H
#define FEATHER_KERNEL_SIN_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureSinKernelsRegistered();

template <DeviceType dev, DataType dtype>
class SinKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SIN_H
