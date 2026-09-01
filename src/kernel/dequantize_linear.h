#ifndef FEATHER_KERNEL_DEQUANTIZE_LINEAR_H
#define FEATHER_KERNEL_DEQUANTIZE_LINEAR_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureDequantizeLinearKernelsRegistered();

class CommonDequantizeLinearKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_DEQUANTIZE_LINEAR_H
