#ifndef FEATHER_KERNEL_QUANTIZE_LINEAR_H
#define FEATHER_KERNEL_QUANTIZE_LINEAR_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureQuantizeLinearKernelsRegistered();

class CommonQuantizeLinearKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_QUANTIZE_LINEAR_H
