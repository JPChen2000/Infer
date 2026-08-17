#ifndef FEATHER_KERNEL_CONSTANT_OF_SHAPE_H
#define FEATHER_KERNEL_CONSTANT_OF_SHAPE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonConstantOfShapeKernelsRegistered();
void EnsureConstantOfShapeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ConstantOfShapeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CONSTANT_OF_SHAPE_H
