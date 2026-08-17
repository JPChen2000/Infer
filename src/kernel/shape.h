#ifndef FEATHER_KERNEL_SHAPE_H
#define FEATHER_KERNEL_SHAPE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

void EnsureCommonShapeKernelsRegistered();
void EnsureShapeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ShapeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_SHAPE_H
