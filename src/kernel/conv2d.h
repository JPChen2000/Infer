#ifndef FEATHER_KERNEL_CONV_H
#define FEATHER_KERNEL_CONV_H
#include <typeinfo>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "util/logger.h"
using feather::Tensor;
namespace feather {
namespace kernel {

void EnsureCommonConv2DKernelsRegistered();
void EnsureX86Conv2DKernelsRegistered();
void EnsureConv2DKernelsRegistered();

template <DeviceType dev, DataType dtype>
class Conv2DKernel : public KernelBase {
   public:
    virtual int32_t compute() {
        LOG_INFO("use of unimplement kernel %s \n", typeid(*this).name());
        return 0;
    }

   protected:
    const Tensor* cached_weight_tensor_{nullptr};
    const Tensor* cached_bias_tensor_{nullptr};
    std::vector<float> cached_weight_buffer_;
    std::vector<float> cached_bias_buffer_;
    std::vector<float> cached_packed_input_buffer_;
    std::vector<float> cached_direct_weight_oc8_buffer_;
    std::vector<float> cached_direct_bias_oc8_buffer_;
    std::vector<float> cached_winograd_weight_oc8_buffer_;
    std::vector<float> cached_winograd_weight_buffer_;
};
}  // namespace kernel
}  // namespace feather
#endif  // FEATHER_KERNEL_CONV_H
