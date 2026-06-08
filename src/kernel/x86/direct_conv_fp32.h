#ifndef FEATHER_KERNEL_X86_DIRECT_CONV_FP32_H
#define FEATHER_KERNEL_X86_DIRECT_CONV_FP32_H

#include <cstdint>
#include <vector>

namespace feather {
namespace kernel {
namespace x86 {

int32_t ComputeDirectConv2DX86Fp32(const float* input, const float* weight, const float* bias, int64_t batch,
                                   int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_h,
                                   int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
                                   int64_t pad_w, float* output);

void ComputeDirectConv2DSpatialOutputX86Fp32(const float* input, const float* weight, const float* bias,
                                             int64_t batch_index, int64_t spatial_index, int64_t in_c, int64_t in_h,
                                             int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                             int64_t out_h, int64_t out_w, int64_t stride_h, int64_t stride_w,
                                             int64_t pad_h, int64_t pad_w, float* input_patch, float* output);

void PackDirectConvWeightsOc8Fp32(const float* weight, const float* bias, int64_t out_c, int64_t patch_size,
                                  std::vector<float>* packed_weight, std::vector<float>* packed_bias);

int32_t ComputeDirectConv2DOc8X86Fp32(const float* input, const float* packed_weight_oc8, const float* packed_bias_oc8,
                                      const float* weight, const float* bias, int64_t batch, int64_t in_c,
                                      int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, float* output);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_DIRECT_CONV_FP32_H
