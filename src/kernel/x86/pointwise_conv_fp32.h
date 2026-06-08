#ifndef FEATHER_KERNEL_X86_POINTWISE_CONV_FP32_H
#define FEATHER_KERNEL_X86_POINTWISE_CONV_FP32_H

#include <cstdint>
#include <vector>

namespace feather {
namespace kernel {
namespace x86 {

int32_t ComputePointwiseConv2DX86Fp32(const float* input, const float* weight, const float* bias, int64_t batch,
                                      int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t stride_h,
                                      int64_t stride_w, float* output);

void ComputePointwiseConv2DOutputChannelX86Fp32(const float* input, const float* weight, const float* bias,
                                                int64_t batch_index, int64_t out_channel, int64_t in_c, int64_t in_h,
                                                int64_t in_w, int64_t out_c, int64_t stride_h, int64_t stride_w,
                                                float* output);

void PackPointwiseWeightsOc8Fp32(const float* weight, const float* bias, int64_t out_c, int64_t in_c,
                                 std::vector<float>* packed_weight, std::vector<float>* packed_bias);

int32_t ComputePointwiseConvPackedOc8X86Fp32(const float* packed_input, const float* packed_weight_oc8,
                                             const float* packed_bias_oc8, const float* weight, const float* bias,
                                             int64_t batch, int64_t output_spatial, int64_t in_c, int64_t out_c,
                                             uint16_t* output);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_POINTWISE_CONV_FP32_H
