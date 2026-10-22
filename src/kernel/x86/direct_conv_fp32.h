// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
