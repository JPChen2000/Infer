#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/pointwise_conv_fp32.h"

namespace {

using feather::kernel::x86::ComputePointwiseConv2DX86Fp32;
using feather::kernel::x86::ComputePointwiseConvPackedOc8NhwcX86Fp32;
using feather::kernel::x86::ComputePointwiseConvPackedOc8X86Fp32;
using feather::kernel::x86::PackPointwiseWeightsOc8Fp32;

std::vector<float> ReferencePointwiseConv(const std::vector<float>& input, const std::vector<float>& weight,
                                          const std::vector<float>& bias, int64_t batch, int64_t in_c, int64_t in_h,
                                          int64_t in_w, int64_t out_c, int64_t stride_h, int64_t stride_w) {
    const int64_t out_h = in_h / stride_h;
    const int64_t out_w = in_w / stride_w;
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                const int64_t ih = oh * stride_h;
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    const int64_t iw = ow * stride_w;
                    float sum = bias.empty() ? 0.0f : bias[static_cast<size_t>(oc)];
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        const int64_t input_offset = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                        sum += input[static_cast<size_t>(input_offset)] * weight[static_cast<size_t>(oc * in_c + ic)];
                    }
                    const int64_t output_offset = ((n * out_c + oc) * out_h + oh) * out_w + ow;
                    output[static_cast<size_t>(output_offset)] = sum;
                }
            }
        }
    }
    return output;
}

std::vector<float> ReferencePointwiseConvFromPackedInput(const std::vector<float>& packed_input,
                                                         const std::vector<float>& weight,
                                                         const std::vector<float>& bias, int64_t batch,
                                                         int64_t output_spatial, int64_t in_c, int64_t out_c) {
    std::vector<float> output(static_cast<size_t>(batch * out_c * output_spatial), 0.0f);
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            const float bias_value = bias.empty() ? 0.0f : bias[static_cast<size_t>(oc)];
            for (int64_t spatial_idx = 0; spatial_idx < output_spatial; ++spatial_idx) {
                const float* input_row = packed_input.data() + ((n * output_spatial + spatial_idx) * in_c);
                const float* weight_row = weight.data() + oc * in_c;
                float sum = bias_value;
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    sum += input_row[ic] * weight_row[ic];
                }
                output[static_cast<size_t>(((n * out_c + oc) * output_spatial) + spatial_idx)] = sum;
            }
        }
    }
    return output;
}

std::vector<float> ReferencePointwiseConvFromPackedInputNhwc(const std::vector<float>& packed_input,
                                                             const std::vector<float>& weight,
                                                             const std::vector<float>& bias, int64_t batch,
                                                             int64_t output_spatial, int64_t in_c, int64_t out_c) {
    std::vector<float> output(static_cast<size_t>(batch * output_spatial * out_c), 0.0f);
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t spatial_idx = 0; spatial_idx < output_spatial; ++spatial_idx) {
            const float* input_row = packed_input.data() + ((n * output_spatial + spatial_idx) * in_c);
            float* out_row = output.data() + ((n * output_spatial + spatial_idx) * out_c);
            for (int64_t oc = 0; oc < out_c; ++oc) {
                const float* weight_row = weight.data() + oc * in_c;
                float sum = bias.empty() ? 0.0f : bias[static_cast<size_t>(oc)];
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    sum += input_row[ic] * weight_row[ic];
                }
                out_row[oc] = sum;
            }
        }
    }
    return output;
}

TEST(x86_pointwise_conv_fp32_test, HandlesBatchStrideAndOutputChannelTail) {
    const int64_t batch = 2;
    const int64_t in_c = 3;
    const int64_t in_h = 4;
    const int64_t in_w = 4;
    const int64_t out_c = 5;
    const int64_t stride_h = 2;
    const int64_t stride_w = 2;
    const int64_t out_h = in_h / stride_h;
    const int64_t out_w = in_w / stride_w;

    const std::vector<float> input = {
        1, 2, 3, 4,  5, 6, 7, 8,   9, 10, 11, 12, 13, 14, 15, 16,
        2, 3, 4, 5,  6, 7, 8, 9,   10, 11, 12, 13, 14, 15, 16, 17,
        -1, -2, -3, -4,  -5, -6, -7, -8,  -9, -10, -11, -12, -13, -14, -15, -16,

        3, 1, 4, 1,  5, 9, 2, 6,   5, 3, 5, 8,   9, 7, 9, 3,
        8, 4, 6, 2,  6, 4, 3, 3,   8, 3, 2, 7,   9, 5, 0, 2,
        1, 1, 2, 3,  5, 8, 13, 21, 34, 55, 89, 144, 1, 0, 1, 0,
    };
    const std::vector<float> weight = {
        1.0f, 0.5f, -1.0f,
        -2.0f, 1.0f, 0.25f,
        0.0f, -1.0f, 2.0f,
        1.5f, 1.5f, 1.5f,
        -0.25f, 0.75f, -1.25f,
    };
    const std::vector<float> bias = {0.5f, -1.0f, 2.0f, 0.0f, -3.0f};
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    ASSERT_EQ(ComputePointwiseConv2DX86Fp32(input.data(), weight.data(), bias.data(), batch, in_c, in_h, in_w, out_c,
                                            stride_h, stride_w, output.data()),
              0);

    const std::vector<float> expected =
        ReferencePointwiseConv(input, weight, bias, batch, in_c, in_h, in_w, out_c, stride_h, stride_w);
    EXPECT_EQ(output, expected);
}

TEST(x86_pointwise_conv_fp32_test, RejectsInvalidArguments) {
    float output = 0.0f;
    EXPECT_EQ(ComputePointwiseConv2DX86Fp32(nullptr, nullptr, nullptr, 1, 1, 1, 1, 1, 1, 1, &output), -1);
}

TEST(x86_pointwise_conv_fp32_test, PackedOc8KernelHandlesFullBlocksAndTailChannels) {
    const int64_t batch = 2;
    const int64_t output_spatial = 5;
    const int64_t in_c = 3;
    const int64_t out_c = 10;

    std::vector<float> packed_input(static_cast<size_t>(batch * output_spatial * in_c));
    for (size_t i = 0; i < packed_input.size(); ++i) {
        packed_input[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.25f;
    }

    std::vector<float> weight(static_cast<size_t>(out_c * in_c));
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((static_cast<int>((i * 5) % 19) - 9)) * 0.125f;
    }

    std::vector<float> bias(static_cast<size_t>(out_c));
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(static_cast<int>(i) - 4) * 0.375f;
    }

    std::vector<float> packed_weight;
    std::vector<float> packed_bias;
    PackPointwiseWeightsOc8Fp32(weight.data(), bias.data(), out_c, in_c, &packed_weight, &packed_bias);

    std::vector<float> output(static_cast<size_t>(batch * out_c * output_spatial), 0.0f);
    ASSERT_EQ(ComputePointwiseConvPackedOc8X86Fp32(packed_input.data(), packed_weight.data(), packed_bias.data(),
                                                   weight.data(), bias.data(), batch, output_spatial, in_c, out_c,
                                                   output.data()),
              0);

    const std::vector<float> expected =
        ReferencePointwiseConvFromPackedInput(packed_input, weight, bias, batch, output_spatial, in_c, out_c);
    EXPECT_EQ(output, expected);
}

TEST(x86_pointwise_conv_fp32_test, PackedOc8NhwcKernelHandlesFullBlocksAndTailChannels) {
    const int64_t batch = 2;
    const int64_t output_spatial = 5;
    const int64_t in_c = 3;
    const int64_t out_c = 10;

    std::vector<float> packed_input(static_cast<size_t>(batch * output_spatial * in_c));
    for (size_t i = 0; i < packed_input.size(); ++i) {
        packed_input[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.125f;
    }

    std::vector<float> weight(static_cast<size_t>(out_c * in_c));
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((static_cast<int>((i * 3) % 17) - 8)) * 0.25f;
    }

    std::vector<float> bias(static_cast<size_t>(out_c));
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(static_cast<int>(i) - 5) * 0.5f;
    }

    std::vector<float> packed_weight;
    std::vector<float> packed_bias;
    PackPointwiseWeightsOc8Fp32(weight.data(), bias.data(), out_c, in_c, &packed_weight, &packed_bias);

    std::vector<float> output(static_cast<size_t>(batch * output_spatial * out_c), 0.0f);
    ASSERT_EQ(ComputePointwiseConvPackedOc8NhwcX86Fp32(
                  packed_input.data(), packed_weight.data(), packed_bias.data(), weight.data(), bias.data(), batch,
                  output_spatial, in_c, out_c, output.data()),
              0);

    const std::vector<float> expected =
        ReferencePointwiseConvFromPackedInputNhwc(packed_input, weight, bias, batch, output_spatial, in_c, out_c);
    EXPECT_EQ(output, expected);
}

}  // namespace
