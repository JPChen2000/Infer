#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/direct_conv_fp32.h"

namespace {

using feather::kernel::x86::ComputeDirectConv2DX86Fp32;
using feather::kernel::x86::ComputeDirectConv2DOc8X86Fp32;
using feather::kernel::x86::PackDirectConvWeightsOc8Fp32;

std::vector<float> ReferenceDirectConv(const std::vector<float>& input, const std::vector<float>& weight,
                                       const std::vector<float>& bias, int64_t batch, int64_t in_c, int64_t in_h,
                                       int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                       int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias.empty() ? 0.0f : bias[static_cast<size_t>(oc)];
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const int64_t ih = oh * stride_h - pad_h + kh;
                                const int64_t iw = ow * stride_w - pad_w + kw;
                                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                const int64_t input_offset = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                                const int64_t weight_offset = ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                                sum += input[static_cast<size_t>(input_offset)] *
                                       weight[static_cast<size_t>(weight_offset)];
                            }
                        }
                    }
                    const int64_t output_offset = ((n * out_c + oc) * out_h + oh) * out_w + ow;
                    output[static_cast<size_t>(output_offset)] = sum;
                }
            }
        }
    }

    return output;
}

TEST(x86_direct_conv_fp32_test, HandlesPaddingStrideAndOutputChannelTail) {
    const int64_t batch = 1;
    const int64_t in_c = 2;
    const int64_t in_h = 4;
    const int64_t in_w = 5;
    const int64_t out_c = 3;
    const int64_t kernel_h = 3;
    const int64_t kernel_w = 3;
    const int64_t stride_h = 2;
    const int64_t stride_w = 1;
    const int64_t pad_h = 1;
    const int64_t pad_w = 1;
    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    const std::vector<float> input = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,

        -1, -2, -3, -4, -5,
        -6, -7, -8, -9, -10,
        -11, -12, -13, -14, -15,
        -16, -17, -18, -19, -20,
    };
    const std::vector<float> weight = {
        1, 0, -1, 2, 0, -2, 1, 0, -1,
        0, 1, 0, 0, 2, 0, 0, 1, 0,

        -1, 1, -1, 1, -1, 1, -1, 1, -1,
        2, -2, 2, -2, 2, -2, 2, -2, 2,

        0.5f, 0.0f, -0.5f, 1.0f, 0.0f, -1.0f, 0.5f, 0.0f, -0.5f,
        -1.0f, 1.0f, -1.0f, 0.0f, 0.5f, 0.0f, 1.0f, -1.0f, 1.0f,
    };
    const std::vector<float> bias = {0.5f, -1.0f, 2.0f};
    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);

    ASSERT_EQ(ComputeDirectConv2DX86Fp32(input.data(), weight.data(), bias.data(), batch, in_c, in_h, in_w, out_c,
                                         kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output.data()),
              0);

    const std::vector<float> expected = ReferenceDirectConv(input, weight, bias, batch, in_c, in_h, in_w, out_c,
                                                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    EXPECT_EQ(output, expected);
}

TEST(x86_direct_conv_fp32_test, RejectsInvalidArguments) {
    float output = 0.0f;
    EXPECT_EQ(
        ComputeDirectConv2DX86Fp32(nullptr, nullptr, nullptr, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, &output), -1);
}

TEST(x86_direct_conv_fp32_test, PackedOc8KernelHandlesFullBlocksAndTailChannels) {
    const int64_t batch = 1;
    const int64_t in_c = 3;
    const int64_t in_h = 5;
    const int64_t in_w = 5;
    const int64_t out_c = 10;
    const int64_t kernel_h = 3;
    const int64_t kernel_w = 3;
    const int64_t stride_h = 1;
    const int64_t stride_w = 1;
    const int64_t pad_h = 1;
    const int64_t pad_w = 1;
    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    const int64_t patch_size = in_c * kernel_h * kernel_w;

    std::vector<float> input(static_cast<size_t>(batch * in_c * in_h * in_w));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.25f;
    }

    std::vector<float> weight(static_cast<size_t>(out_c * patch_size));
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((static_cast<int>((i * 3) % 19) - 9)) * 0.125f;
    }

    std::vector<float> bias(static_cast<size_t>(out_c));
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(static_cast<int>(i) - 5) * 0.5f;
    }

    std::vector<float> packed_weight;
    std::vector<float> packed_bias;
    PackDirectConvWeightsOc8Fp32(weight.data(), bias.data(), out_c, patch_size, &packed_weight, &packed_bias);

    std::vector<float> output(static_cast<size_t>(batch * out_c * out_h * out_w), 0.0f);
    ASSERT_EQ(ComputeDirectConv2DOc8X86Fp32(input.data(), packed_weight.data(), packed_bias.data(), weight.data(),
                                            bias.data(), batch, in_c, in_h, in_w, out_c, kernel_h, kernel_w,
                                            stride_h, stride_w, pad_h, pad_w, output.data()),
              0);

    const std::vector<float> expected = ReferenceDirectConv(input, weight, bias, batch, in_c, in_h, in_w, out_c,
                                                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    EXPECT_EQ(output, expected);
}

}  // namespace
