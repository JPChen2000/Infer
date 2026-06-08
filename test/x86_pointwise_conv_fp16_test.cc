#include <gtest/gtest.h>

#include <vector>

#include "src/kernel/x86/pointwise_conv_fp32.h"
#include "util/fp16.h"

namespace {

using feather::FloatToHalf;
using feather::kernel::x86::ComputePointwiseConvPackedOc8X86Fp32;
using feather::kernel::x86::PackPointwiseWeightsOc8Fp32;

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

TEST(x86_pointwise_conv_fp16_test, PackedOc8KernelHandlesFullBlocksAndTailChannels) {
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

    std::vector<uint16_t> output(static_cast<size_t>(batch * out_c * output_spatial), 0u);
    ASSERT_EQ(ComputePointwiseConvPackedOc8X86Fp32(packed_input.data(), packed_weight.data(), packed_bias.data(),
                                                   weight.data(), bias.data(), batch, output_spatial, in_c, out_c,
                                                   output.data()),
              0);

    const std::vector<float> expected =
        ReferencePointwiseConvFromPackedInput(packed_input, weight, bias, batch, output_spatial, in_c, out_c);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], FloatToHalf(expected[i]));
    }
}

}  // namespace
