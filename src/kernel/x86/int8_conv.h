#ifndef FEATHER_KERNEL_X86_INT8_CONV_H
#define FEATHER_KERNEL_X86_INT8_CONV_H

#include <cstdint>
#include <vector>

namespace feather {
namespace kernel {
namespace x86 {

// Reorders [out_channel, patch_index] weights into [oc8_block, patch_index, lane].
// Only complete groups of eight output channels are packed; a tail is handled by
// the scalar convolution path.
void PackInt8ConvWeightsOc8(const int8_t* weight, int64_t output_channels, int64_t patch_size,
                            std::vector<int8_t>* packed);

void AccumulateInt8Oc8Maddubs(
    const int8_t* input, const int8_t* packed_weight, int64_t patch_size,
    int32_t* accumulators);

void AccumulateInt8Oc8MaddubsPairPacked(
    const int8_t* input, const int8_t* packed_weight, int64_t pair_count,
    int32_t* accumulators);

void AccumulateInt8Oc8Pairwise(const int8_t* input, const int8_t* packed_weight, int64_t patch_size,
                               int32_t* accumulators);

// Reorders complete groups of eight output channels into four-element VNNI
// dot-product groups. The final group is zero padded when the patch is not
// divisible by four; weight_sums stores the signed sum for zero-point repair.
void PackInt8ConvWeightsMaddubsPair(
    const int8_t* weight, int64_t output_channels, int64_t input_channels,
    int64_t kernel_h, int64_t kernel_w, std::vector<int8_t>* packed);

// Reorders pointwise [output_channel, input_channel] weights into pairs of
// [weight, ~weight] bytes for the AVX2 maddubs signed-INT8 transform. An odd
// input-channel tail is padded with a zero weight.
void PackInt8PointwiseWeightsMaddubsPair(
    const int8_t* weight, int64_t output_channels, int64_t input_channels,
    std::vector<int8_t>* packed);

void PackInt8ConvWeightsVnni(const int8_t* weight, int64_t output_channels, int64_t patch_size,
                             std::vector<int8_t>* packed, std::vector<int32_t>* weight_sums);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_INT8_CONV_H
