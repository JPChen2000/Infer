#include "src/kernel/x86/direct_conv_fp32.h"

#include <immintrin.h>

#include <vector>

namespace feather {
namespace kernel {
namespace x86 {

namespace {

float HorizontalAdd(__m256 vec) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vec);
    float sum = 0.0f;
    for (float value : tmp) {
        sum += value;
    }
    return sum;
}

float DotProductAvx(const float* lhs, const float* rhs, int64_t len) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 lhs_vec = _mm256_loadu_ps(lhs + i);
        const __m256 rhs_vec = _mm256_loadu_ps(rhs + i);
        acc = _mm256_fmadd_ps(lhs_vec, rhs_vec, acc);
    }
    float sum = HorizontalAdd(acc);
    for (; i < len; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

void Store8FloatsToStrided(const __m256 value, float* dst, int64_t stride) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    for (int i = 0; i < 8; ++i) {
        dst[static_cast<int64_t>(i) * stride] = tmp[i];
    }
}

}  // namespace

void ComputeDirectConv2DSpatialOutputX86Fp32(const float* input, const float* weight, const float* bias,
                                             int64_t batch_index, int64_t spatial_index, int64_t in_c, int64_t in_h,
                                             int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                             int64_t out_h, int64_t out_w, int64_t stride_h, int64_t stride_w,
                                             int64_t pad_h, int64_t pad_w, float* input_patch, float* output) {
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t oh = spatial_index / out_w;
    const int64_t ow = spatial_index % out_w;
    const int64_t ih_base = oh * stride_h - pad_h;
    const int64_t iw_base = ow * stride_w - pad_w;

    int64_t patch_index = 0;
    for (int64_t ic = 0; ic < in_c; ++ic) {
        const float* input_plane = input + ((batch_index * in_c + ic) * input_spatial);
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = ih_base + kh;
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = iw_base + kw;
                float value = 0.0f;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    value = input_plane[ih * in_w + iw];
                }
                input_patch[patch_index++] = value;
            }
        }
    }

    for (int64_t oc = 0; oc < out_c; ++oc) {
        const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
        const float* weight_oc = weight + oc * patch_size;
        output[((batch_index * out_c + oc) * output_spatial) + spatial_index] =
            bias_value + DotProductAvx(input_patch, weight_oc, patch_size);
    }
}

void PackDirectConvWeightsOc8Fp32(const float* weight, const float* bias, int64_t out_c, int64_t patch_size,
                                  std::vector<float>* packed_weight, std::vector<float>* packed_bias) {
    const int64_t oc8_blocks = out_c / 8;
    packed_weight->assign(static_cast<size_t>(oc8_blocks * patch_size * 8), 0.0f);
    packed_bias->assign(static_cast<size_t>(oc8_blocks * 8), 0.0f);

    float* packed_weight_ptr = packed_weight->data();
    float* packed_bias_ptr = packed_bias->data();
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int lane = 0; lane < 8; ++lane) {
            const int64_t oc = block * 8 + lane;
            packed_bias_ptr[block * 8 + lane] = bias != nullptr ? bias[oc] : 0.0f;
        }
        for (int64_t k = 0; k < patch_size; ++k) {
            for (int lane = 0; lane < 8; ++lane) {
                const int64_t oc = block * 8 + lane;
                packed_weight_ptr[(block * patch_size + k) * 8 + lane] = weight[oc * patch_size + k];
            }
        }
    }
}

int32_t ComputeDirectConv2DX86Fp32(const float* input, const float* weight, const float* bias, int64_t batch,
                                   int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_h,
                                   int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
                                   int64_t pad_w, float* output) {
    if (input == nullptr || weight == nullptr || output == nullptr || batch <= 0 || in_c <= 0 || in_h <= 0 ||
        in_w <= 0 || out_c <= 0 || kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 ||
        pad_w < 0) {
        return -1;
    }

    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        return -1;
    }

    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    std::vector<float> input_patch(static_cast<size_t>(patch_size));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t spatial_idx = 0; spatial_idx < output_spatial; ++spatial_idx) {
            ComputeDirectConv2DSpatialOutputX86Fp32(input, weight, bias, n, spatial_idx, in_c, in_h, in_w, out_c,
                                                    kernel_h, kernel_w, out_h, out_w, stride_h, stride_w, pad_h,
                                                    pad_w, input_patch.data(), output);
        }
    }
    return 0;
}

int32_t ComputeDirectConv2DOc8X86Fp32(const float* input, const float* packed_weight_oc8, const float* packed_bias_oc8,
                                      const float* weight, const float* bias, int64_t batch, int64_t in_c,
                                      int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, float* output) {
    if (input == nullptr || packed_weight_oc8 == nullptr || packed_bias_oc8 == nullptr || weight == nullptr ||
        output == nullptr || batch <= 0 || in_c <= 0 || in_h <= 0 || in_w <= 0 || out_c <= 0 || kernel_h <= 0 ||
        kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0) {
        return -1;
    }

    const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        return -1;
    }

    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;
    std::vector<float> input_patch(static_cast<size_t>(patch_size));

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t spatial_idx = 0; spatial_idx < output_spatial; ++spatial_idx) {
            ComputeDirectConv2DSpatialOutputX86Fp32(input, weight, bias, n, spatial_idx, in_c, in_h, in_w, 0,
                                                    kernel_h, kernel_w, out_h, out_w, stride_h, stride_w, pad_h,
                                                    pad_w, input_patch.data(), output);

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                const float* block_weight = packed_weight_oc8 + block * patch_size * 8;
                for (int64_t k = 0; k < patch_size; ++k) {
                    const __m256 weight_vec = _mm256_loadu_ps(block_weight + k * 8);
                    const __m256 input_vec = _mm256_set1_ps(input_patch[static_cast<size_t>(k)]);
                    acc = _mm256_fmadd_ps(input_vec, weight_vec, acc);
                }
                Store8FloatsToStrided(acc, output + ((n * out_c + block * 8) * output_spatial) + spatial_idx,
                                      output_spatial);
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                const float* weight_oc = weight + oc * patch_size;
                output[((n * out_c + oc) * output_spatial) + spatial_idx] =
                    bias_value + DotProductAvx(input_patch.data(), weight_oc, patch_size);
            }
        }
    }
    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
