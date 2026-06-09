#include "src/kernel/conv2d.h"

#include <immintrin.h>

#include <algorithm>
#include <future>
#include <vector>

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "src/kernel/x86/direct_conv_fp32.h"
#include "src/kernel/x86/pointwise_conv_fp32.h"
#include "util/thread_pool_nv.h"
#include "util/threading.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

size_t GetConvThreadCount(int64_t total_work_items) {
    return ThreadCountForWorkItems(total_work_items);
}

ThreadPoolNv& GetConvThreadPool() {
    static ThreadPoolNv pool(DefaultThreadCount());
    return pool;
}

template <typename Fn>
void ParallelForWorkItems(int64_t total_work_items, Fn&& fn) {
    const size_t thread_count = GetConvThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetConvThreadPool();
    std::vector<std::future<int>> futures;
    futures.reserve(thread_count);

    const int64_t chunk_size =
        (total_work_items + static_cast<int64_t>(thread_count) - 1) / static_cast<int64_t>(thread_count);
    for (size_t tid = 0; tid < thread_count; ++tid) {
        const int64_t begin = static_cast<int64_t>(tid) * chunk_size;
        const int64_t end = std::min(total_work_items, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.emplace_back(pool.enqueue([begin, end, &fn](int) {
            fn(begin, end);
            return 0;
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

bool CanUseFastConv2DPath(const feather::operators::Conv2dParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return false;
    }
    return param->input->dims().size() == 4 && param->w->dims().size() == 4 && param->group == 1 &&
           param->dilation_h == 1 && param->dilation_w == 1 &&
           NormalizeDataLayout(param->input->layout()) == DataLayout::NCHW;
}

bool IsPointwiseConv2D(const feather::operators::Conv2dParam* param) {
    if (!CanUseFastConv2DPath(param)) {
        return false;
    }
    return param->w->dims()[2] == 1 && param->w->dims()[3] == 1 && param->pad_h == 0 && param->pad_w == 0;
}

bool CanUsePointwiseConv2DNhwcPath(const feather::operators::Conv2dParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return false;
    }
    return param->input->dims().size() == 4 && param->w->dims().size() == 4 &&
           NormalizeDataLayout(param->input->layout()) == DataLayout::NHWC && param->group == 1 &&
           param->dilation_h == 1 && param->dilation_w == 1 && param->w->dims()[2] == 1 && param->w->dims()[3] == 1 &&
           param->pad_h == 0 && param->pad_w == 0;
}

bool CanUseNhwcToNchwFastConv2DPath(const feather::operators::Conv2dParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return false;
    }
    return param->input->dims().size() == 4 && param->w->dims().size() == 4 &&
           NormalizeDataLayout(param->input->layout()) == DataLayout::NHWC && param->group == 1 &&
           param->dilation_h == 1 && param->dilation_w == 1;
}

template <typename T>
void ConvertImageLayout4D(const T* src, DataLayout src_layout, T* dst, DataLayout dst_layout, const ImageShape4D& shape) {
    const DataLayout normalized_src = NormalizeDataLayout(src_layout);
    const DataLayout normalized_dst = NormalizeDataLayout(dst_layout);
    if (src == nullptr || dst == nullptr || normalized_src == normalized_dst) {
        if (src != nullptr && dst != nullptr && src != dst && normalized_src == normalized_dst) {
            std::copy_n(src, static_cast<size_t>(shape.n * shape.c * shape.h * shape.w), dst);
        }
        return;
    }

    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t c = 0; c < shape.c; ++c) {
            for (int64_t h = 0; h < shape.h; ++h) {
                for (int64_t w = 0; w < shape.w; ++w) {
                    const int64_t src_offset = OffsetForImage4D(normalized_src, n, c, h, w, shape.c, shape.h, shape.w);
                    const int64_t dst_offset = OffsetForImage4D(normalized_dst, n, c, h, w, shape.c, shape.h, shape.w);
                    dst[dst_offset] = src[src_offset];
                }
            }
        }
    }
}

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

void Store8FloatsToHalfStrided(const __m256 value, uint16_t* dst, int64_t stride) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    for (int i = 0; i < 8; ++i) {
        dst[static_cast<int64_t>(i) * stride] = FloatToHalf(tmp[i]);
    }
}

void Store8FloatsToStrided(const __m256 value, float* dst, int64_t stride) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    for (int i = 0; i < 8; ++i) {
        dst[static_cast<int64_t>(i) * stride] = tmp[i];
    }
}

void ConvertHalfRowToFloat(const uint16_t* src, float* dst, int64_t len) {
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m128i half_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        const __m256 float_vec = _mm256_cvtph_ps(half_vec);
        _mm256_storeu_ps(dst + i, float_vec);
    }
    for (; i < len; ++i) {
        dst[i] = HalfToFloat(src[i]);
    }
}

bool CanUseDirectConvOc8Kernel(const feather::operators::Conv2dParam* param) {
    if (!CanUseFastConv2DPath(param) || IsPointwiseConv2D(param)) {
        return false;
    }
    return param->w->dims()[0] >= 8;
}

bool CanUseDirectConv3x3SpecializedKernel(const feather::operators::Conv2dParam* param) {
    if (!CanUseDirectConvOc8Kernel(param)) {
        return false;
    }
    return param->w->dims()[2] == 3 && param->w->dims()[3] == 3 && param->pad_h == 1 && param->pad_w == 1 &&
           (param->stride_h == 1 || param->stride_h == 2) && param->stride_h == param->stride_w;
}

bool CanUseWinograd3x3Kernel(const feather::operators::Conv2dParam* param) {
    (void)param;
    // Winograd stays on disk for follow-up tuning, but the current version regresses
    // end-to-end YOLO latency on x86, so we keep it disabled for now.
    return false;
}

void PackDirectConvWeightsOc8(const float* weight, const float* bias, int64_t out_c, int64_t patch_size,
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

void TransformWinogradWeight3x3(const float* g, float* u) {
    float temp[12];
    for (int col = 0; col < 3; ++col) {
        const float g0 = g[col];
        const float g1 = g[3 + col];
        const float g2 = g[6 + col];
        temp[col] = g0;
        temp[3 + col] = 0.5f * (g0 + g1 + g2);
        temp[6 + col] = 0.5f * (g0 - g1 + g2);
        temp[9 + col] = g2;
    }
    for (int row = 0; row < 4; ++row) {
        const float t0 = temp[row * 3 + 0];
        const float t1 = temp[row * 3 + 1];
        const float t2 = temp[row * 3 + 2];
        u[row * 4 + 0] = t0;
        u[row * 4 + 1] = 0.5f * (t0 + t1 + t2);
        u[row * 4 + 2] = 0.5f * (t0 - t1 + t2);
        u[row * 4 + 3] = t2;
    }
}

void TransformWinogradInput4x4(const float* d, float* v) {
    float temp[16];
    for (int col = 0; col < 4; ++col) {
        const float d0 = d[col];
        const float d1 = d[4 + col];
        const float d2 = d[8 + col];
        const float d3 = d[12 + col];
        temp[col] = d0 - d2;
        temp[4 + col] = d1 + d2;
        temp[8 + col] = d2 - d1;
        temp[12 + col] = d1 - d3;
    }
    for (int row = 0; row < 4; ++row) {
        const float t0 = temp[row * 4 + 0];
        const float t1 = temp[row * 4 + 1];
        const float t2 = temp[row * 4 + 2];
        const float t3 = temp[row * 4 + 3];
        v[row * 4 + 0] = t0 - t2;
        v[row * 4 + 1] = t1 + t2;
        v[row * 4 + 2] = t2 - t1;
        v[row * 4 + 3] = t1 - t3;
    }
}

void TransformWinogradOutput2x2(const float* m, float* y) {
    float temp[8];
    for (int col = 0; col < 4; ++col) {
        const float m0 = m[col];
        const float m1 = m[4 + col];
        const float m2 = m[8 + col];
        const float m3 = m[12 + col];
        temp[col] = m0 + m1 + m2;
        temp[4 + col] = m1 - m2 - m3;
    }
    for (int row = 0; row < 2; ++row) {
        const float t0 = temp[row * 4 + 0];
        const float t1 = temp[row * 4 + 1];
        const float t2 = temp[row * 4 + 2];
        const float t3 = temp[row * 4 + 3];
        y[row * 2 + 0] = t0 + t1 + t2;
        y[row * 2 + 1] = t1 - t2 - t3;
    }
}

void PackWinogradWeights3x3Fp32(const float* weight, int64_t out_c, int64_t in_c, std::vector<float>* packed_oc8,
                                std::vector<float>* transformed_weight) {
    const int64_t alpha2 = 16;
    const int64_t oc8_blocks = out_c / 8;
    packed_oc8->assign(static_cast<size_t>(oc8_blocks * in_c * alpha2 * 8), 0.0f);
    transformed_weight->assign(static_cast<size_t>(out_c * in_c * alpha2), 0.0f);

    float u[16];
    for (int64_t oc = 0; oc < out_c; ++oc) {
        for (int64_t ic = 0; ic < in_c; ++ic) {
            const float* g = weight + ((oc * in_c + ic) * 9);
            TransformWinogradWeight3x3(g, u);
            float* scalar_dst = transformed_weight->data() + ((oc * in_c + ic) * alpha2);
            std::copy_n(u, alpha2, scalar_dst);
            if (oc < oc8_blocks * 8) {
                const int64_t block = oc / 8;
                const int64_t lane = oc % 8;
                float* packed_dst = packed_oc8->data() + ((block * in_c + ic) * alpha2 * 8);
                for (int64_t t = 0; t < alpha2; ++t) {
                    packed_dst[t * 8 + lane] = u[t];
                }
            }
        }
    }
}

void ComputePointwiseConv2DKernelFastX86Fp32(const feather::operators::Conv2dParam* param, const float* input,
                                             const float* weight, const float* bias, float* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t total_work_items = batch * out_c;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / out_c;
            const int64_t oc = work_index % out_c;
            x86::ComputePointwiseConv2DOutputChannelX86Fp32(input, weight, bias, n, oc, in_c, in_h, in_w, out_c,
                                                            param->stride_h, param->stride_w, output);
        }
    });
}

void PackPointwiseInputFp16ToFloat(const feather::operators::Conv2dParam* param, const uint16_t* input,
                                   std::vector<float>* packed_input) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;

    const size_t packed_size = static_cast<size_t>(batch * output_spatial * in_c);
    if (packed_input->size() != packed_size) {
        packed_input->resize(packed_size);
    }
    float* packed = packed_input->data();

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih = oh * param->stride_h;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw = ow * param->stride_w;
                float* packed_row = packed + ((n * output_spatial + oh * out_w + ow) * in_c);
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    packed_row[ic] = HalfToFloat(input[((n * in_c + ic) * input_spatial) + ih * in_w + iw]);
                }
            }
        }
    }
}

void PackPointwiseInputFp32(const feather::operators::Conv2dParam* param, const float* input,
                            std::vector<float>* packed_input) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;

    const size_t packed_size = static_cast<size_t>(batch * output_spatial * in_c);
    if (packed_input->size() != packed_size) {
        packed_input->resize(packed_size);
    }
    float* packed = packed_input->data();

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih = oh * param->stride_h;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw = ow * param->stride_w;
                float* packed_row = packed + ((n * output_spatial + oh * out_w + ow) * in_c);
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    packed_row[ic] = input[((n * in_c + ic) * input_spatial) + ih * in_w + iw];
                }
            }
        }
    }
}

void PackPointwiseInputNhwcFp16ToFloat(const feather::operators::Conv2dParam* param, const uint16_t* input,
                                       std::vector<float>* packed_input) {
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        packed_input->clear();
        return;
    }

    const int64_t batch = input_shape.n;
    const int64_t in_c = input_shape.c;
    const int64_t out_h = output_shape.h;
    const int64_t out_w = output_shape.w;
    const int64_t input_h = input_shape.h;
    const int64_t input_w = input_shape.w;
    const int64_t output_spatial = out_h * out_w;

    const size_t packed_size = static_cast<size_t>(batch * output_spatial * in_c);
    if (packed_input->size() != packed_size) {
        packed_input->resize(packed_size);
    }
    float* packed = packed_input->data();

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih = oh * param->stride_h;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw = ow * param->stride_w;
                const int64_t input_offset = ((n * input_h + ih) * input_w + iw) * in_c;
                float* packed_row = packed + ((n * output_spatial + oh * out_w + ow) * in_c);
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    packed_row[ic] = HalfToFloat(input[input_offset + ic]);
                }
            }
        }
    }
}

void PackPointwiseInputNhwcFp32(const feather::operators::Conv2dParam* param, const float* input,
                                std::vector<float>* packed_input) {
    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        packed_input->clear();
        return;
    }

    const int64_t batch = input_shape.n;
    const int64_t in_c = input_shape.c;
    const int64_t out_h = output_shape.h;
    const int64_t out_w = output_shape.w;
    const int64_t input_h = input_shape.h;
    const int64_t input_w = input_shape.w;
    const int64_t output_spatial = out_h * out_w;

    const size_t packed_size = static_cast<size_t>(batch * output_spatial * in_c);
    if (packed_input->size() != packed_size) {
        packed_input->resize(packed_size);
    }
    float* packed = packed_input->data();

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih = oh * param->stride_h;
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw = ow * param->stride_w;
                const int64_t input_offset = ((n * input_h + ih) * input_w + iw) * in_c;
                float* packed_row = packed + ((n * output_spatial + oh * out_w + ow) * in_c);
                std::copy_n(input + input_offset, static_cast<size_t>(in_c), packed_row);
            }
        }
    }
}

void ComputePointwiseConv2DKernelPackedX86Fp32(const feather::operators::Conv2dParam* param, const float* packed_input,
                                               const float* weight, const float* bias, uint16_t* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t out_c = param->w->dims()[0];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t output_spatial = out_h * out_w;
    const int64_t total_work_items = batch * out_c;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / out_c;
            const int64_t oc = work_index % out_c;
            uint16_t* out_base = output + work_index * output_spatial;
            const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
            const float* weight_base = weight + oc * in_c;

            for (int64_t spatial_idx = 0; spatial_idx < output_spatial; ++spatial_idx) {
                const float* input_row = packed_input + ((n * output_spatial + spatial_idx) * in_c);
                out_base[spatial_idx] = FloatToHalf(bias_value + DotProductAvx(input_row, weight_base, in_c));
            }
        }
    });
}

void ComputeDirectConv2DKernelFastX86Fp32(const feather::operators::Conv2dParam* param, const float* input,
                                          const float* weight, const float* bias, float* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t total_work_items = batch * output_spatial;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> input_patch(static_cast<size_t>(patch_size));
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const int64_t oh = spatial_idx / out_w;
            const int64_t ow = spatial_idx % out_w;
            const int64_t ih_base = oh * param->stride_h - param->pad_h;
            const int64_t iw_base = ow * param->stride_w - param->pad_w;

            int64_t patch_index = 0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const int64_t plane_offset = (n * in_c + ic) * (in_h * in_w);
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih = ih_base + kh;
                    if (ih >= 0 && ih < in_h && iw_base >= 0 && iw_base + kernel_w <= in_w) {
                        std::copy_n(input + plane_offset + ih * in_w + iw_base, kernel_w,
                                    input_patch.data() + patch_index);
                        patch_index += kernel_w;
                        continue;
                    }
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        const int64_t iw = iw_base + kw;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = input[plane_offset + ih * in_w + iw];
                        }
                        input_patch[static_cast<size_t>(patch_index++)] = value;
                    }
                }
            }

            for (int64_t oc = 0; oc < out_c; ++oc) {
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                const float* weight_oc = weight + oc * patch_size;
                output[((n * out_c + oc) * output_spatial) + spatial_idx] =
                    bias_value + DotProductAvx(input_patch.data(), weight_oc, patch_size);
            }
        }
    });
}

void ComputeDirectConv2DOc8SpatialReuseX86Fp32(const feather::operators::Conv2dParam* param, const float* input,
                                               const float* packed_weight_oc8, const float* packed_bias_oc8,
                                               const float* weight, const float* bias, float* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t total_work_items = batch * output_spatial;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> input_patch(static_cast<size_t>(patch_size));
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const int64_t oh = spatial_idx / out_w;
            const int64_t ow = spatial_idx % out_w;
            const int64_t ih_base = oh * param->stride_h - param->pad_h;
            const int64_t iw_base = ow * param->stride_w - param->pad_w;

            int64_t patch_index = 0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const int64_t plane_offset = (n * in_c + ic) * input_spatial;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih = ih_base + kh;
                    if (ih >= 0 && ih < in_h && iw_base >= 0 && iw_base + kernel_w <= in_w) {
                        std::copy_n(input + plane_offset + ih * in_w + iw_base, kernel_w,
                                    input_patch.data() + patch_index);
                        patch_index += kernel_w;
                        continue;
                    }
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        const int64_t iw = iw_base + kw;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = input[plane_offset + ih * in_w + iw];
                        }
                        input_patch[static_cast<size_t>(patch_index++)] = value;
                    }
                }
            }

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                const float* block_weight = packed_weight_oc8 + block * patch_size * 8;
                for (int64_t k = 0; k < patch_size; ++k) {
                    const __m256 weight_vec = _mm256_loadu_ps(block_weight + k * 8);
                    const __m256 input_vec = _mm256_set1_ps(input_patch[static_cast<size_t>(k)]);
                    acc = _mm256_fmadd_ps(input_vec, weight_vec, acc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, acc);
                float* out_ptr = output + ((n * out_c + block * 8) * output_spatial) + spatial_idx;
                for (int lane = 0; lane < 8; ++lane) {
                    out_ptr[static_cast<int64_t>(lane) * output_spatial] = tmp[lane];
                }
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                const float* weight_oc = weight + oc * patch_size;
                output[((n * out_c + oc) * output_spatial) + spatial_idx] =
                    bias_value + DotProductAvx(input_patch.data(), weight_oc, patch_size);
            }
        }
    });
}

void ComputeDirectConv3x3Oc8SpecializedX86Fp32(const feather::operators::Conv2dParam* param, const float* input,
                                               const float* packed_weight_oc8, const float* packed_bias_oc8,
                                               const float* weight, const float* bias, float* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * 9;
    const int64_t total_work_items = batch * output_spatial;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> input_patch(static_cast<size_t>(patch_size));
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const int64_t oh = spatial_idx / out_w;
            const int64_t ow = spatial_idx % out_w;
            const int64_t ih_base = oh * param->stride_h - 1;
            const int64_t iw_base = ow * param->stride_w - 1;
            const bool interior = ih_base >= 0 && ih_base + 2 < in_h && iw_base >= 0 && iw_base + 2 < in_w;

            if (interior) {
                for (int64_t block = 0; block < oc8_blocks; ++block) {
                    __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                    const float* block_weight = packed_weight_oc8 + block * patch_size * 8;
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        const float* src = input + ((n * in_c + ic) * input_spatial) + ih_base * in_w + iw_base;
                        const float* block_ic = block_weight + ic * 9 * 8;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[0]), _mm256_loadu_ps(block_ic + 0 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[1]), _mm256_loadu_ps(block_ic + 1 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[2]), _mm256_loadu_ps(block_ic + 2 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[in_w + 0]), _mm256_loadu_ps(block_ic + 3 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[in_w + 1]), _mm256_loadu_ps(block_ic + 4 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[in_w + 2]), _mm256_loadu_ps(block_ic + 5 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[2 * in_w + 0]),
                                              _mm256_loadu_ps(block_ic + 6 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[2 * in_w + 1]),
                                              _mm256_loadu_ps(block_ic + 7 * 8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(src[2 * in_w + 2]),
                                              _mm256_loadu_ps(block_ic + 8 * 8), acc);
                    }
                    Store8FloatsToStrided(acc, output + ((n * out_c + block * 8) * output_spatial) + spatial_idx,
                                          output_spatial);
                }

                for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                    float sum = bias != nullptr ? bias[oc] : 0.0f;
                    const float* weight_oc = weight + oc * patch_size;
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        const float* src = input + ((n * in_c + ic) * input_spatial) + ih_base * in_w + iw_base;
                        const float* weight_ic = weight_oc + ic * 9;
                        sum += src[0] * weight_ic[0] + src[1] * weight_ic[1] + src[2] * weight_ic[2] +
                               src[in_w + 0] * weight_ic[3] + src[in_w + 1] * weight_ic[4] +
                               src[in_w + 2] * weight_ic[5] + src[2 * in_w + 0] * weight_ic[6] +
                               src[2 * in_w + 1] * weight_ic[7] + src[2 * in_w + 2] * weight_ic[8];
                    }
                    output[((n * out_c + oc) * output_spatial) + spatial_idx] = sum;
                }
                continue;
            }

            int64_t patch_index = 0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const int64_t plane_offset = (n * in_c + ic) * input_spatial;
                for (int64_t kh = 0; kh < 3; ++kh) {
                    const int64_t ih = ih_base + kh;
                    for (int64_t kw = 0; kw < 3; ++kw) {
                        const int64_t iw = iw_base + kw;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = input[plane_offset + ih * in_w + iw];
                        }
                        input_patch[static_cast<size_t>(patch_index++)] = value;
                    }
                }
            }

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                const float* block_weight = packed_weight_oc8 + block * patch_size * 8;
                for (int64_t k = 0; k < patch_size; ++k) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(input_patch[static_cast<size_t>(k)]),
                                          _mm256_loadu_ps(block_weight + k * 8), acc);
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
    });
}

void ComputeWinograd3x3Fp32(const feather::operators::Conv2dParam* param, const float* input,
                            const float* packed_weight_oc8, const float* transformed_weight, const float* bias,
                            float* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t tile_h = (out_h + 1) / 2;
    const int64_t tile_w = (out_w + 1) / 2;
    const int64_t total_tiles = batch * tile_h * tile_w;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;

    ParallelForWorkItems(total_tiles, [&](int64_t begin, int64_t end) {
        float input_tile[16];
        float transformed_input[16];
        float transformed_output[16];
        float output_tile[4];
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / (tile_h * tile_w);
            const int64_t tile_index = work_index % (tile_h * tile_w);
            const int64_t th = tile_index / tile_w;
            const int64_t tw = tile_index % tile_w;
            const int64_t oh_base = th * 2;
            const int64_t ow_base = tw * 2;
            const int64_t ih_base = oh_base - 1;
            const int64_t iw_base = ow_base - 1;

            std::vector<__m256> acc(static_cast<size_t>(oc8_blocks * 16), _mm256_setzero_ps());
            std::vector<float> tail_acc(static_cast<size_t>((out_c - oc_tail_begin) * 16), 0.0f);

            for (int64_t ic = 0; ic < in_c; ++ic) {
                const float* input_plane = input + ((n * in_c + ic) * input_spatial);
                for (int tile_row = 0; tile_row < 4; ++tile_row) {
                    const int64_t ih = ih_base + tile_row;
                    for (int tile_col = 0; tile_col < 4; ++tile_col) {
                        const int64_t iw = iw_base + tile_col;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = input_plane[ih * in_w + iw];
                        }
                        input_tile[tile_row * 4 + tile_col] = value;
                    }
                }
                TransformWinogradInput4x4(input_tile, transformed_input);

                for (int64_t block = 0; block < oc8_blocks; ++block) {
                    const float* block_weight = packed_weight_oc8 + ((block * in_c + ic) * 16 * 8);
                    for (int t = 0; t < 16; ++t) {
                        acc[static_cast<size_t>(block * 16 + t)] =
                            _mm256_fmadd_ps(_mm256_set1_ps(transformed_input[t]),
                                            _mm256_loadu_ps(block_weight + t * 8),
                                            acc[static_cast<size_t>(block * 16 + t)]);
                    }
                }

                for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                    const float* weight_ic = transformed_weight + ((oc * in_c + ic) * 16);
                    float* tail = tail_acc.data() + ((oc - oc_tail_begin) * 16);
                    for (int t = 0; t < 16; ++t) {
                        tail[t] += transformed_input[t] * weight_ic[t];
                    }
                }
            }

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                const __m256 bias_vec =
                    _mm256_set1_ps(bias != nullptr ? 0.0f : 0.0f);  // bias added after inverse transform
                (void)bias_vec;
                alignas(32) float tmp[16][8];
                for (int t = 0; t < 16; ++t) {
                    _mm256_store_ps(tmp[t], acc[static_cast<size_t>(block * 16 + t)]);
                }
                for (int lane = 0; lane < 8; ++lane) {
                    for (int t = 0; t < 16; ++t) {
                        transformed_output[t] = tmp[t][lane];
                    }
                    TransformWinogradOutput2x2(transformed_output, output_tile);
                    const int64_t oc = block * 8 + lane;
                    const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                    for (int tile_row = 0; tile_row < 2; ++tile_row) {
                        const int64_t oh = oh_base + tile_row;
                        if (oh >= out_h) {
                            continue;
                        }
                        for (int tile_col = 0; tile_col < 2; ++tile_col) {
                            const int64_t ow = ow_base + tile_col;
                            if (ow >= out_w) {
                                continue;
                            }
                            output[((n * out_c + oc) * output_spatial) + oh * out_w + ow] =
                                output_tile[tile_row * 2 + tile_col] + bias_value;
                        }
                    }
                }
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                std::copy_n(tail_acc.data() + ((oc - oc_tail_begin) * 16), 16, transformed_output);
                TransformWinogradOutput2x2(transformed_output, output_tile);
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                for (int tile_row = 0; tile_row < 2; ++tile_row) {
                    const int64_t oh = oh_base + tile_row;
                    if (oh >= out_h) {
                        continue;
                    }
                    for (int tile_col = 0; tile_col < 2; ++tile_col) {
                        const int64_t ow = ow_base + tile_col;
                        if (ow >= out_w) {
                            continue;
                        }
                        output[((n * out_c + oc) * output_spatial) + oh * out_w + ow] =
                            output_tile[tile_row * 2 + tile_col] + bias_value;
                    }
                }
            }
        }
    });
}

void ComputeWinograd3x3Fp16(const feather::operators::Conv2dParam* param, const uint16_t* input,
                            const float* packed_weight_oc8, const float* transformed_weight, const float* bias,
                            uint16_t* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t input_spatial = in_h * in_w;
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t output_spatial = out_h * out_w;
    const int64_t tile_h = (out_h + 1) / 2;
    const int64_t tile_w = (out_w + 1) / 2;
    const int64_t total_tiles = batch * tile_h * tile_w;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;

    ParallelForWorkItems(total_tiles, [&](int64_t begin, int64_t end) {
        float input_tile[16];
        float transformed_input[16];
        float transformed_output[16];
        float output_tile[4];
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / (tile_h * tile_w);
            const int64_t tile_index = work_index % (tile_h * tile_w);
            const int64_t th = tile_index / tile_w;
            const int64_t tw = tile_index % tile_w;
            const int64_t oh_base = th * 2;
            const int64_t ow_base = tw * 2;
            const int64_t ih_base = oh_base - 1;
            const int64_t iw_base = ow_base - 1;

            std::vector<__m256> acc(static_cast<size_t>(oc8_blocks * 16), _mm256_setzero_ps());
            std::vector<float> tail_acc(static_cast<size_t>((out_c - oc_tail_begin) * 16), 0.0f);

            for (int64_t ic = 0; ic < in_c; ++ic) {
                const uint16_t* input_plane = input + ((n * in_c + ic) * input_spatial);
                for (int tile_row = 0; tile_row < 4; ++tile_row) {
                    const int64_t ih = ih_base + tile_row;
                    for (int tile_col = 0; tile_col < 4; ++tile_col) {
                        const int64_t iw = iw_base + tile_col;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = HalfToFloat(input_plane[ih * in_w + iw]);
                        }
                        input_tile[tile_row * 4 + tile_col] = value;
                    }
                }
                TransformWinogradInput4x4(input_tile, transformed_input);

                for (int64_t block = 0; block < oc8_blocks; ++block) {
                    const float* block_weight = packed_weight_oc8 + ((block * in_c + ic) * 16 * 8);
                    for (int t = 0; t < 16; ++t) {
                        acc[static_cast<size_t>(block * 16 + t)] =
                            _mm256_fmadd_ps(_mm256_set1_ps(transformed_input[t]),
                                            _mm256_loadu_ps(block_weight + t * 8),
                                            acc[static_cast<size_t>(block * 16 + t)]);
                    }
                }

                for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                    const float* weight_ic = transformed_weight + ((oc * in_c + ic) * 16);
                    float* tail = tail_acc.data() + ((oc - oc_tail_begin) * 16);
                    for (int t = 0; t < 16; ++t) {
                        tail[t] += transformed_input[t] * weight_ic[t];
                    }
                }
            }

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                alignas(32) float tmp[16][8];
                for (int t = 0; t < 16; ++t) {
                    _mm256_store_ps(tmp[t], acc[static_cast<size_t>(block * 16 + t)]);
                }
                for (int lane = 0; lane < 8; ++lane) {
                    for (int t = 0; t < 16; ++t) {
                        transformed_output[t] = tmp[t][lane];
                    }
                    TransformWinogradOutput2x2(transformed_output, output_tile);
                    const int64_t oc = block * 8 + lane;
                    const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                    for (int tile_row = 0; tile_row < 2; ++tile_row) {
                        const int64_t oh = oh_base + tile_row;
                        if (oh >= out_h) {
                            continue;
                        }
                        for (int tile_col = 0; tile_col < 2; ++tile_col) {
                            const int64_t ow = ow_base + tile_col;
                            if (ow >= out_w) {
                                continue;
                            }
                            output[((n * out_c + oc) * output_spatial) + oh * out_w + ow] =
                                FloatToHalf(output_tile[tile_row * 2 + tile_col] + bias_value);
                        }
                    }
                }
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                std::copy_n(tail_acc.data() + ((oc - oc_tail_begin) * 16), 16, transformed_output);
                TransformWinogradOutput2x2(transformed_output, output_tile);
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                for (int tile_row = 0; tile_row < 2; ++tile_row) {
                    const int64_t oh = oh_base + tile_row;
                    if (oh >= out_h) {
                        continue;
                    }
                    for (int tile_col = 0; tile_col < 2; ++tile_col) {
                        const int64_t ow = ow_base + tile_col;
                        if (ow >= out_w) {
                            continue;
                        }
                        output[((n * out_c + oc) * output_spatial) + oh * out_w + ow] =
                            FloatToHalf(output_tile[tile_row * 2 + tile_col] + bias_value);
                    }
                }
            }
        }
    });
}

void ComputeDirectConv2DSpatialReuseX86Fp16(const feather::operators::Conv2dParam* param, const uint16_t* input,
                                            const float* weight, const float* bias, uint16_t* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t total_work_items = batch * output_spatial;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> input_patch(static_cast<size_t>(patch_size));
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const int64_t oh = spatial_idx / out_w;
            const int64_t ow = spatial_idx % out_w;
            const int64_t ih_base = oh * param->stride_h - param->pad_h;
            const int64_t iw_base = ow * param->stride_w - param->pad_w;

            int64_t patch_index = 0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const int64_t plane_offset = (n * in_c + ic) * input_spatial;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih = ih_base + kh;
                    if (ih >= 0 && ih < in_h && iw_base >= 0 && iw_base + kernel_w <= in_w) {
                        ConvertHalfRowToFloat(input + plane_offset + ih * in_w + iw_base,
                                              input_patch.data() + patch_index, kernel_w);
                        patch_index += kernel_w;
                        continue;
                    }
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        const int64_t iw = iw_base + kw;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = HalfToFloat(input[plane_offset + ih * in_w + iw]);
                        }
                        input_patch[static_cast<size_t>(patch_index++)] = value;
                    }
                }
            }

            for (int64_t oc = 0; oc < out_c; ++oc) {
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                const float* weight_oc = weight + oc * patch_size;
                output[((n * out_c + oc) * output_spatial) + spatial_idx] =
                    FloatToHalf(bias_value + DotProductAvx(input_patch.data(), weight_oc, patch_size));
            }
        }
    });
}

void ComputeDirectConv2DSpatialReuseOc8X86Fp16(const feather::operators::Conv2dParam* param, const uint16_t* input,
                                               const float* packed_weight_oc8, const float* packed_bias_oc8,
                                               const float* weight, const float* bias, uint16_t* output) {
    const int64_t batch = param->input->dims()[0];
    const int64_t in_c = param->input->dims()[1];
    const int64_t in_h = param->input->dims()[2];
    const int64_t in_w = param->input->dims()[3];
    const int64_t out_c = param->w->dims()[0];
    const int64_t kernel_h = param->w->dims()[2];
    const int64_t kernel_w = param->w->dims()[3];
    const int64_t out_h = param->out->dims()[2];
    const int64_t out_w = param->out->dims()[3];
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    const int64_t patch_size = in_c * kernel_h * kernel_w;
    const int64_t total_work_items = batch * output_spatial;
    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        std::vector<float> input_patch(static_cast<size_t>(patch_size));
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const int64_t oh = spatial_idx / out_w;
            const int64_t ow = spatial_idx % out_w;
            const int64_t ih_base = oh * param->stride_h - param->pad_h;
            const int64_t iw_base = ow * param->stride_w - param->pad_w;

            int64_t patch_index = 0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const int64_t plane_offset = (n * in_c + ic) * input_spatial;
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih = ih_base + kh;
                    if (ih >= 0 && ih < in_h && iw_base >= 0 && iw_base + kernel_w <= in_w) {
                        ConvertHalfRowToFloat(input + plane_offset + ih * in_w + iw_base,
                                              input_patch.data() + patch_index, kernel_w);
                        patch_index += kernel_w;
                        continue;
                    }
                    for (int64_t kw = 0; kw < kernel_w; ++kw) {
                        const int64_t iw = iw_base + kw;
                        float value = 0.0f;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value = HalfToFloat(input[plane_offset + ih * in_w + iw]);
                        }
                        input_patch[static_cast<size_t>(patch_index++)] = value;
                    }
                }
            }

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                const float* block_weight = packed_weight_oc8 + block * patch_size * 8;
                for (int64_t k = 0; k < patch_size; ++k) {
                    const __m256 weight_vec = _mm256_loadu_ps(block_weight + k * 8);
                    const __m256 input_vec = _mm256_set1_ps(input_patch[static_cast<size_t>(k)]);
                    acc = _mm256_fmadd_ps(input_vec, weight_vec, acc);
                }
                Store8FloatsToHalfStrided(
                    acc, output + ((n * out_c + block * 8) * output_spatial) + spatial_idx, output_spatial);
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
                const float* weight_oc = weight + oc * patch_size;
                output[((n * out_c + oc) * output_spatial) + spatial_idx] =
                    FloatToHalf(bias_value + DotProductAvx(input_patch.data(), weight_oc, patch_size));
            }
        }
    });
}

template <DataType dtype>
std::vector<float> ConvertTensorToFloatBuffer(const std::shared_ptr<Tensor>& tensor) {
    std::vector<float> buffer(static_cast<size_t>(tensor->numel()));
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        buffer[static_cast<size_t>(i)] = TensorIO<dtype>::Read(tensor.get(), i);
    }
    return buffer;
}

template <DataType dtype>
void WriteFloatBufferToTensor(const std::vector<float>& buffer, Tensor* tensor) {
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        TensorIO<dtype>::Write(tensor, i, buffer[static_cast<size_t>(i)]);
    }
}

template <DataType dtype>
int32_t ComputeConv2DFallback(feather::operators::Conv2dParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);

    if (param->input->dims().size() == 2 && param->w->dims().size() == 2) {
        const int in_h = static_cast<int>(param->input->dims()[0]);
        const int in_w = static_cast<int>(param->input->dims()[1]);
        const int kernel_h = static_cast<int>(param->w->dims()[0]);
        const int kernel_w = static_cast<int>(param->w->dims()[1]);
        const int out_h =
            (in_h + 2 * param->pad_h - param->dilation_h * (kernel_h - 1) - 1) / param->stride_h + 1;
        const int out_w =
            (in_w + 2 * param->pad_w - param->dilation_w * (kernel_w - 1) - 1) / param->stride_w + 1;

        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float value = 0.0f;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int ih = oh * param->stride_h + kh * param->dilation_h - param->pad_h;
                        const int iw = ow * param->stride_w + kw * param->dilation_w - param->pad_w;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value += TensorIO<dtype>::Read(param->input.get(), ih * in_w + iw) *
                                     TensorIO<dtype>::Read(param->w.get(), kh * kernel_w + kw);
                        }
                    }
                }
                if (param->bias != nullptr && param->bias->IsInitialized()) {
                    value += TensorIO<dtype>::Read(param->bias.get(), oh * out_w + ow);
                }
                TensorIO<dtype>::Write(param->out.get(), oh * out_w + ow, value);
            }
        }
        return 0;
    }

    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return -1;
    }
    const DataLayout layout = NormalizeDataLayout(param->input->layout());
    const int batch = static_cast<int>(input_shape.n);
    const int in_c = static_cast<int>(input_shape.c);
    const int in_h = static_cast<int>(input_shape.h);
    const int in_w = static_cast<int>(input_shape.w);
    const int out_c = static_cast<int>(param->w->dims()[0]);
    const int kernel_c = static_cast<int>(param->w->dims()[1]);
    const int kernel_h = static_cast<int>(param->w->dims()[2]);
    const int kernel_w = static_cast<int>(param->w->dims()[3]);
    const int out_h = static_cast<int>(output_shape.h);
    const int out_w = static_cast<int>(output_shape.w);
    const int group = std::max(1, param->group);
    const int out_c_per_group = out_c / group;
    const int in_c_per_group = in_c / group;

    for (int n = 0; n < batch; ++n) {
        for (int g = 0; g < group; ++g) {
            for (int oc = 0; oc < out_c_per_group; ++oc) {
                const int global_oc = g * out_c_per_group + oc;
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float value = 0.0f;
                        for (int ic = 0; ic < in_c_per_group; ++ic) {
                            const int global_ic = g * in_c_per_group + ic;
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int ih = oh * param->stride_h + kh * param->dilation_h - param->pad_h;
                                    const int iw = ow * param->stride_w + kw * param->dilation_w - param->pad_w;
                                    if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                        continue;
                                    }
                                    const int64_t input_offset =
                                        OffsetForImage4D(layout, n, global_ic, ih, iw, in_c, in_h, in_w);
                                    const int64_t kernel_offset =
                                        ((static_cast<int64_t>(global_oc) * kernel_c + ic) * kernel_h + kh) * kernel_w + kw;
                                    value += TensorIO<dtype>::Read(param->input.get(), input_offset) *
                                             TensorIO<dtype>::Read(param->w.get(), kernel_offset);
                                }
                            }
                        }
                        const int64_t out_offset =
                            OffsetForImage4D(layout, n, global_oc, oh, ow, out_c, out_h, out_w);
                        if (param->bias != nullptr && param->bias->IsInitialized()) {
                            value += TensorIO<dtype>::Read(param->bias.get(), global_oc);
                        }
                        TensorIO<dtype>::Write(param->out.get(), out_offset, value);
                    }
                }
            }
        }
    }
    return 0;
}

}  // namespace

template <>
int32_t Conv2DKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Conv2D::FP32");
    auto* param = static_cast<feather::operators::Conv2dParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }
    param->out->set_data_type(DataType::FP32);

    if (CanUsePointwiseConv2DNhwcPath(param)) {
        const float* input = param->input->data<float>();
        const float* weight = param->w->data<float>();
        const float* bias =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
        float* output = param->out->mutable_data<float>();
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_weight_tensor_ != param->w.get()) {
            cached_direct_weight_oc8_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }

        PackPointwiseInputNhwcFp32(param, input, &cached_packed_input_buffer_);
        if (param->w->dims()[0] >= 8) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                x86::PackPointwiseWeightsOc8Fp32(weight, bias, param->w->dims()[0], param->w->dims()[1],
                                                &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
            }
        }
        const int64_t batch = param->input->dims()[0];
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            return -1;
        }
        const int64_t output_spatial = output_shape.h * output_shape.w;
        return x86::ComputePointwiseConvPackedOc8NhwcX86Fp32(
            cached_packed_input_buffer_.data(),
            cached_direct_weight_oc8_buffer_.empty() ? nullptr : cached_direct_weight_oc8_buffer_.data(),
            cached_direct_bias_oc8_buffer_.empty() ? nullptr : cached_direct_bias_oc8_buffer_.data(), weight, bias,
            batch, output_spatial, param->w->dims()[1],
            param->w->dims()[0], output);
    }

    if (CanUseNhwcToNchwFastConv2DPath(param)) {
        ImageShape4D input_shape;
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
            !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            return -1;
        }

        auto nchw_input =
            std::make_shared<Tensor>(std::vector<int64_t>{input_shape.n, input_shape.c, input_shape.h, input_shape.w});
        auto nchw_output = std::make_shared<Tensor>(
            std::vector<int64_t>{output_shape.n, param->w->dims()[0], output_shape.h, output_shape.w});
        nchw_input->set_layout(DataLayout::NCHW);
        nchw_output->set_layout(DataLayout::NCHW);

        ConvertImageLayout4D(param->input->data<float>(), param->input->layout(), nchw_input->mutable_data<float>(),
                             DataLayout::NCHW, input_shape);

        feather::operators::Conv2dParam nchw_param = *param;
        nchw_param.input = nchw_input;
        nchw_param.out = nchw_output;

        if (cached_weight_tensor_ != param->w.get()) {
            cached_direct_weight_oc8_buffer_.clear();
            cached_winograd_weight_oc8_buffer_.clear();
            cached_winograd_weight_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }

        const float* input = nchw_input->data<float>();
        const float* weight = param->w->data<float>();
        const float* bias =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
        float* output = nchw_output->mutable_data<float>();

        if (IsPointwiseConv2D(&nchw_param)) {
            if (nchw_param.w->dims()[0] >= 8) {
                PackPointwiseInputFp32(&nchw_param, input, &cached_packed_input_buffer_);
                if (cached_direct_weight_oc8_buffer_.empty()) {
                    x86::PackPointwiseWeightsOc8Fp32(
                        weight, bias, nchw_param.w->dims()[0], nchw_param.w->dims()[1],
                        &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
                }
                x86::ComputePointwiseConvPackedOc8X86Fp32(
                    cached_packed_input_buffer_.data(), cached_direct_weight_oc8_buffer_.data(),
                    cached_direct_bias_oc8_buffer_.data(), weight, bias, nchw_param.input->dims()[0],
                    nchw_param.out->dims()[2] * nchw_param.out->dims()[3], nchw_param.input->dims()[1],
                    nchw_param.w->dims()[0], output);
            } else {
                ComputePointwiseConv2DKernelFastX86Fp32(&nchw_param, input, weight, bias, output);
            }
        } else if (CanUseWinograd3x3Kernel(&nchw_param)) {
            if (cached_winograd_weight_buffer_.empty()) {
                PackWinogradWeights3x3Fp32(weight, nchw_param.w->dims()[0], nchw_param.w->dims()[1],
                                           &cached_winograd_weight_oc8_buffer_, &cached_winograd_weight_buffer_);
            }
            ComputeWinograd3x3Fp32(&nchw_param, input, cached_winograd_weight_oc8_buffer_.data(),
                                   cached_winograd_weight_buffer_.data(), bias, output);
        } else if (CanUseDirectConvOc8Kernel(&nchw_param)) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                x86::PackDirectConvWeightsOc8Fp32(
                    weight, bias, nchw_param.w->dims()[0],
                    nchw_param.w->dims()[1] * nchw_param.w->dims()[2] * nchw_param.w->dims()[3],
                    &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
            }
            if (CanUseDirectConv3x3SpecializedKernel(&nchw_param)) {
                ComputeDirectConv3x3Oc8SpecializedX86Fp32(
                    &nchw_param, input, cached_direct_weight_oc8_buffer_.data(), cached_direct_bias_oc8_buffer_.data(),
                    weight, bias, output);
            } else {
                ComputeDirectConv2DOc8SpatialReuseX86Fp32(
                    &nchw_param, input, cached_direct_weight_oc8_buffer_.data(), cached_direct_bias_oc8_buffer_.data(),
                    weight, bias, output);
            }
        } else {
            ComputeDirectConv2DKernelFastX86Fp32(&nchw_param, input, weight, bias, output);
        }

        ConvertImageLayout4D(nchw_output->data<float>(), DataLayout::NCHW, param->out->mutable_data<float>(),
                             param->out->layout(), output_shape);
        return 0;
    }

    if (CanUseFastConv2DPath(param)) {
        if (cached_weight_tensor_ != param->w.get()) {
            cached_direct_weight_oc8_buffer_.clear();
            cached_winograd_weight_oc8_buffer_.clear();
            cached_winograd_weight_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }
        const float* input = param->input->data<float>();
        const float* weight = param->w->data<float>();
        const float* bias =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
        float* output = param->out->mutable_data<float>();

        if (IsPointwiseConv2D(param)) {
            if (param->w->dims()[0] >= 8) {
                PackPointwiseInputFp32(param, input, &cached_packed_input_buffer_);
                if (cached_direct_weight_oc8_buffer_.empty()) {
                    x86::PackPointwiseWeightsOc8Fp32(
                        weight, bias, param->w->dims()[0], param->w->dims()[1], &cached_direct_weight_oc8_buffer_,
                        &cached_direct_bias_oc8_buffer_);
                }
                return x86::ComputePointwiseConvPackedOc8X86Fp32(
                    cached_packed_input_buffer_.data(), cached_direct_weight_oc8_buffer_.data(),
                    cached_direct_bias_oc8_buffer_.data(), weight, bias, param->input->dims()[0],
                    param->out->dims()[2] * param->out->dims()[3], param->input->dims()[1], param->w->dims()[0],
                    output);
            }
            ComputePointwiseConv2DKernelFastX86Fp32(param, input, weight, bias, output);
            return 0;
        }

        if (CanUseWinograd3x3Kernel(param)) {
            if (cached_winograd_weight_buffer_.empty()) {
                PackWinogradWeights3x3Fp32(weight, param->w->dims()[0], param->w->dims()[1],
                                           &cached_winograd_weight_oc8_buffer_, &cached_winograd_weight_buffer_);
            }
            ComputeWinograd3x3Fp32(param, input, cached_winograd_weight_oc8_buffer_.data(),
                                   cached_winograd_weight_buffer_.data(), bias, output);
            return 0;
        }

        if (CanUseDirectConvOc8Kernel(param)) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                x86::PackDirectConvWeightsOc8Fp32(
                    weight, bias, param->w->dims()[0], param->w->dims()[1] * param->w->dims()[2] * param->w->dims()[3],
                    &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
            }
            if (CanUseDirectConv3x3SpecializedKernel(param)) {
                ComputeDirectConv3x3Oc8SpecializedX86Fp32(
                    param, input, cached_direct_weight_oc8_buffer_.data(), cached_direct_bias_oc8_buffer_.data(),
                    weight, bias, output);
                return 0;
            }
            ComputeDirectConv2DOc8SpatialReuseX86Fp32(
                param, input, cached_direct_weight_oc8_buffer_.data(), cached_direct_bias_oc8_buffer_.data(), weight,
                bias, output);
            return 0;
        }

        ComputeDirectConv2DKernelFastX86Fp32(param, input, weight, bias, output);
        return 0;
    }

    return ComputeConv2DFallback<DataType::FP32>(param);
}

template <>
int32_t Conv2DKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Conv2D::FP16");
    auto* param = static_cast<feather::operators::Conv2dParam*>(param_);
    if (param == nullptr) {
        return -1;
    }

    if (CanUsePointwiseConv2DNhwcPath(param)) {
        if (cached_weight_tensor_ != param->w.get()) {
            cached_weight_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->w);
            cached_direct_weight_oc8_buffer_.clear();
            cached_direct_bias_oc8_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_bias_buffer_.clear();
            if (current_bias_tensor != nullptr) {
                cached_bias_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->bias);
            }
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }
        param->out->mutable_data<uint16_t>();
        PackPointwiseInputNhwcFp16ToFloat(param, param->input->data<uint16_t>(), &cached_packed_input_buffer_);
        if (param->w->dims()[0] >= 8) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                x86::PackPointwiseWeightsOc8Fp32(
                    cached_weight_buffer_.data(), cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                    param->w->dims()[0], param->w->dims()[1], &cached_direct_weight_oc8_buffer_,
                    &cached_direct_bias_oc8_buffer_);
            }
        }
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            return -1;
        }
        const int64_t batch = param->input->dims()[0];
        const int64_t output_spatial = output_shape.h * output_shape.w;
        return x86::ComputePointwiseConvPackedOc8NhwcX86Fp32(
            cached_packed_input_buffer_.data(),
            cached_direct_weight_oc8_buffer_.empty() ? nullptr : cached_direct_weight_oc8_buffer_.data(),
            cached_direct_bias_oc8_buffer_.empty() ? nullptr : cached_direct_bias_oc8_buffer_.data(),
            cached_weight_buffer_.data(),
            cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), batch, output_spatial,
            param->w->dims()[1], param->w->dims()[0], param->out->mutable_data<uint16_t>());
    }

    if (CanUseNhwcToNchwFastConv2DPath(param)) {
        ImageShape4D input_shape;
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
            !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            return -1;
        }

        auto nchw_input =
            std::make_shared<Tensor>(std::vector<int64_t>{input_shape.n, input_shape.c, input_shape.h, input_shape.w});
        auto nchw_output = std::make_shared<Tensor>(
            std::vector<int64_t>{output_shape.n, param->w->dims()[0], output_shape.h, output_shape.w});
        nchw_input->set_layout(DataLayout::NCHW);
        nchw_output->set_layout(DataLayout::NCHW);
        ConvertImageLayout4D(param->input->data<uint16_t>(), param->input->layout(), nchw_input->mutable_data<uint16_t>(),
                             DataLayout::NCHW, input_shape);

        feather::operators::Conv2dParam nchw_param = *param;
        nchw_param.input = nchw_input;
        nchw_param.out = nchw_output;

        if (cached_weight_tensor_ != param->w.get()) {
            cached_weight_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->w);
            cached_direct_weight_oc8_buffer_.clear();
            cached_direct_bias_oc8_buffer_.clear();
            cached_winograd_weight_oc8_buffer_.clear();
            cached_winograd_weight_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_bias_buffer_.clear();
            if (current_bias_tensor != nullptr) {
                cached_bias_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->bias);
            }
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }
        nchw_output->mutable_data<uint16_t>();

        if (IsPointwiseConv2D(&nchw_param)) {
            PackPointwiseInputFp16ToFloat(&nchw_param, nchw_input->data<uint16_t>(), &cached_packed_input_buffer_);
            if (nchw_param.w->dims()[0] >= 8) {
                if (cached_direct_weight_oc8_buffer_.empty()) {
                    x86::PackPointwiseWeightsOc8Fp32(
                        cached_weight_buffer_.data(),
                        cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), nchw_param.w->dims()[0],
                        nchw_param.w->dims()[1], &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
                }
                x86::ComputePointwiseConvPackedOc8X86Fp32(
                    cached_packed_input_buffer_.data(), cached_direct_weight_oc8_buffer_.data(),
                    cached_direct_bias_oc8_buffer_.data(), cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), nchw_param.input->dims()[0],
                    nchw_param.out->dims()[2] * nchw_param.out->dims()[3], nchw_param.input->dims()[1],
                    nchw_param.w->dims()[0], nchw_output->mutable_data<uint16_t>());
            } else {
                ComputePointwiseConv2DKernelPackedX86Fp32(
                    &nchw_param, cached_packed_input_buffer_.data(), cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                    nchw_output->mutable_data<uint16_t>());
            }
        } else if (CanUseWinograd3x3Kernel(&nchw_param)) {
            if (cached_winograd_weight_buffer_.empty()) {
                PackWinogradWeights3x3Fp32(cached_weight_buffer_.data(), nchw_param.w->dims()[0],
                                           nchw_param.w->dims()[1], &cached_winograd_weight_oc8_buffer_,
                                           &cached_winograd_weight_buffer_);
            }
            ComputeWinograd3x3Fp16(&nchw_param, nchw_input->data<uint16_t>(), cached_winograd_weight_oc8_buffer_.data(),
                                   cached_winograd_weight_buffer_.data(),
                                   cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                                   nchw_output->mutable_data<uint16_t>());
        } else if (CanUseDirectConvOc8Kernel(&nchw_param)) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                PackDirectConvWeightsOc8(
                    cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), nchw_param.w->dims()[0],
                    nchw_param.w->dims()[1] * nchw_param.w->dims()[2] * nchw_param.w->dims()[3],
                    &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
            }
            ComputeDirectConv2DSpatialReuseOc8X86Fp16(
                &nchw_param, nchw_input->data<uint16_t>(), cached_direct_weight_oc8_buffer_.data(),
                cached_direct_bias_oc8_buffer_.data(), cached_weight_buffer_.data(),
                cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), nchw_output->mutable_data<uint16_t>());
        } else {
            ComputeDirectConv2DSpatialReuseX86Fp16(
                &nchw_param, nchw_input->data<uint16_t>(), cached_weight_buffer_.data(),
                cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), nchw_output->mutable_data<uint16_t>());
        }

        ConvertImageLayout4D(nchw_output->data<uint16_t>(), DataLayout::NCHW, param->out->mutable_data<uint16_t>(),
                             param->out->layout(), output_shape);
        return 0;
    }

    if (CanUseFastConv2DPath(param)) {
        if (cached_weight_tensor_ != param->w.get()) {
            cached_weight_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->w);
            cached_direct_weight_oc8_buffer_.clear();
            cached_direct_bias_oc8_buffer_.clear();
            cached_winograd_weight_oc8_buffer_.clear();
            cached_winograd_weight_buffer_.clear();
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_bias_buffer_.clear();
            if (current_bias_tensor != nullptr) {
                cached_bias_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->bias);
            }
            cached_direct_bias_oc8_buffer_.clear();
            cached_bias_tensor_ = current_bias_tensor;
        }
        param->out->mutable_data<uint16_t>();

        if (IsPointwiseConv2D(param)) {
            PackPointwiseInputFp16ToFloat(param, param->input->data<uint16_t>(), &cached_packed_input_buffer_);
            if (param->w->dims()[0] >= 8) {
                if (cached_direct_weight_oc8_buffer_.empty()) {
                    x86::PackPointwiseWeightsOc8Fp32(
                        cached_weight_buffer_.data(),
                        cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), param->w->dims()[0],
                        param->w->dims()[1], &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
                }
                return x86::ComputePointwiseConvPackedOc8X86Fp32(
                    cached_packed_input_buffer_.data(), cached_direct_weight_oc8_buffer_.data(),
                    cached_direct_bias_oc8_buffer_.data(), cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(), param->input->dims()[0],
                    param->out->dims()[2] * param->out->dims()[3], param->input->dims()[1], param->w->dims()[0],
                    param->out->mutable_data<uint16_t>());
            }
            ComputePointwiseConv2DKernelPackedX86Fp32(
                param, cached_packed_input_buffer_.data(), cached_weight_buffer_.data(),
                cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                param->out->mutable_data<uint16_t>());
        } else {
            if (CanUseWinograd3x3Kernel(param)) {
                if (cached_winograd_weight_buffer_.empty()) {
                    PackWinogradWeights3x3Fp32(cached_weight_buffer_.data(), param->w->dims()[0], param->w->dims()[1],
                                               &cached_winograd_weight_oc8_buffer_, &cached_winograd_weight_buffer_);
                }
                ComputeWinograd3x3Fp16(param, param->input->data<uint16_t>(),
                                       cached_winograd_weight_oc8_buffer_.data(),
                                       cached_winograd_weight_buffer_.data(),
                                       cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                                       param->out->mutable_data<uint16_t>());
            } else if (CanUseDirectConvOc8Kernel(param)) {
                if (cached_direct_weight_oc8_buffer_.empty()) {
                    PackDirectConvWeightsOc8(
                        cached_weight_buffer_.data(),
                        cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                        param->w->dims()[0], param->w->dims()[1] * param->w->dims()[2] * param->w->dims()[3],
                        &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
                }
                ComputeDirectConv2DSpatialReuseOc8X86Fp16(
                    param, param->input->data<uint16_t>(), cached_direct_weight_oc8_buffer_.data(),
                    cached_direct_bias_oc8_buffer_.data(), cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                    param->out->mutable_data<uint16_t>());
            } else {
                ComputeDirectConv2DSpatialReuseX86Fp16(
                    param, param->input->data<uint16_t>(), cached_weight_buffer_.data(),
                    cached_bias_buffer_.empty() ? nullptr : cached_bias_buffer_.data(),
                    param->out->mutable_data<uint16_t>());
            }
        }
        return 0;
    }

    return ComputeConv2DFallback<DataType::FP16>(param);
}

void EnsureX86Conv2DKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Conv2D",
            []() { return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Conv2D",
            []() { return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
