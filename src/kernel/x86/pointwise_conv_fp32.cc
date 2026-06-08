#include "src/kernel/x86/pointwise_conv_fp32.h"

#include <immintrin.h>

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

#include "util/thread_pool_nv.h"
#include "util/fp16.h"

namespace feather {
namespace kernel {
namespace x86 {

namespace {

inline int64_t ComputePointwiseOutSize(int64_t input_size, int64_t stride) {
    return (input_size - 1) / stride + 1;
}

inline void FillOutputWithBias(float* output, int64_t count, float bias_value) {
    const __m256 bias_vec = _mm256_set1_ps(bias_value);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(output + i, bias_vec);
    }
    for (; i < count; ++i) {
        output[i] = bias_value;
    }
}

inline void Store8FloatsToHalfStrided(const __m256 value, uint16_t* dst, int64_t stride) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    for (int i = 0; i < 8; ++i) {
        dst[static_cast<int64_t>(i) * stride] = FloatToHalf(tmp[i]);
    }
}

size_t GetPointwiseThreadCount(int64_t total_work_items) {
    if (total_work_items <= 1) {
        return 1;
    }
    const unsigned int hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    return std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(total_work_items), hardware_threads));
}

ThreadPoolNv& GetPointwiseThreadPool() {
    static ThreadPoolNv pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

template <typename Fn>
void ParallelForPointwiseWorkItems(int64_t total_work_items, Fn&& fn) {
    const size_t thread_count = GetPointwiseThreadCount(total_work_items);
    if (thread_count <= 1) {
        fn(0, total_work_items);
        return;
    }

    ThreadPoolNv& pool = GetPointwiseThreadPool();
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

}  // namespace

void ComputePointwiseConv2DOutputChannelX86Fp32(const float* input, const float* weight, const float* bias,
                                                int64_t batch_index, int64_t out_channel, int64_t in_c, int64_t in_h,
                                                int64_t in_w, int64_t out_c, int64_t stride_h, int64_t stride_w,
                                                float* output) {
    const int64_t out_h = ComputePointwiseOutSize(in_h, stride_h);
    const int64_t out_w = ComputePointwiseOutSize(in_w, stride_w);
    const int64_t input_spatial = in_h * in_w;
    const int64_t output_spatial = out_h * out_w;
    float* out_base = output + ((batch_index * out_c + out_channel) * output_spatial);
    const float bias_value = bias != nullptr ? bias[out_channel] : 0.0f;
    FillOutputWithBias(out_base, output_spatial, bias_value);

    const float* weight_base = weight + out_channel * in_c;
    for (int64_t ic = 0; ic < in_c; ++ic) {
        const float kernel_value = weight_base[ic];
        if (kernel_value == 0.0f) {
            continue;
        }

        const float* input_base = input + ((batch_index * in_c + ic) * input_spatial);
        const __m256 kernel_vec = _mm256_set1_ps(kernel_value);
        for (int64_t oh = 0; oh < out_h; ++oh) {
            const int64_t ih = oh * stride_h;
            const float* input_row = input_base + ih * in_w;
            float* output_row = out_base + oh * out_w;
            if (stride_w == 1) {
                int64_t ow = 0;
                for (; ow + 8 <= out_w; ow += 8) {
                    const __m256 input_vec = _mm256_loadu_ps(input_row + ow);
                    const __m256 output_vec = _mm256_loadu_ps(output_row + ow);
                    _mm256_storeu_ps(output_row + ow, _mm256_fmadd_ps(input_vec, kernel_vec, output_vec));
                }
                for (; ow < out_w; ++ow) {
                    output_row[ow] += input_row[ow] * kernel_value;
                }
                continue;
            }

            for (int64_t ow = 0; ow < out_w; ++ow) {
                output_row[ow] += input_row[ow * stride_w] * kernel_value;
            }
        }
    }
}

int32_t ComputePointwiseConv2DX86Fp32(const float* input, const float* weight, const float* bias, int64_t batch,
                                      int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t stride_h,
                                      int64_t stride_w, float* output) {
    if (input == nullptr || weight == nullptr || output == nullptr || batch <= 0 || in_c <= 0 || in_h <= 0 ||
        in_w <= 0 || out_c <= 0 || stride_h <= 0 || stride_w <= 0) {
        return -1;
    }

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            ComputePointwiseConv2DOutputChannelX86Fp32(input, weight, bias, n, oc, in_c, in_h, in_w, out_c, stride_h,
                                                       stride_w, output);
        }
    }
    return 0;
}

void PackPointwiseWeightsOc8Fp32(const float* weight, const float* bias, int64_t out_c, int64_t in_c,
                                 std::vector<float>* packed_weight, std::vector<float>* packed_bias) {
    const int64_t oc8_blocks = out_c / 8;
    packed_weight->assign(static_cast<size_t>(oc8_blocks * in_c * 8), 0.0f);
    packed_bias->assign(static_cast<size_t>(oc8_blocks * 8), 0.0f);

    float* packed_weight_ptr = packed_weight->data();
    float* packed_bias_ptr = packed_bias->data();
    for (int64_t block = 0; block < oc8_blocks; ++block) {
        for (int lane = 0; lane < 8; ++lane) {
            const int64_t oc = block * 8 + lane;
            packed_bias_ptr[block * 8 + lane] = bias != nullptr ? bias[oc] : 0.0f;
        }
        for (int64_t ic = 0; ic < in_c; ++ic) {
            for (int lane = 0; lane < 8; ++lane) {
                const int64_t oc = block * 8 + lane;
                packed_weight_ptr[(block * in_c + ic) * 8 + lane] = weight[oc * in_c + ic];
            }
        }
    }
}

int32_t ComputePointwiseConvPackedOc8X86Fp32(const float* packed_input, const float* packed_weight_oc8,
                                             const float* packed_bias_oc8, const float* weight, const float* bias,
                                             int64_t batch, int64_t output_spatial, int64_t in_c, int64_t out_c,
                                             uint16_t* output) {
    if (packed_input == nullptr || packed_weight_oc8 == nullptr || packed_bias_oc8 == nullptr || weight == nullptr ||
        output == nullptr || batch <= 0 || output_spatial <= 0 || in_c <= 0 || out_c <= 0) {
        return -1;
    }

    const int64_t oc8_blocks = out_c / 8;
    const int64_t oc_tail_begin = oc8_blocks * 8;
    const int64_t total_work_items = batch * output_spatial;

    ParallelForPointwiseWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / output_spatial;
            const int64_t spatial_idx = work_index % output_spatial;
            const float* input_row = packed_input + ((n * output_spatial + spatial_idx) * in_c);

            for (int64_t block = 0; block < oc8_blocks; ++block) {
                __m256 acc = _mm256_loadu_ps(packed_bias_oc8 + block * 8);
                const float* block_weight = packed_weight_oc8 + block * in_c * 8;
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    const __m256 weight_vec = _mm256_loadu_ps(block_weight + ic * 8);
                    const __m256 input_vec = _mm256_set1_ps(input_row[ic]);
                    acc = _mm256_fmadd_ps(input_vec, weight_vec, acc);
                }
                Store8FloatsToHalfStrided(
                    acc, output + ((n * out_c + block * 8) * output_spatial) + spatial_idx, output_spatial);
            }

            for (int64_t oc = oc_tail_begin; oc < out_c; ++oc) {
                const float* weight_row = weight + oc * in_c;
                float sum = bias != nullptr ? bias[oc] : 0.0f;
                for (int64_t ic = 0; ic < in_c; ++ic) {
                    sum += input_row[ic] * weight_row[ic];
                }
                output[((n * out_c + oc) * output_spatial) + spatial_idx] = FloatToHalf(sum);
            }
        }
    });
    return 0;
}

}  // namespace x86
}  // namespace kernel
}  // namespace feather
