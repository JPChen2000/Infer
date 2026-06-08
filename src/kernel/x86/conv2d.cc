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
           param->dilation_h == 1 && param->dilation_w == 1;
}

bool IsPointwiseConv2D(const feather::operators::Conv2dParam* param) {
    if (!CanUseFastConv2DPath(param)) {
        return false;
    }
    return param->w->dims()[2] == 1 && param->w->dims()[3] == 1 && param->pad_h == 0 && param->pad_w == 0;
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
            x86::ComputeDirectConv2DSpatialOutputX86Fp32(
                input, weight, bias, n, spatial_idx, in_c, in_h, in_w, out_c, kernel_h, kernel_w, out_h, out_w,
                param->stride_h, param->stride_w, param->pad_h, param->pad_w, input_patch.data(), output);
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

    const int batch = static_cast<int>(param->input->dims()[0]);
    const int in_c = static_cast<int>(param->input->dims()[1]);
    const int in_h = static_cast<int>(param->input->dims()[2]);
    const int in_w = static_cast<int>(param->input->dims()[3]);
    const int out_c = static_cast<int>(param->w->dims()[0]);
    const int kernel_c = static_cast<int>(param->w->dims()[1]);
    const int kernel_h = static_cast<int>(param->w->dims()[2]);
    const int kernel_w = static_cast<int>(param->w->dims()[3]);
    const int out_h = static_cast<int>(param->out->dims()[2]);
    const int out_w = static_cast<int>(param->out->dims()[3]);
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
                                        ((static_cast<int64_t>(n) * in_c + global_ic) * in_h + ih) * in_w + iw;
                                    const int64_t kernel_offset =
                                        ((static_cast<int64_t>(global_oc) * kernel_c + ic) * kernel_h + kh) * kernel_w + kw;
                                    value += TensorIO<dtype>::Read(param->input.get(), input_offset) *
                                             TensorIO<dtype>::Read(param->w.get(), kernel_offset);
                                }
                            }
                        }
                        const int64_t out_offset =
                            ((static_cast<int64_t>(n) * out_c + global_oc) * out_h + oh) * out_w + ow;
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

    if (CanUseFastConv2DPath(param)) {
        if (cached_weight_tensor_ != param->w.get()) {
            cached_direct_weight_oc8_buffer_.clear();
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
            ComputePointwiseConv2DKernelFastX86Fp32(param, input, weight, bias, output);
            return 0;
        }

        if (CanUseDirectConvOc8Kernel(param)) {
            if (cached_direct_weight_oc8_buffer_.empty()) {
                x86::PackDirectConvWeightsOc8Fp32(
                    weight, bias, param->w->dims()[0], param->w->dims()[1] * param->w->dims()[2] * param->w->dims()[3],
                    &cached_direct_weight_oc8_buffer_, &cached_direct_bias_oc8_buffer_);
            }
            return x86::ComputeDirectConv2DOc8X86Fp32(
                input, cached_direct_weight_oc8_buffer_.data(), cached_direct_bias_oc8_buffer_.data(), weight, bias,
                param->input->dims()[0], param->input->dims()[1], param->input->dims()[2], param->input->dims()[3],
                param->w->dims()[0], param->w->dims()[2], param->w->dims()[3], param->stride_h, param->stride_w,
                param->pad_h, param->pad_w, output);
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

    if (CanUseFastConv2DPath(param)) {
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
            if (CanUseDirectConvOc8Kernel(param)) {
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
