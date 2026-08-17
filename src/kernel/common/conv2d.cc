#include "core/tensor.h"
#include "src/kernel/conv2d.h"
#include "src/operator/params.h"
#include "util/logger.h"
#include "util/thread_pool_nv.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/types.h"

#include <algorithm>
#include <future>
#include <vector>

#include "src/kernel/common/kernel_io.h"

using feather::DataType;
using feather::Tensor;

namespace feather {
namespace kernel {

namespace {
bool g_conv2d_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Conv2D",
                                               []() { return std::make_unique<Conv2DKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Conv2D",
                                               []() { return std::make_unique<Conv2DKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Conv2D",
                                               []() { return std::make_unique<Conv2DKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

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

void FillOutputWithBias(float* output_base, int64_t spatial_size, const float* bias, int64_t oc) {
    const float bias_value = bias != nullptr ? bias[oc] : 0.0f;
    std::fill(output_base, output_base + spatial_size, bias_value);
}

void ComputePointwiseConv2DKernelFastImpl(const feather::operators::Conv2dParam* param, const float* input,
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
    const int64_t total_work_items = batch * out_c;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / out_c;
            const int64_t oc = work_index % out_c;
            float* out_base = output + work_index * output_spatial;
            FillOutputWithBias(out_base, output_spatial, bias, oc);

            const float* weight_base = weight + oc * in_c;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const float kernel_value = weight_base[ic];
                const float* input_base = input + (n * in_c + ic) * input_spatial;
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    const int64_t ih = oh * param->stride_h;
                    const float* input_row = input_base + ih * in_w;
                    float* output_row = out_base + oh * out_w;
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        output_row[ow] += input_row[ow * param->stride_w] * kernel_value;
                    }
                }
            }
        }
    });
}

void ComputeDirectConv2DKernelFastImpl(const feather::operators::Conv2dParam* param, const float* input,
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
    const int64_t kernel_spatial = kernel_h * kernel_w;
    const int64_t total_work_items = batch * out_c;

    ParallelForWorkItems(total_work_items, [&](int64_t begin, int64_t end) {
        for (int64_t work_index = begin; work_index < end; ++work_index) {
            const int64_t n = work_index / out_c;
            const int64_t oc = work_index % out_c;
            float* out_base = output + work_index * output_spatial;
            FillOutputWithBias(out_base, output_spatial, bias, oc);

            const float* weight_oc = weight + oc * in_c * kernel_spatial;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                const float* input_plane = input + (n * in_c + ic) * input_spatial;
                const float* weight_ic = weight_oc + ic * kernel_spatial;
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    float* output_row = out_base + oh * out_w;
                    const int64_t ih_base = oh * param->stride_h - param->pad_h;
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = output_row[ow];
                        const int64_t iw_base = ow * param->stride_w - param->pad_w;
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            const int64_t ih = ih_base + kh;
                            if (ih < 0 || ih >= in_h) {
                                continue;
                            }
                            const float* input_row = input_plane + ih * in_w;
                            const float* weight_row = weight_ic + kh * kernel_w;
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const int64_t iw = iw_base + kw;
                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                sum += input_row[iw] * weight_row[kw];
                            }
                        }
                        output_row[ow] = sum;
                    }
                }
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

int32_t ComputePointwiseConv2DKernelFastFp32(const feather::operators::Conv2dParam* param) {
    const float* input = param->input->data<float>();
    const float* weight = param->w->data<float>();
    const float* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    float* output = param->out->mutable_data<float>();
    ComputePointwiseConv2DKernelFastImpl(param, input, weight, bias, output);
    return 0;
}

int32_t ComputeDirectConv2DKernelFastFp32(const feather::operators::Conv2dParam* param) {
    const float* input = param->input->data<float>();
    const float* weight = param->w->data<float>();
    const float* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    float* output = param->out->mutable_data<float>();
    ComputeDirectConv2DKernelFastImpl(param, input, weight, bias, output);
    return 0;
}

int32_t ComputePointwiseConv2DKernelFastFp16(const feather::operators::Conv2dParam* param) {
    return -1;
}

int32_t ComputePointwiseConv2DKernelFastFp16(const feather::operators::Conv2dParam* param,
                                             const std::vector<float>& weight_buffer,
                                             const std::vector<float>& bias_buffer) {
    const std::vector<float> input_buffer = ConvertTensorToFloatBuffer<DataType::FP16>(param->input);
    std::vector<float> output_buffer(static_cast<size_t>(param->out->numel()), 0.0f);
    param->out->mutable_data<uint16_t>();

    ComputePointwiseConv2DKernelFastImpl(param, input_buffer.data(), weight_buffer.data(),
                                         bias_buffer.empty() ? nullptr : bias_buffer.data(), output_buffer.data());
    WriteFloatBufferToTensor<DataType::FP16>(output_buffer, param->out.get());
    return 0;
}

int32_t ComputeDirectConv2DKernelFastFp16(const feather::operators::Conv2dParam* param) {
    return -1;
}

int32_t ComputeDirectConv2DKernelFastFp16(const feather::operators::Conv2dParam* param,
                                          const std::vector<float>& weight_buffer,
                                          const std::vector<float>& bias_buffer) {
    const std::vector<float> input_buffer = ConvertTensorToFloatBuffer<DataType::FP16>(param->input);
    std::vector<float> output_buffer(static_cast<size_t>(param->out->numel()), 0.0f);
    param->out->mutable_data<uint16_t>();

    ComputeDirectConv2DKernelFastImpl(param, input_buffer.data(), weight_buffer.data(),
                                      bias_buffer.empty() ? nullptr : bias_buffer.data(), output_buffer.data());
    WriteFloatBufferToTensor<DataType::FP16>(output_buffer, param->out.get());
    return 0;
}
}  // namespace

std::shared_ptr<Tensor> im2col(const std::shared_ptr<Tensor>& input,
        const int32_t kernel_h, const int32_t kernel_w,
        const int32_t stride_h, const int32_t stride_w,
        const int32_t pad_h, const int32_t pad_w) {
    int in_h = input->dims()[0];
    int in_w = input->dims()[1];
    int out_h = (in_h - kernel_h + 2 * pad_h) / stride_h + 1;
    int out_w = (in_w - kernel_w + 2 * pad_w) / stride_w + 1;
    
    auto col_tensor = std::make_shared<Tensor>(std::vector<int64_t>{out_h * out_w, kernel_h * kernel_w});

    const float* in = input->data<float>();
    float* col_out = col_tensor->mutable_data<float>();
    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            for (int ki = 0; ki < kernel_h; ++ki) {
                for (int kj = 0; kj < kernel_w; ++kj) {
                    int in_i = i * stride_h + ki - pad_h;
                    int in_j = j * stride_w + kj - pad_w;
                    if (in_i >= 0 && in_i < in_h && in_j >= 0 && in_j < in_w) {
                        *(col_out + (i * out_w + j) * kernel_h * kernel_w + ki * kernel_w + kj) =
                            *(in + in_i * in_w + in_j);
                    }
                }
            }
        }
    }
    return col_tensor;
}

template <DataType dtype>
int32_t ComputeConv2DKernel(feather::operators::Conv2dParam* param) {
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);

    if (CanUseFastConv2DPath(param)) {
        if constexpr (dtype == DataType::FP32) {
            if (IsPointwiseConv2D(param)) {
                LOG_INFO("[Conv2D] use fast pointwise kernel fp32");
                return ComputePointwiseConv2DKernelFastFp32(param);
            }
            LOG_INFO("[Conv2D] use fast direct kernel fp32");
            return ComputeDirectConv2DKernelFastFp32(param);
        }
        if constexpr (dtype == DataType::FP16) {
            if (IsPointwiseConv2D(param)) {
                LOG_INFO("[Conv2D] use fast pointwise kernel fp16");
                return ComputePointwiseConv2DKernelFastFp16(param);
            }
            LOG_INFO("[Conv2D] use fast direct kernel fp16");
            return ComputeDirectConv2DKernelFastFp16(param);
        }
    }

    if (param->input->dims().size() == 2 && param->w->dims().size() == 2) {
        const int in_h = param->input->dims()[0];
        const int in_w = param->input->dims()[1];
        const int kernel_h = param->w->dims()[0];
        const int kernel_w = param->w->dims()[1];
        const int out_h = (in_h + 2 * param->pad_h - param->dilation_h * (kernel_h - 1) - 1) / param->stride_h + 1;
        const int out_w = (in_w + 2 * param->pad_w - param->dilation_w * (kernel_w - 1) - 1) / param->stride_w + 1;

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

template <>
int32_t Conv2DKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Conv2D::FP32");
    return ComputeConv2DKernel<DataType::FP32>(static_cast<feather::operators::Conv2dParam*>(param_));
}

template <>
int32_t Conv2DKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Conv2D::FP16");
    auto* param = static_cast<feather::operators::Conv2dParam*>(param_);
    if (param == nullptr) {
        return -1;
    }
    if (CanUseFastConv2DPath(param)) {
        if (cached_weight_tensor_ != param->w.get()) {
            cached_weight_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->w);
            cached_weight_tensor_ = param->w.get();
        }
        const Tensor* current_bias_tensor =
            param->bias != nullptr && param->bias->IsInitialized() ? param->bias.get() : nullptr;
        if (cached_bias_tensor_ != current_bias_tensor) {
            cached_bias_buffer_.clear();
            if (current_bias_tensor != nullptr) {
                cached_bias_buffer_ = ConvertTensorToFloatBuffer<DataType::FP16>(param->bias);
            }
            cached_bias_tensor_ = current_bias_tensor;
        }
        if (IsPointwiseConv2D(param)) {
            LOG_INFO("[Conv2D] use fast pointwise kernel fp16 cached");
            return ComputePointwiseConv2DKernelFastFp16(param, cached_weight_buffer_, cached_bias_buffer_);
        }
        LOG_INFO("[Conv2D] use fast direct kernel fp16 cached");
        return ComputeDirectConv2DKernelFastFp16(param, cached_weight_buffer_, cached_bias_buffer_);
    }
    return ComputeConv2DKernel<DataType::FP16>(param);
}

template <>
int32_t Conv2DKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Conv2D::BF16");
    return ComputeConv2DKernel<DataType::BF16>(static_cast<feather::operators::Conv2dParam*>(param_));
}

typedef feather::kernel::Conv2DKernel<DeviceType::COMMON, DataType::FP32> Conv2DCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Conv2D, Conv2DCommonFP32Kernel);

typedef feather::kernel::Conv2DKernel<DeviceType::COMMON, DataType::FP16> Conv2DCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Conv2D, Conv2DCommonFP16Kernel);

typedef feather::kernel::Conv2DKernel<DeviceType::COMMON, DataType::BF16> Conv2DCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Conv2D, Conv2DCommonBF16Kernel);

void EnsureCommonConv2DKernelsRegistered() { (void)g_conv2d_kernels_registered; }

void EnsureConv2DKernelsRegistered() {
    EnsureCommonConv2DKernelsRegistered();
    EnsureX86Conv2DKernelsRegistered();
}
}  // namespace kernel
}  // namespace feather
