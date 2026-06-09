#include "src/kernel/conv2d.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_conv2d_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Conv2D",
                              []() { return std::make_unique<Conv2DKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Conv2D",
                              []() { return std::make_unique<Conv2DKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

#ifdef FEATHER_WITH_CUDNN
struct CudnnConvSignature {
    DataType dtype{DataType::UNKNOWN};
    DataLayout layout{DataLayout::ND};
    std::array<int64_t, 4> input_dims{};
    std::array<int64_t, 4> weight_dims{};
    std::array<int64_t, 4> output_dims{};
    int stride_h{};
    int stride_w{};
    int pad_h{};
    int pad_w{};
    int dilation_h{};
    int dilation_w{};
    int group{};

    bool operator==(const CudnnConvSignature& other) const {
        return dtype == other.dtype && layout == other.layout && input_dims == other.input_dims &&
               weight_dims == other.weight_dims && output_dims == other.output_dims &&
               stride_h == other.stride_h && stride_w == other.stride_w && pad_h == other.pad_h &&
               pad_w == other.pad_w && dilation_h == other.dilation_h && dilation_w == other.dilation_w &&
               group == other.group;
    }
};

struct CudnnConvPlan {
    CudnnConvSignature signature;
    cudnnConvolutionFwdAlgo_t algo{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM};
    size_t workspace_bytes{0};
    bool initialized{false};
};

std::mutex& CudnnPlanMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<const void*, CudnnConvPlan>& CudnnPlanCache() {
    static std::unordered_map<const void*, CudnnConvPlan> cache;
    return cache;
}

bool CudnnCheck(cudnnStatus_t status) {
    return status == CUDNN_STATUS_SUCCESS;
}

cudnnDataType_t CudnnTensorDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return CUDNN_DATA_HALF;
        case DataType::FP32:
        default:
            return CUDNN_DATA_FLOAT;
    }
}

cudnnDataType_t CudnnComputeDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return CUDNN_DATA_FLOAT;
        case DataType::FP32:
        default:
            return CUDNN_DATA_FLOAT;
    }
}

CudnnConvSignature MakeCudnnConvSignature(const feather::operators::Conv2dParam* param, DataType dtype) {
    CudnnConvSignature signature;
    signature.dtype = dtype;
    signature.layout = NormalizeDataLayout(param->input->layout());
    for (size_t i = 0; i < 4; ++i) {
        signature.input_dims[i] = param->input->dims()[i];
        signature.weight_dims[i] = param->w->dims()[i];
        signature.output_dims[i] = param->out->dims()[i];
    }
    signature.stride_h = param->stride_h;
    signature.stride_w = param->stride_w;
    signature.pad_h = param->pad_h;
    signature.pad_w = param->pad_w;
    signature.dilation_h = param->dilation_h;
    signature.dilation_w = param->dilation_w;
    signature.group = param->group;
    return signature;
}

bool SetCudnnTensor4dDescriptor(cudnnTensorDescriptor_t desc, DataType dtype, DataLayout layout,
                                const ImageShape4D& shape) {
    const auto tensor_format =
        NormalizeDataLayout(layout) == DataLayout::NHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    return CudnnCheck(
        cudnnSetTensor4dDescriptor(desc, tensor_format, CudnnTensorDataType(dtype), static_cast<int>(shape.n),
                                   static_cast<int>(shape.c), static_cast<int>(shape.h), static_cast<int>(shape.w)));
}

struct ScopedCudnnTensorDesc {
    cudnnTensorDescriptor_t desc{nullptr};
    ScopedCudnnTensorDesc() { (void)cudnnCreateTensorDescriptor(&desc); }
    ~ScopedCudnnTensorDesc() {
        if (desc != nullptr) {
            cudnnDestroyTensorDescriptor(desc);
        }
    }
};

struct ScopedCudnnFilterDesc {
    cudnnFilterDescriptor_t desc{nullptr};
    ScopedCudnnFilterDesc() { (void)cudnnCreateFilterDescriptor(&desc); }
    ~ScopedCudnnFilterDesc() {
        if (desc != nullptr) {
            cudnnDestroyFilterDescriptor(desc);
        }
    }
};

struct ScopedCudnnConvDesc {
    cudnnConvolutionDescriptor_t desc{nullptr};
    ScopedCudnnConvDesc() { (void)cudnnCreateConvolutionDescriptor(&desc); }
    ~ScopedCudnnConvDesc() {
        if (desc != nullptr) {
            cudnnDestroyConvolutionDescriptor(desc);
        }
    }
};

template <DataType dtype>
bool SelectCudnnForwardPlan(const void* kernel_identity, cudnnHandle_t handle, cudnnTensorDescriptor_t input_desc,
                            cudnnFilterDescriptor_t filter_desc, cudnnConvolutionDescriptor_t conv_desc,
                            cudnnTensorDescriptor_t output_desc, const feather::operators::Conv2dParam* param,
                            size_t* workspace_bytes, cudnnConvolutionFwdAlgo_t* algo) {
    if (workspace_bytes == nullptr || algo == nullptr) {
        return false;
    }
    const auto signature = MakeCudnnConvSignature(param, dtype);
    {
        std::lock_guard<std::mutex> lock(CudnnPlanMutex());
        auto it = CudnnPlanCache().find(kernel_identity);
        if (it != CudnnPlanCache().end() && it->second.initialized && it->second.signature == signature) {
            *workspace_bytes = it->second.workspace_bytes;
            *algo = it->second.algo;
            return true;
        }
    }

    constexpr int kAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returned_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[kAlgoCount];
    if (!CudnnCheck(cudnnFindConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
                                                         kAlgoCount, &returned_count, perf)) ||
        returned_count <= 0) {
        return false;
    }

    const auto selection_shape = CudnnConvSelectionShape{signature.dtype,
                                                         signature.layout,
                                                         signature.input_dims,
                                                         signature.weight_dims,
                                                         signature.output_dims,
                                                         signature.dilation_h,
                                                         signature.dilation_w,
                                                         signature.group};
    const int selected_index = SelectPreferredCudaConv2DPerfIndex(selection_shape, perf, returned_count);
    if (selected_index < 0 || selected_index >= returned_count) {
        return false;
    }
    cudnnConvolutionFwdAlgo_t selected_algo = perf[selected_index].algo;

    size_t selected_workspace_bytes = 0;
    if (!CudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc,
                                                            selected_algo, &selected_workspace_bytes))) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(CudnnPlanMutex());
        CudnnPlanCache()[kernel_identity] = CudnnConvPlan{signature, selected_algo, selected_workspace_bytes, true};
    }
    *workspace_bytes = selected_workspace_bytes;
    *algo = selected_algo;
    return true;
}

template <DataType dtype>
bool RunConv2DWithCudnn(const void* kernel_identity, feather::operators::Conv2dParam* param,
                        cuda_detail::DeviceBuffer<cuda_detail::StorageT<dtype>>* input,
                        cuda_detail::DeviceBuffer<cuda_detail::StorageT<dtype>>* weight,
                        cuda_detail::DeviceBuffer<cuda_detail::StorageT<dtype>>* bias,
                        cuda_detail::DeviceBuffer<cuda_detail::StorageT<dtype>>* out, bool has_bias) {
    if (param == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        return false;
    }
    if (param->input->dims().size() != 4 || param->w->dims().size() != 4 || param->out->dims().size() != 4) {
        return false;
    }

    auto handle = cuda_detail::CudnnHandle();
    if (handle == nullptr) {
        return false;
    }

    ImageShape4D input_shape;
    ImageShape4D output_shape;
    if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
        !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
        return false;
    }

    ScopedCudnnTensorDesc input_desc;
    ScopedCudnnTensorDesc output_desc;
    ScopedCudnnTensorDesc bias_desc;
    ScopedCudnnFilterDesc filter_desc;
    ScopedCudnnConvDesc conv_desc;
    if (input_desc.desc == nullptr || output_desc.desc == nullptr || filter_desc.desc == nullptr || conv_desc.desc == nullptr) {
        return false;
    }
    if (!SetCudnnTensor4dDescriptor(input_desc.desc, dtype, param->input->layout(), input_shape) ||
        !SetCudnnTensor4dDescriptor(output_desc.desc, dtype, param->out->layout(), output_shape)) {
        return false;
    }

    const int filter_dims[4] = {static_cast<int>(param->w->dims()[0]), static_cast<int>(param->w->dims()[1]),
                                static_cast<int>(param->w->dims()[2]), static_cast<int>(param->w->dims()[3])};
    if (!CudnnCheck(cudnnSetFilterNdDescriptor(filter_desc.desc, CudnnTensorDataType(dtype), CUDNN_TENSOR_NCHW, 4,
                                               filter_dims))) {
        return false;
    }

    const int pads[2] = {param->pad_h, param->pad_w};
    const int strides[2] = {param->stride_h, param->stride_w};
    const int dilations[2] = {param->dilation_h, param->dilation_w};
    if (!CudnnCheck(cudnnSetConvolutionNdDescriptor(conv_desc.desc, 2, pads, strides, dilations,
                                                    CUDNN_CROSS_CORRELATION, CudnnComputeDataType(dtype))) ||
        !CudnnCheck(cudnnSetConvolutionGroupCount(conv_desc.desc, std::max(1, param->group))) ||
        !CudnnCheck(cudnnSetConvolutionMathType(conv_desc.desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION))) {
        return false;
    }

    size_t workspace_bytes = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (!SelectCudnnForwardPlan<dtype>(kernel_identity, handle, input_desc.desc, filter_desc.desc, conv_desc.desc,
                                       output_desc.desc, param, &workspace_bytes, &algo)) {
        return false;
    }

    void* workspace_ptr = nullptr;
    if (workspace_bytes != 0 && cuda_detail::AcquireTemporaryDeviceBuffer(workspace_bytes, &workspace_ptr) != 0) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool ok = CudnnCheck(cudnnConvolutionForward(handle, &alpha, input_desc.desc, input->get(), filter_desc.desc,
                                                       weight->get(), conv_desc.desc, algo, workspace_ptr,
                                                       workspace_bytes, &beta, output_desc.desc, out->get()));
    if (workspace_ptr != nullptr) {
        cuda_detail::ReleaseTemporaryDeviceBuffer(workspace_ptr, workspace_bytes);
    }
    if (!ok) {
        return false;
    }

    if (has_bias) {
        if (bias_desc.desc == nullptr) {
            return false;
        }
        const auto bias_shape = ImageShape4D{1, param->w->dims()[0], 1, 1};
        if (!SetCudnnTensor4dDescriptor(bias_desc.desc, dtype, param->out->layout(), bias_shape) ||
            !CudnnCheck(cudnnAddTensor(handle, &alpha, bias_desc.desc, bias->get(), &alpha, output_desc.desc,
                                       out->get()))) {
            return false;
        }
    }

    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return false;
    }
    SetLastCudaConv2DBackend(CudaConv2DBackend::kCudnn);
    return true;
}
#endif

template <typename T>
__global__ void Conv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                   int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c, int64_t kernel_c,
                                   int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w, int stride_h,
                                   int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group,
                                   bool channel_last) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t out_c_per_group = out_c / group;
    const int64_t in_c_per_group = in_c / group;
    const int64_t g = oc / out_c_per_group;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
        const int64_t global_ic = g * in_c_per_group + ic;
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
            if (ih < 0 || ih >= in_h) {
                continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
                if (iw < 0 || iw >= in_w) {
                    continue;
                }
                const int64_t input_offset = channel_last ? ((n * in_h + ih) * in_w + iw) * in_c + global_ic
                                                          : ((n * in_c + global_ic) * in_h + ih) * in_w + iw;
                const int64_t weight_offset = ((oc * kernel_c + ic) * kernel_h + kh) * kernel_w + kw;
                sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
            }
        }
    }
    const int64_t out_offset = channel_last ? ((n * out_h + oh) * out_w + ow) * out_c + oc : idx;
    cuda_detail::WriteDevice(out, out_offset, sum);
}

template <typename T>
__global__ void PointwiseConv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                            int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c,
                                            int64_t kernel_c, int64_t out_h, int64_t out_w, int stride_h,
                                            int stride_w, int pad_h, int pad_w, int group, bool channel_last) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t ih = oh * stride_h - pad_h;
    const int64_t iw = ow * stride_w - pad_w;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        const int64_t out_c_per_group = out_c / group;
        const int64_t in_c_per_group = in_c / group;
        const int64_t g = oc / out_c_per_group;
        for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
            const int64_t global_ic = g * in_c_per_group + ic;
            const int64_t input_offset = channel_last ? ((n * in_h + ih) * in_w + iw) * in_c + global_ic
                                                      : ((n * in_c + global_ic) * in_h + ih) * in_w + iw;
            const int64_t weight_offset = oc * kernel_c + ic;
            sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
        }
    }
    const int64_t out_offset = channel_last ? ((n * out_h + oh) * out_w + ow) * out_c + oc : idx;
    cuda_detail::WriteDevice(out, out_offset, sum);
}

template <typename T>
__global__ void DepthwiseConv2D4DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t batch,
                                            int64_t in_c, int64_t in_h, int64_t in_w, int64_t out_c,
                                            int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w,
                                            int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
                                            int dilation_w, bool channel_last) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_c * out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t ow = idx % out_w;
    const int64_t oh = (idx / out_w) % out_h;
    const int64_t oc = (idx / (out_w * out_h)) % out_c;
    const int64_t n = idx / (out_c * out_h * out_w);
    const int64_t depth_multiplier = out_c / in_c;
    const int64_t ic = oc / depth_multiplier;
    float sum = bias != nullptr ? cuda_detail::ReadDevice(bias, oc) : 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
        if (ih < 0 || ih >= in_h) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
            if (iw < 0 || iw >= in_w) {
                continue;
            }
            const int64_t input_offset = channel_last ? ((n * in_h + ih) * in_w + iw) * in_c + ic
                                                      : ((n * in_c + ic) * in_h + ih) * in_w + iw;
            const int64_t weight_offset = ((oc * 1) * kernel_h + kh) * kernel_w + kw;
            sum += cuda_detail::ReadDevice(input, input_offset) * cuda_detail::ReadDevice(weight, weight_offset);
        }
    }
    const int64_t out_offset = channel_last ? ((n * out_h + oh) * out_w + ow) * out_c + oc : idx;
    cuda_detail::WriteDevice(out, out_offset, sum);
}

template <typename T>
__global__ void Conv2D2DKernelCuda(const T* input, const T* weight, const T* bias, T* out, int64_t in_h,
                                   int64_t in_w, int64_t kernel_h, int64_t kernel_w, int64_t out_h, int64_t out_w,
                                   int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
                                   int dilation_w) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int64_t oh = idx / out_w;
    const int64_t ow = idx % out_w;
    float sum = 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
        if (ih < 0 || ih >= in_h) {
            continue;
        }
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
            if (iw < 0 || iw >= in_w) {
                continue;
            }
            sum += cuda_detail::ReadDevice(input, ih * in_w + iw) * cuda_detail::ReadDevice(weight, kh * kernel_w + kw);
        }
    }
    if (bias != nullptr) {
        sum += cuda_detail::ReadDevice(bias, idx);
    }
    cuda_detail::WriteDevice(out, idx, sum);
}

template <DataType dtype>
int RunConv2DFallback(feather::operators::Conv2dParam* param) {
    using T = cuda_detail::StorageT<dtype>;
    SetLastCudaConv2DBackend(CudaConv2DBackend::kFallback);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->input->dims().size() == 2 && param->w->dims().size() == 2) {
        cuda_detail::DeviceBuffer<T> input;
        cuda_detail::DeviceBuffer<T> weight;
        cuda_detail::DeviceBuffer<T> bias;
        cuda_detail::DeviceBuffer<T> out;
        T* bias_ptr = nullptr;
        if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
            cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
            cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
            return -1;
        }
        if (param->bias != nullptr && param->bias->IsInitialized()) {
            if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
                return -1;
            }
            bias_ptr = bias.get();
        }
        const int64_t out_numel = param->out->numel();
        Conv2D2DKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                               cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
            input.get(), weight.get(), bias_ptr, out.get(), param->input->dims()[0], param->input->dims()[1],
            param->w->dims()[0], param->w->dims()[1], param->out->dims()[0], param->out->dims()[1],
            param->stride_h, param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w);
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
            return -1;
        }
        return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
    } else if (param->input->dims().size() == 4 && param->w->dims().size() == 4) {
        cuda_detail::DeviceBuffer<T> input;
        cuda_detail::DeviceBuffer<T> weight;
        cuda_detail::DeviceBuffer<T> bias;
        cuda_detail::DeviceBuffer<T> out;
        T* bias_ptr = nullptr;
        if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
            cuda_detail::CopyTensorToDevice(param->w.get(), &weight) != 0 ||
            cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
            return -1;
        }
        if (param->bias != nullptr && param->bias->IsInitialized()) {
            if (cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) != 0) {
                return -1;
            }
            bias_ptr = bias.get();
        }
        const int64_t out_numel = param->out->numel();
        ImageShape4D input_shape;
        ImageShape4D output_shape;
        if (!DecodeImageShape4D(param->input->dims().data(), param->input->layout(), &input_shape) ||
            !DecodeImageShape4D(param->out->dims().data(), param->out->layout(), &output_shape)) {
            return -1;
        }
        const bool channel_last = IsChannelLastLayout(param->input->layout());
        const auto group = std::max(1, param->group);
        const bool is_pointwise = param->w->dims()[2] == 1 && param->w->dims()[3] == 1 && param->dilation_h == 1 &&
                                  param->dilation_w == 1;
        const bool is_depthwise = group == input_shape.c && param->w->dims()[1] == 1 &&
                                  param->w->dims()[0] % input_shape.c == 0;
        if (is_pointwise) {
            PointwiseConv2D4DKernelCuda<T>
                <<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), weight.get(), bias_ptr, out.get(), input_shape.n, input_shape.c,
                    input_shape.h, input_shape.w, param->w->dims()[0], param->w->dims()[1],
                    output_shape.h, output_shape.w, param->stride_h, param->stride_w, param->pad_h,
                    param->pad_w, group, channel_last);
        } else if (is_depthwise) {
            DepthwiseConv2D4DKernelCuda<T>
                <<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                    input.get(), weight.get(), bias_ptr, out.get(), input_shape.n, input_shape.c,
                    input_shape.h, input_shape.w, param->w->dims()[0], param->w->dims()[2],
                    param->w->dims()[3], output_shape.h, output_shape.w, param->stride_h,
                    param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w, channel_last);
        } else {
            Conv2D4DKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(out_numel, cuda_detail::kCudaThreads)),
                                   cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
                input.get(), weight.get(), bias_ptr, out.get(), input_shape.n, input_shape.c,
                input_shape.h, input_shape.w, param->w->dims()[0], param->w->dims()[1],
                param->w->dims()[2], param->w->dims()[3], output_shape.h, output_shape.w,
                param->stride_h, param->stride_w, param->pad_h, param->pad_w, param->dilation_h, param->dilation_w,
                group, channel_last);
        }
        if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
            return -1;
        }
        return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
    } else {
        return -1;
    }
}

template <DataType dtype>
int RunConv2D(const void* kernel_identity, feather::operators::Conv2dParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

#ifdef FEATHER_WITH_CUDNN
    if (param->input->dims().size() == 4 && param->w->dims().size() == 4 && param->out->dims().size() == 4) {
        cuda_detail::DeviceBuffer<T> input;
        cuda_detail::DeviceBuffer<T> weight;
        cuda_detail::DeviceBuffer<T> bias;
        cuda_detail::DeviceBuffer<T> out;
        const bool has_bias = param->bias != nullptr && param->bias->IsInitialized();
        if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) == 0 &&
            cuda_detail::CopyTensorToDevice(param->w.get(), &weight) == 0 &&
            cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) == 0 &&
            (!has_bias || cuda_detail::CopyTensorToDevice(param->bias.get(), &bias) == 0) &&
            RunConv2DWithCudnn<dtype>(kernel_identity, param, &input, &weight, &bias, &out, has_bias)) {
            return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
        }
    }
#else
    (void)kernel_identity;
#endif
    return RunConv2DFallback<dtype>(param);
}

}  // namespace

template <>
int32_t Conv2DKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunConv2D<DataType::FP32>(this, static_cast<feather::operators::Conv2dParam*>(param_), "CUDA::Conv2D::FP32");
}

template <>
int32_t Conv2DKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunConv2D<DataType::FP16>(this, static_cast<feather::operators::Conv2dParam*>(param_), "CUDA::Conv2D::FP16");
}

void EnsureCudaConv2DKernelsRegistered() { (void)g_cuda_conv2d_kernels_registered; }

}  // namespace kernel
}  // namespace feather
