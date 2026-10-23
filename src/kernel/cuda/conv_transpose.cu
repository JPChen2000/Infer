// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include "src/kernel/conv_transpose.h"

#include <algorithm>
#include <climits>
#include <memory>

#include "model/model_format.h"
#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace {
namespace cd = cuda_detail;
using operators::ConvTransposeParam;

bool AttributeFitsCuda(int64_t value) { return value >= 0 && value <= INT_MAX; }

Status AttributeSupport(const model::NodeDesc* node) {
    if (!node) return Status::Ok();  // Metadata-only clones validate bound parameters in Supports.
    for (const auto* name : {"strides", "dilations", "pads", "output_padding", "group"}) {
        const auto found = node->attributes.find(name);
        if (found == node->attributes.end()) continue;
        if (const auto* scalar = std::get_if<int64_t>(&found->second)) {
            if (AttributeFitsCuda(*scalar)) continue;
        } else if (const auto* values = std::get_if<std::vector<int64_t>>(&found->second)) {
            if (std::all_of(values->begin(), values->end(), AttributeFitsCuda)) continue;
        }
        return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose attributes must fit nonnegative int32");
    }
    return Status::Ok();
}

// Bound scalar dimensions/attributes so device coordinate arithmetic cannot
// overflow int64_t. Tensor products are checked separately by kernel_io.
Status TensorSupport(const Tensor* tensor) {
    if (!tensor || tensor->data_type() != DataType::FP32 ||
        !cd::HasValidCudaShape(tensor->dims().data())) {
        return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose requires valid FP32 tensors");
    }
    for (const auto dim : tensor->dims().data()) {
        if (dim > INT_MAX) {
            return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose dimensions must fit int32");
        }
    }
    return Status::Ok();
}

struct TransposeSpec {
    int64_t in_c, in_h, in_w, out_c, out_h, out_w;
    int64_t kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w;
    int64_t pad_h, pad_w, group;
};

// Each thread owns one output element, avoiding atomics and covering grouped,
// dilated, asymmetric padding and output_padding cases without cuDNN.
__global__ void ConvTransposeCuda(const float* input, const float* weight, const float* bias,
                                 float* output, int64_t count, TransposeSpec s) {
    const int64_t input_per_group = s.in_c / s.group;
    const int64_t output_per_group = s.out_c / s.group;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t ow = index % s.out_w, oh = index / s.out_w % s.out_h;
        const int64_t oc = index / s.out_w / s.out_h % s.out_c;
        const int64_t batch = index / s.out_w / s.out_h / s.out_c;
        const int64_t first_ic = oc / output_per_group * input_per_group;
        float value = bias ? bias[oc] : 0.0f;
        for (int64_t ic = first_ic; ic < first_ic + input_per_group; ++ic) {
            for (int64_t kh = 0; kh < s.kernel_h; ++kh) {
                const int64_t h = oh + s.pad_h - kh * s.dilation_h;
                if (h < 0 || h % s.stride_h != 0 || h / s.stride_h >= s.in_h) continue;
                for (int64_t kw = 0; kw < s.kernel_w; ++kw) {
                    const int64_t w = ow + s.pad_w - kw * s.dilation_w;
                    if (w < 0 || w % s.stride_w != 0 || w / s.stride_w >= s.in_w) continue;
                    const int64_t source = ((batch * s.in_c + ic) * s.in_h + h / s.stride_h) * s.in_w + w / s.stride_w;
                    const int64_t filter = ((ic * output_per_group + oc % output_per_group) * s.kernel_h + kh) * s.kernel_w + kw;
                    value += input[source] * weight[filter];
                }
            }
        }
        output[index] = value;
    }
}

__global__ void AddBiasCuda(float* output, const float* bias, int64_t count, int64_t channels, int64_t plane) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        output[i] += bias[i / plane % channels];
    }
}

#ifdef FEATHER_WITH_CUDNN
struct CudnnDescriptors {
    cudnnTensorDescriptor_t input{nullptr}, output{nullptr};
    cudnnFilterDescriptor_t filter{nullptr};
    cudnnConvolutionDescriptor_t convolution{nullptr};
    CudnnDescriptors() {
        (void)cudnnCreateTensorDescriptor(&input);
        (void)cudnnCreateTensorDescriptor(&output);
        (void)cudnnCreateFilterDescriptor(&filter);
        (void)cudnnCreateConvolutionDescriptor(&convolution);
    }
    ~CudnnDescriptors() {
        if (input) cudnnDestroyTensorDescriptor(input);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (convolution) cudnnDestroyConvolutionDescriptor(convolution);
    }
};

bool TryCudnn(const ConvTransposeParam& p, const float* input, const float* weight, float* output) {
    // cuDNN backward-data describes the corresponding forward convolution.
    // Asymmetric padding and extended output_padding use the generic kernel.
    if (p.pads[0] != p.pads[2] || p.pads[1] != p.pads[3] ||
        p.output_padding[0] >= p.strides[0] || p.output_padding[1] >= p.strides[1]) return false;
    const auto handle = cd::CudnnHandle();
    if (!handle) return false;
    CudnnDescriptors d;
    if (!d.input || !d.output || !d.filter || !d.convolution) return false;
    const auto& in = p.input->dims();
    const auto& out = p.out->dims();
    const auto& w = p.weight->dims();
    const int filter_dims[4] = {static_cast<int>(w[0]), static_cast<int>(w[1]),
                              static_cast<int>(w[2]), static_cast<int>(w[3])};
    if (cudnnSetTensor4dDescriptor(d.input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                  in[0], in[1], in[2], in[3]) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensor4dDescriptor(d.output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                  out[0], out[1], out[2], out[3]) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilterNdDescriptor(d.filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4,
                                  filter_dims) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolution2dDescriptor(d.convolution, p.pads[0], p.pads[1], p.strides[0], p.strides[1],
                                        p.dilations[0], p.dilations[1], CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionGroupCount(d.convolution, p.group) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionMathType(d.convolution, CUDNN_FMA_MATH) != CUDNN_STATUS_SUCCESS) return false;

    // Algorithm 1 uses FP32 GEMM with FMA math and avoids autotuning during
    // inference. Session-owned workspace stays on the same inference stream.
    constexpr auto algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t bytes = 0;
    if (cudnnGetConvolutionBackwardDataWorkspaceSize(handle, d.filter, d.input, d.convolution,
                                                     d.output, algorithm, &bytes) != CUDNN_STATUS_SUCCESS) return false;
    void* workspace = nullptr;
    if (cd::AcquireTemporaryDeviceBuffer(bytes, &workspace) != 0) return false;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cudnnConvolutionBackwardData(handle, &alpha, d.filter, weight, d.input, input,
                                                     d.convolution, algorithm, workspace, bytes, &beta,
                                                     d.output, output);
    cd::ReleaseTemporaryDeviceBuffer(workspace, bytes);
    return status == CUDNN_STATUS_SUCCESS;
}
#endif

class CudaConvTransposeKernel final : public KernelBase {
 public:
    Status Supports() const override {
        const auto* p = ParamAs<ConvTransposeParam>();
        std::vector<int64_t> shape;
        if (!p || !operators::ConvTransposeOutputShape(*p, &shape)) {
            return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose requires valid NCHW FP32 parameters");
        }
        for (const auto* tensor : {p->input.get(), p->weight.get(), p->out.get()}) {
            const auto status = TensorSupport(tensor);
            if (!status.ok()) return status;
        }
        if (p->bias && !TensorSupport(p->bias.get()).ok()) {
            return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose requires valid FP32 bias");
        }
        for (const auto value : {p->strides[0], p->strides[1], p->dilations[0], p->dilations[1],
                                 p->pads[0], p->pads[1], p->pads[2], p->pads[3], p->group,
                                 p->output_padding[0], p->output_padding[1]}) {
            if (!AttributeFitsCuda(value)) {
                return Status::Error(StatusCode::kUnsupported, "CUDA ConvTranspose attributes must fit int32");
            }
        }
        return Status::Ok();
    }

    int32_t compute() override {
        if (!Supports().ok()) return -1;
        const auto& p = *ParamAs<ConvTransposeParam>();
        std::vector<int64_t> shape;
        if (!operators::ConvTransposeOutputShape(p, &shape) ||
            !cd::IsTensorReady<DataType::FP32>(p.input.get()) ||
            !cd::IsTensorReady<DataType::FP32>(p.weight.get()) ||
            !cd::IsOutputReady<DataType::FP32>(p.out.get(), &shape) ||
            (p.bias && !cd::IsTensorReady<DataType::FP32>(p.bias.get()))) return -1;
        cd::DeviceBuffer<float> input, weight, bias, output;
        if (cd::CopyTensorToDevice(p.input.get(), &input) != 0 ||
            cd::CopyTensorToDevice(p.weight.get(), &weight) != 0 ||
            (p.bias && cd::CopyTensorToDevice(p.bias.get(), &bias) != 0) ||
            cd::AllocateTensorOnDevice(p.out.get(), &output) != 0) return -1;
        const auto& in = p.input->dims();
        const auto& w = p.weight->dims();
        const int64_t count = p.out->numel();
        const int blocks = static_cast<int>(std::min<int64_t>(65535, cd::DivUp(count, cd::kCudaThreads)));
        bool computed = false;
#ifdef FEATHER_WITH_CUDNN
        computed = TryCudnn(p, input.get(), weight.get(), output.get());
        if (computed && p.bias) {
            AddBiasCuda<<<blocks, cd::kCudaThreads, 0, cd::InferenceStream()>>>(
                output.get(), bias.get(), count, shape[1], shape[2] * shape[3]);
        }
#endif
        if (!computed) {
            const TransposeSpec spec{in[1], in[2], in[3], shape[1], shape[2], shape[3], w[2], w[3],
                                     p.strides[0], p.strides[1], p.dilations[0], p.dilations[1],
                                     p.pads[0], p.pads[1], p.group};
            ConvTransposeCuda<<<blocks, cd::kCudaThreads, 0, cd::InferenceStream()>>>(
                input.get(), weight.get(), bias.get(), output.get(), count, spec);
        }
        if (cd::CudaCheck(cudaGetLastError()) != 0) return -1;
        return cd::CopyDeviceToTensor(&output, p.out.get());
    }
};
}  // namespace

void EnsureCudaConvTransposeKernelsRegistered() {
    auto& dispatcher = KernelDispatcher::instance();
    static const bool registered = [&dispatcher] {
        dispatcher.RegisterCandidate(DeviceType::CUDA, DataType::FP32, DataLayout::ND, "ConvTranspose",
            "cuda_fp32_conv_transpose", [] { return std::make_unique<CudaConvTransposeKernel>(); },
            [](const KernelRequest& request) {
                for (const auto& tensor : request.tensors) {
                    if (!tensor) continue;
                    const auto status = TensorSupport(tensor.get());
                    if (!status.ok()) return status;
                }
                return AttributeSupport(request.node);
            });
        return true;
    }();
    (void)registered;
}
}  // namespace kernel
}  // namespace feather
