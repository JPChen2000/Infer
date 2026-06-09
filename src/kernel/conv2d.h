#ifndef FEATHER_KERNEL_CONV_H
#define FEATHER_KERNEL_CONV_H
#include <array>
#include <limits>
#include <typeinfo>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "util/logger.h"
#ifdef FEATHER_WITH_CUDNN
#include <cudnn.h>
#endif
using feather::Tensor;
namespace feather {
namespace kernel {

enum class CudaConv2DBackend {
    kUnknown,
    kFallback,
    kCudnn,
};

void EnsureCommonConv2DKernelsRegistered();
void EnsureX86Conv2DKernelsRegistered();
void EnsureConv2DKernelsRegistered();

namespace detail {
inline thread_local CudaConv2DBackend g_last_cuda_conv2d_backend = CudaConv2DBackend::kUnknown;
}

inline CudaConv2DBackend LastCudaConv2DBackend() { return detail::g_last_cuda_conv2d_backend; }
inline void ResetLastCudaConv2DBackend() { detail::g_last_cuda_conv2d_backend = CudaConv2DBackend::kUnknown; }
inline void SetLastCudaConv2DBackend(CudaConv2DBackend backend) { detail::g_last_cuda_conv2d_backend = backend; }

#ifdef FEATHER_WITH_CUDNN
struct CudnnConvSelectionShape {
    DataType dtype{DataType::UNKNOWN};
    DataLayout layout{DataLayout::ND};
    std::array<int64_t, 4> input_dims{};
    std::array<int64_t, 4> weight_dims{};
    std::array<int64_t, 4> output_dims{};
    int dilation_h{1};
    int dilation_w{1};
    int group{1};
};

inline bool IsCudaConv2DPointwiseSelection(const CudnnConvSelectionShape& shape) {
    return shape.weight_dims[2] == 1 && shape.weight_dims[3] == 1 && shape.dilation_h == 1 && shape.dilation_w == 1;
}

inline bool IsCudaConv2DDepthwiseSelection(const CudnnConvSelectionShape& shape) {
    const int group = std::max(1, shape.group);
    return group == shape.input_dims[1] && shape.weight_dims[1] == 1 && shape.weight_dims[0] % shape.input_dims[1] == 0;
}

inline bool IsCudaConv2DGroupedSelection(const CudnnConvSelectionShape& shape) {
    return std::max(1, shape.group) > 1;
}

inline bool IsCudaConv2DImplicitGemmLikeAlgo(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return true;
        default:
            return false;
    }
}

inline bool IsCudaConv2DTensorOpMath(cudnnMathType_t math_type) {
    return math_type == CUDNN_TENSOR_OP_MATH || math_type == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
}

inline size_t PreferredCudaConv2DWorkspaceCapBytes(const CudnnConvSelectionShape& shape) {
    if (IsCudaConv2DPointwiseSelection(shape)) {
        return static_cast<size_t>(32) << 20;
    }
    if (IsCudaConv2DGroupedSelection(shape)) {
        return static_cast<size_t>(16) << 20;
    }
    return std::numeric_limits<size_t>::max();
}

inline int SelectPreferredCudaConv2DPerfIndex(const CudnnConvSelectionShape& shape,
                                              const cudnnConvolutionFwdAlgoPerf_t* perf,
                                              int count) {
    if (perf == nullptr || count <= 0) {
        return -1;
    }

    const bool prefer_implicit_gemm =
        IsCudaConv2DPointwiseSelection(shape) || IsCudaConv2DDepthwiseSelection(shape) || IsCudaConv2DGroupedSelection(shape);
    const bool prefer_tensor_op = shape.dtype == DataType::FP16 || prefer_implicit_gemm;
    const size_t workspace_cap = PreferredCudaConv2DWorkspaceCapBytes(shape);

    int best_success = -1;
    int best_capped = -1;
    int best_tensor = -1;
    int best_implicit_gemm = -1;
    int best_tensor_implicit_gemm = -1;
    int best_capped_tensor = -1;
    int best_capped_implicit_gemm = -1;
    int best_capped_tensor_implicit_gemm = -1;

    for (int i = 0; i < count; ++i) {
        if (perf[i].status != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        if (best_success == -1) {
            best_success = i;
        }

        const bool is_capped = perf[i].memory <= workspace_cap;
        const bool is_tensor_op = IsCudaConv2DTensorOpMath(perf[i].mathType);
        const bool is_implicit_gemm = IsCudaConv2DImplicitGemmLikeAlgo(perf[i].algo);

        if (is_capped && best_capped == -1) {
            best_capped = i;
        }
        if (is_tensor_op && best_tensor == -1) {
            best_tensor = i;
        }
        if (is_implicit_gemm && best_implicit_gemm == -1) {
            best_implicit_gemm = i;
        }
        if (is_tensor_op && is_implicit_gemm && best_tensor_implicit_gemm == -1) {
            best_tensor_implicit_gemm = i;
        }
        if (is_capped && is_tensor_op && best_capped_tensor == -1) {
            best_capped_tensor = i;
        }
        if (is_capped && is_implicit_gemm && best_capped_implicit_gemm == -1) {
            best_capped_implicit_gemm = i;
        }
        if (is_capped && is_tensor_op && is_implicit_gemm && best_capped_tensor_implicit_gemm == -1) {
            best_capped_tensor_implicit_gemm = i;
        }
    }

    if (prefer_implicit_gemm) {
        if (best_capped_tensor_implicit_gemm != -1) {
            return best_capped_tensor_implicit_gemm;
        }
        if (best_capped_implicit_gemm != -1) {
            return best_capped_implicit_gemm;
        }
        if (best_tensor_implicit_gemm != -1) {
            return best_tensor_implicit_gemm;
        }
        if (best_implicit_gemm != -1) {
            return best_implicit_gemm;
        }
    }

    if (prefer_tensor_op) {
        if (best_capped_tensor != -1) {
            return best_capped_tensor;
        }
        if (best_tensor != -1) {
            return best_tensor;
        }
    }

    if (best_capped != -1) {
        return best_capped;
    }
    return best_success;
}
#endif

template <DeviceType dev, DataType dtype>
class Conv2DKernel : public KernelBase {
   public:
    virtual int32_t compute() {
        LOG_INFO("use of unimplement kernel %s \n", typeid(*this).name());
        return 0;
    }

   protected:
    const Tensor* cached_weight_tensor_{nullptr};
    const Tensor* cached_bias_tensor_{nullptr};
    std::vector<float> cached_weight_buffer_;
    std::vector<float> cached_bias_buffer_;
    std::vector<float> cached_packed_input_buffer_;
    std::vector<float> cached_direct_weight_oc8_buffer_;
    std::vector<float> cached_direct_bias_oc8_buffer_;
    std::vector<float> cached_winograd_weight_oc8_buffer_;
    std::vector<float> cached_winograd_weight_buffer_;
};
}  // namespace kernel
}  // namespace feather
#endif  // FEATHER_KERNEL_CONV_H
