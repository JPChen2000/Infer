#include "src/kernel/transpose.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

struct CudaTransposePerm {
    int rank{0};
    int perm[cuda_detail::kMaxCudaRank]{};
};

bool g_cuda_transpose_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Transpose",
                              []() { return std::make_unique<TransposeKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Transpose",
                              []() { return std::make_unique<TransposeKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Transpose",
                              []() { return std::make_unique<TransposeKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <typename T>
__global__ void TransposeKernelCuda(const T* input, T* out, int64_t numel, cuda_detail::CudaShape out_shape,
                                    cuda_detail::CudaShape in_shape, CudaTransposePerm perm) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) {
        return;
    }
    int64_t remaining = linear;
    int64_t in_offset = 0;
    for (int axis = 0; axis < out_shape.rank; ++axis) {
        const int64_t coord = remaining / out_shape.strides[axis];
        remaining %= out_shape.strides[axis];
        in_offset += coord * in_shape.strides[perm.perm[axis]];
    }
    out[linear] = input[in_offset];
}

template <DataType dtype>
int RunTranspose(feather::operators::TransposeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& in_dims = param->input->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (in_dims.size() > cuda_detail::kMaxCudaRank || out_dims.size() > cuda_detail::kMaxCudaRank ||
        in_dims.size() != param->perm.size()) {
        return -1;
    }
    cuda_detail::CudaShape in_shape;
    cuda_detail::CudaShape out_shape;
    CudaTransposePerm perm;
    if (!cuda_detail::MakeCudaShape(in_dims, &in_shape) || !cuda_detail::MakeCudaShape(out_dims, &out_shape)) {
        return -1;
    }
    perm.rank = static_cast<int>(param->perm.size());
    for (size_t i = 0; i < param->perm.size(); ++i) {
        if (param->perm[i] < 0 || param->perm[i] >= static_cast<int64_t>(in_dims.size())) {
            return -1;
        }
        perm.perm[i] = static_cast<int>(param->perm[i]);
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    TransposeKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(param->out->numel(), cuda_detail::kCudaThreads)),
                            cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), param->out->numel(), out_shape, in_shape, perm);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t TransposeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunTranspose<DataType::FP32>(static_cast<feather::operators::TransposeParam*>(param_),
                                       "CUDA::Transpose::FP32");
}

template <>
int32_t TransposeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunTranspose<DataType::FP16>(static_cast<feather::operators::TransposeParam*>(param_),
                                       "CUDA::Transpose::FP16");
}

template <>
int32_t TransposeKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunTranspose<DataType::BF16>(static_cast<feather::operators::TransposeParam*>(param_),
                                        "CUDA::Transpose::BF16");
}

void EnsureCudaTransposeKernelsRegistered() { (void)g_cuda_transpose_kernels_registered; }

}  // namespace kernel
}  // namespace feather
