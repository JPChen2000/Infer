#include "src/kernel/softmax.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_softmax_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Softmax",
                              []() { return std::make_unique<SoftmaxKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Softmax",
                              []() { return std::make_unique<SoftmaxKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Softmax",
                              []() { return std::make_unique<SoftmaxKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <typename T>
__global__ void SoftmaxKernelCuda(const T* input, T* out, int64_t rows, int64_t axis_dim, int64_t inner) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int64_t outer_idx = row / inner;
    const int64_t inner_idx = row % inner;
    const int64_t base = outer_idx * axis_dim * inner + inner_idx;
    float max_value = cuda_detail::ReadDevice(input, base);
    for (int64_t axis_idx = 1; axis_idx < axis_dim; ++axis_idx) {
        max_value = fmaxf(max_value, cuda_detail::ReadDevice(input, base + axis_idx * inner));
    }
    float sum = 0.0f;
    for (int64_t axis_idx = 0; axis_idx < axis_dim; ++axis_idx) {
        const float value = expf(cuda_detail::ReadDevice(input, base + axis_idx * inner) - max_value);
        cuda_detail::WriteDevice(out, base + axis_idx * inner, value);
        sum += value;
    }
    for (int64_t axis_idx = 0; axis_idx < axis_dim; ++axis_idx) {
        const int64_t offset = base + axis_idx * inner;
        cuda_detail::WriteDevice(out, offset, cuda_detail::ReadDevice(out, offset) / sum);
    }
}

template <DataType dtype>
int RunSoftmax(feather::operators::SoftmaxParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& dims = param->input->dims().data();
    const int rank = static_cast<int>(dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }
    int64_t outer = 1;
    int64_t inner = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= dims[static_cast<size_t>(i)];
    }
    for (int i = axis + 1; i < rank; ++i) {
        inner *= dims[static_cast<size_t>(i)];
    }
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    const int64_t rows = outer * inner;
    SoftmaxKernelCuda<T><<<static_cast<int>(cuda_detail::DivUp(rows, cuda_detail::kCudaThreads)),
                          cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), out.get(), rows, dims[static_cast<size_t>(axis)], inner);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t SoftmaxKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunSoftmax<DataType::FP32>(static_cast<feather::operators::SoftmaxParam*>(param_),
                                     "CUDA::Softmax::FP32");
}

template <>
int32_t SoftmaxKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunSoftmax<DataType::FP16>(static_cast<feather::operators::SoftmaxParam*>(param_),
                                     "CUDA::Softmax::FP16");
}

template <>
int32_t SoftmaxKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunSoftmax<DataType::BF16>(static_cast<feather::operators::SoftmaxParam*>(param_),
                                      "CUDA::Softmax::BF16");
}

void EnsureCudaSoftmaxKernelsRegistered() { (void)g_cuda_softmax_kernels_registered; }

}  // namespace kernel
}  // namespace feather
