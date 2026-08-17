#include "src/kernel/reduce_mean.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_reduce_mean_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "ReduceMean", []() {
        return std::make_unique<ReduceMeanKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

struct CudaReduceMeanSpec {
    cuda_detail::CudaShape input_shape{};
    cuda_detail::CudaShape output_shape{};
    int reduced_mask[cuda_detail::kMaxCudaRank]{};
    int reduced_axes[cuda_detail::kMaxCudaRank]{};
    int64_t reduced_strides[cuda_detail::kMaxCudaRank]{};
    int reduced_rank{0};
    int keepdims{0};
    int64_t reduce_count{1};
};

template <typename T>
__global__ void ReduceMeanKernelCuda(const T* input, T* output, int64_t output_numel, CudaReduceMeanSpec spec) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }

    int64_t input_base = 0;
    if (spec.keepdims != 0) {
        int64_t remaining = output_index;
        for (int axis = 0; axis < spec.input_shape.rank; ++axis) {
            const int64_t coordinate = remaining / spec.output_shape.strides[axis];
            remaining %= spec.output_shape.strides[axis];
            if (spec.reduced_mask[axis] == 0) {
                input_base += coordinate * spec.input_shape.strides[axis];
            }
        }
    } else {
        int64_t remaining = output_index;
        int output_axis = 0;
        for (int axis = 0; axis < spec.input_shape.rank; ++axis) {
            if (spec.reduced_mask[axis] != 0) {
                continue;
            }
            const int64_t coordinate = remaining / spec.output_shape.strides[output_axis];
            remaining %= spec.output_shape.strides[output_axis];
            input_base += coordinate * spec.input_shape.strides[axis];
            ++output_axis;
        }
    }

    float sum = 0.0f;
    for (int64_t reduced_index = 0; reduced_index < spec.reduce_count; ++reduced_index) {
        int64_t remaining = reduced_index;
        int64_t input_index = input_base;
        for (int axis = 0; axis < spec.reduced_rank; ++axis) {
            const int64_t coordinate = remaining / spec.reduced_strides[axis];
            remaining %= spec.reduced_strides[axis];
            input_index += coordinate * spec.input_shape.strides[spec.reduced_axes[axis]];
        }
        sum += cuda_detail::ReadDevice(input, input_index);
    }
    cuda_detail::WriteDevice(output, output_index, sum / static_cast<float>(spec.reduce_count));
}

template <DataType dtype>
int32_t RunReduceMean(feather::operators::ReduceMeanParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->data_type() != dtype) {
        return -1;
    }
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    if (input_dims.size() > cuda_detail::kMaxCudaRank || output_dims.size() > cuda_detail::kMaxCudaRank) {
        return -1;
    }

    std::vector<int64_t> axes = param->axes;
    const int64_t rank = static_cast<int64_t>(input_dims.size());
    if (axes.empty()) {
        axes.resize(static_cast<size_t>(rank));
        std::iota(axes.begin(), axes.end(), 0);
    }
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return -1;
        }
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
        return -1;
    }

    CudaReduceMeanSpec spec{};
    if (!cuda_detail::MakeCudaShape(input_dims, &spec.input_shape) ||
        !cuda_detail::MakeCudaShape(output_dims, &spec.output_shape)) {
        return -1;
    }
    spec.keepdims = param->keepdims ? 1 : 0;
    for (const int64_t axis : axes) {
        spec.reduced_mask[axis] = 1;
        spec.reduced_axes[spec.reduced_rank++] = static_cast<int>(axis);
        spec.reduce_count *= input_dims[axis];
    }
    int64_t reduced_stride = 1;
    for (int axis = spec.reduced_rank - 1; axis >= 0; --axis) {
        spec.reduced_strides[axis] = reduced_stride;
        reduced_stride *= input_dims[spec.reduced_axes[axis]];
    }
    const int expected_output_rank =
        spec.keepdims != 0 ? spec.input_shape.rank : spec.input_shape.rank - spec.reduced_rank;
    if ((expected_output_rank == 0 && spec.output_shape.rank != 1) ||
        (expected_output_rank > 0 && spec.output_shape.rank != expected_output_rank)) {
        return -1;
    }

    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t output_numel = param->out->numel();
    ReduceMeanKernelCuda<T>
        <<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(input.get(), output.get(), output_numel, spec);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

}  // namespace

template <>
int32_t ReduceMeanKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunReduceMean<DataType::FP32>(static_cast<feather::operators::ReduceMeanParam*>(param_),
                                         "CUDA::ReduceMean::FP32");
}

template <>
int32_t ReduceMeanKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunReduceMean<DataType::FP16>(static_cast<feather::operators::ReduceMeanParam*>(param_),
                                         "CUDA::ReduceMean::FP16");
}

template <>
int32_t ReduceMeanKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunReduceMean<DataType::BF16>(static_cast<feather::operators::ReduceMeanParam*>(param_),
                                         "CUDA::ReduceMean::BF16");
}

void EnsureCudaReduceMeanKernelsRegistered() { (void)g_cuda_reduce_mean_kernels_registered; }

}  // namespace kernel
}  // namespace feather
