#include "src/kernel/gather.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_gather_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Gather", []() {
        return std::make_unique<GatherKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Gather", []() {
        return std::make_unique<GatherKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Gather", []() {
        return std::make_unique<GatherKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    return true;
}();

struct CudaGatherSpec {
    cuda_detail::CudaShape data_shape{};
    cuda_detail::CudaShape indices_shape{};
    cuda_detail::CudaShape output_shape{};
    int axis{0};
};

template <typename T, typename IndexT>
__global__ void GatherKernelCuda(const T* data, const IndexT* indices, T* output, int64_t output_numel,
                                  CudaGatherSpec spec) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_numel) {
        return;
    }
    int64_t remaining = output_index;
    int64_t data_offset = 0;
    int64_t index_offset = 0;
    for (int output_axis = 0; output_axis < spec.output_shape.rank; ++output_axis) {
        const int64_t coordinate = remaining / spec.output_shape.strides[output_axis];
        remaining %= spec.output_shape.strides[output_axis];
        if (output_axis < spec.axis) {
            data_offset += coordinate * spec.data_shape.strides[output_axis];
        } else if (output_axis < spec.axis + spec.indices_shape.rank) {
            const int indices_axis = output_axis - spec.axis;
            index_offset += coordinate * spec.indices_shape.strides[indices_axis];
        } else {
            const int data_axis = output_axis - spec.indices_shape.rank + 1;
            data_offset += coordinate * spec.data_shape.strides[data_axis];
        }
    }
    int64_t gather_index = static_cast<int64_t>(indices[index_offset]);
    if (gather_index < 0) {
        gather_index += spec.data_shape.dims[spec.axis];
    }
    if (gather_index < 0 || gather_index >= spec.data_shape.dims[spec.axis]) {
        return;
    }
    data_offset += gather_index * spec.data_shape.strides[spec.axis];
    cuda_detail::WriteDevice(output, output_index, cuda_detail::ReadDevice(data, data_offset));
}

template <DataType dtype, typename IndexT>
int32_t RunGatherWithIndices(feather::operators::GatherParam* param, CudaGatherSpec spec,
                             const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    cuda_detail::DeviceBuffer<T> data;
    cuda_detail::DeviceBuffer<IndexT> indices;
    cuda_detail::DeviceBuffer<T> output;
    if (cuda_detail::CopyTensorToDevice(param->data.get(), &data) != 0 ||
        cuda_detail::CopyTensorToDevice(param->indices.get(), &indices) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &output) != 0) {
        return -1;
    }
    const int64_t output_numel = param->out->numel();
    GatherKernelCuda<T, IndexT>
        <<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)), cuda_detail::kCudaThreads, 0,
           cuda_detail::InferenceStream()>>>(data.get(), indices.get(), output.get(), output_numel, spec);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&output, param->out.get());
}

template <DataType dtype>
int32_t RunGather(feather::operators::GatherParam* param, const char* timer_name) {
    if (param == nullptr || param->data == nullptr || param->indices == nullptr || param->out == nullptr ||
        param->data->data_type() != dtype) {
        return -1;
    }
    const auto& data_dims = param->data->dims().data();
    const int rank = static_cast<int>(data_dims.size());
    const int axis = param->axis < 0 ? param->axis + rank : param->axis;
    if (axis < 0 || axis >= rank) {
        return -1;
    }
    CudaGatherSpec spec{};
    if (!cuda_detail::MakeCudaShape(data_dims, &spec.data_shape) ||
        !cuda_detail::MakeCudaShape(param->indices->dims().data(), &spec.indices_shape) ||
        !cuda_detail::MakeCudaShape(param->out->dims().data(), &spec.output_shape)) {
        return -1;
    }
    spec.axis = axis;
    const int expected_output_rank = spec.data_shape.rank - 1 + spec.indices_shape.rank;
    if (spec.output_shape.rank != expected_output_rank) {
        return -1;
    }
    if (param->indices->data_type() == DataType::INT64) {
        return RunGatherWithIndices<dtype, int64_t>(param, spec, timer_name);
    }
    if (param->indices->data_type() == DataType::INT32) {
        return RunGatherWithIndices<dtype, int32_t>(param, spec, timer_name);
    }
    return -1;
}

}  // namespace

template <>
int32_t GatherKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunGather<DataType::FP32>(static_cast<feather::operators::GatherParam*>(param_),
                                     "CUDA::Gather::FP32");
}

template <>
int32_t GatherKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunGather<DataType::FP16>(static_cast<feather::operators::GatherParam*>(param_),
                                     "CUDA::Gather::FP16");
}

template <>
int32_t GatherKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunGather<DataType::BF16>(static_cast<feather::operators::GatherParam*>(param_),
                                     "CUDA::Gather::BF16");
}

void EnsureCudaGatherKernelsRegistered() { (void)g_cuda_gather_kernels_registered; }

}  // namespace kernel
}  // namespace feather
