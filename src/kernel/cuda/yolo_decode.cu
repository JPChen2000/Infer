#include "src/kernel/yolo_decode.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_yolo_decode_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "YoloDecode",
                              []() { return std::make_unique<YoloDecodeKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "YoloDecode",
                              []() { return std::make_unique<YoloDecodeKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

__device__ inline float SigmoidDevice(float value) {
    return 1.0f / (1.0f + expf(-value));
}

template <typename T>
__global__ void YoloDecodeCudaKernel(const T* input, const T* xy_scale_tensor, const T* grid,
                                     const T* stride_tensor, const T* wh_scale_tensor, const T* anchor_grid,
                                     T* out, int64_t output_numel, int64_t batch, int64_t channels,
                                     int64_t height, int64_t width, int64_t anchors, int64_t attrs,
                                     int64_t grid_batch) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= output_numel) {
        return;
    }

    const int64_t attr = linear % attrs;
    const int64_t flattened_point = (linear / attrs) % (anchors * height * width);
    const int64_t n = linear / (attrs * anchors * height * width);
    const int64_t x = flattened_point % width;
    const int64_t y = (flattened_point / width) % height;
    const int64_t anchor = flattened_point / (height * width);
    if (n >= batch) {
        return;
    }

    const int64_t channel = anchor * attrs + attr;
    const int64_t input_offset = ((n * channels + channel) * height + y) * width + x;
    const float value = SigmoidDevice(cuda_detail::ReadDevice(input, input_offset));
    const float xy_scale = cuda_detail::ReadDevice(xy_scale_tensor, 0);
    const float stride = cuda_detail::ReadDevice(stride_tensor, 0);
    const float wh_scale = cuda_detail::ReadDevice(wh_scale_tensor, 0);
    const int64_t grid_n = grid_batch == 1 ? 0 : n;

    float decoded = value;
    if (attr < 2) {
        const int64_t grid_offset = ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + attr;
        decoded = (value * xy_scale + cuda_detail::ReadDevice(grid, grid_offset)) * stride;
    } else if (attr < 4) {
        const int64_t anchor_offset = ((((grid_n * anchors + anchor) * height + y) * width + x) * 2) + (attr - 2);
        const float scaled = value * wh_scale;
        decoded = scaled * scaled * cuda_detail::ReadDevice(anchor_grid, anchor_offset);
    }
    cuda_detail::WriteDevice(out, linear, decoded);
}

template <DataType dtype>
int RunYoloDecode(feather::operators::YoloDecodeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->xy_scale == nullptr ||
        param->grid == nullptr || param->stride == nullptr || param->wh_scale == nullptr ||
        param->anchor_grid == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto& in_dims = param->input->dims().data();
    const auto& grid_dims = param->grid->dims().data();
    if (in_dims.size() != 4 || grid_dims.size() != 5) {
        return -1;
    }
    const int64_t batch = in_dims[0];
    const int64_t channels = in_dims[1];
    const int64_t height = in_dims[2];
    const int64_t width = in_dims[3];
    const int64_t anchors = grid_dims[1];
    if (anchors <= 0 || channels % anchors != 0) {
        return -1;
    }
    const int64_t attrs = channels / anchors;
    const int64_t output_numel = param->out->numel();

    param->out->set_data_type(dtype);
    cuda_detail::DeviceBuffer<T> input;
    cuda_detail::DeviceBuffer<T> xy_scale;
    cuda_detail::DeviceBuffer<T> grid;
    cuda_detail::DeviceBuffer<T> stride;
    cuda_detail::DeviceBuffer<T> wh_scale;
    cuda_detail::DeviceBuffer<T> anchor_grid;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->input.get(), &input) != 0 ||
        cuda_detail::CopyTensorToDevice(param->xy_scale.get(), &xy_scale) != 0 ||
        cuda_detail::CopyTensorToDevice(param->grid.get(), &grid) != 0 ||
        cuda_detail::CopyTensorToDevice(param->stride.get(), &stride) != 0 ||
        cuda_detail::CopyTensorToDevice(param->wh_scale.get(), &wh_scale) != 0 ||
        cuda_detail::CopyTensorToDevice(param->anchor_grid.get(), &anchor_grid) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }

    YoloDecodeCudaKernel<T><<<static_cast<int>(cuda_detail::DivUp(output_numel, cuda_detail::kCudaThreads)),
                              cuda_detail::kCudaThreads, 0, cuda_detail::InferenceStream()>>>(
        input.get(), xy_scale.get(), grid.get(), stride.get(), wh_scale.get(), anchor_grid.get(), out.get(),
        output_numel, batch, channels, height, width, anchors, attrs, grid_dims[0]);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t YoloDecodeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunYoloDecode<DataType::FP32>(static_cast<feather::operators::YoloDecodeParam*>(param_),
                                        "CUDA::YoloDecode::FP32");
}

template <>
int32_t YoloDecodeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunYoloDecode<DataType::FP16>(static_cast<feather::operators::YoloDecodeParam*>(param_),
                                        "CUDA::YoloDecode::FP16");
}

void EnsureCudaYoloDecodeKernelsRegistered() { (void)g_cuda_yolo_decode_kernels_registered; }

}  // namespace kernel
}  // namespace feather
