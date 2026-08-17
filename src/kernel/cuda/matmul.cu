#include "src/kernel/matmul.h"

#include <algorithm>
#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/linear_kernels.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

struct CudaBatchedMatMulSpec {
    cuda_detail::CudaShape a_shape{};
    cuda_detail::CudaShape b_shape{};
    int64_t batch_strides[cuda_detail::kMaxCudaRank]{};
    int batch_rank{0};
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
};

bool g_cuda_matmul_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "MatMul",
                              []() { return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "MatMul",
                              []() { return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "MatMul",
                              []() { return std::make_unique<MatMulKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <typename T>
__global__ void BatchedMatMulKernelCuda(const T* a, const T* b, T* out, CudaBatchedMatMulSpec spec) {
    __shared__ float a_tile[cuda_detail::kLinearTile][cuda_detail::kLinearTile];
    __shared__ float b_tile[cuda_detail::kLinearTile][cuda_detail::kLinearTile];

    const int64_t batch = static_cast<int64_t>(blockIdx.z);
    const int64_t row = static_cast<int64_t>(blockIdx.y) * cuda_detail::kLinearTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * cuda_detail::kLinearTile + threadIdx.x;
    const int a_batch_rank = spec.a_shape.rank - 2;
    const int b_batch_rank = spec.b_shape.rank - 2;
    const int a_gap = spec.batch_rank - a_batch_rank;
    const int b_gap = spec.batch_rank - b_batch_rank;

    int64_t remaining = batch;
    int64_t a_batch_offset = 0;
    int64_t b_batch_offset = 0;
    for (int axis = 0; axis < spec.batch_rank; ++axis) {
        const int64_t coordinate = remaining / spec.batch_strides[axis];
        remaining %= spec.batch_strides[axis];
        const int a_axis = axis - a_gap;
        if (a_axis >= 0 && spec.a_shape.dims[a_axis] != 1) {
            a_batch_offset += coordinate * spec.a_shape.strides[a_axis];
        }
        const int b_axis = axis - b_gap;
        if (b_axis >= 0 && spec.b_shape.dims[b_axis] != 1) {
            b_batch_offset += coordinate * spec.b_shape.strides[b_axis];
        }
    }

    float sum = 0.0f;
    for (int64_t tile = 0; tile < spec.k; tile += cuda_detail::kLinearTile) {
        const int64_t a_col = tile + threadIdx.x;
        const int64_t b_row = tile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            row < spec.m && a_col < spec.k
                ? cuda_detail::ReadDevice(a, a_batch_offset + row * spec.a_shape.strides[a_batch_rank] +
                                                  a_col * spec.a_shape.strides[a_batch_rank + 1])
                : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < spec.k && col < spec.n
                ? cuda_detail::ReadDevice(b, b_batch_offset + b_row * spec.b_shape.strides[b_batch_rank] +
                                                  col * spec.b_shape.strides[b_batch_rank + 1])
                : 0.0f;
        __syncthreads();

        for (int i = 0; i < cuda_detail::kLinearTile; ++i) {
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < spec.m && col < spec.n) {
        cuda_detail::WriteDevice(out, batch * spec.m * spec.n + row * spec.n + col, sum);
    }
}

template <DataType dtype>
int RunBatchedMatMul(feather::operators::MatMulParam* param) {
    using T = cuda_detail::StorageT<dtype>;
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const auto& out_dims = param->out->dims().data();
    const int a_rank = static_cast<int>(a_dims.size());
    const int b_rank = static_cast<int>(b_dims.size());
    const int out_rank = static_cast<int>(out_dims.size());
    if (a_rank < 2 || b_rank < 2 || out_rank < 2 || a_rank > cuda_detail::kMaxCudaRank ||
        b_rank > cuda_detail::kMaxCudaRank || out_rank > cuda_detail::kMaxCudaRank) {
        return -1;
    }

    CudaBatchedMatMulSpec spec{};
    if (!cuda_detail::MakeCudaShape(a_dims, &spec.a_shape) || !cuda_detail::MakeCudaShape(b_dims, &spec.b_shape)) {
        return -1;
    }
    spec.batch_rank = out_rank - 2;
    if (spec.batch_rank != std::max(a_rank, b_rank) - 2 || a_dims[a_rank - 1] != b_dims[b_rank - 2] ||
        out_dims[out_rank - 2] != a_dims[a_rank - 2] || out_dims[out_rank - 1] != b_dims[b_rank - 1]) {
        return -1;
    }
    spec.m = a_dims[a_rank - 2];
    spec.k = a_dims[a_rank - 1];
    spec.n = b_dims[b_rank - 1];

    int64_t batch_count = 1;
    const int a_batch_rank = a_rank - 2;
    const int b_batch_rank = b_rank - 2;
    for (int axis = spec.batch_rank - 1; axis >= 0; --axis) {
        spec.batch_strides[axis] = batch_count;
        const int a_axis = axis - (spec.batch_rank - a_batch_rank);
        const int b_axis = axis - (spec.batch_rank - b_batch_rank);
        const int64_t a_dim = a_axis < 0 ? 1 : a_dims[a_axis];
        const int64_t b_dim = b_axis < 0 ? 1 : b_dims[b_axis];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return -1;
        }
        const int64_t expected_dim = std::max(a_dim, b_dim);
        if (out_dims[axis] != expected_dim || expected_dim <= 0 || batch_count > 65535 / expected_dim) {
            return -1;
        }
        batch_count *= expected_dim;
    }

    cuda_detail::DeviceBuffer<T> a;
    cuda_detail::DeviceBuffer<T> b;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &a) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &b) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    const dim3 block(cuda_detail::kLinearTile, cuda_detail::kLinearTile);
    const dim3 grid(static_cast<unsigned int>(cuda_detail::DivUp(spec.n, cuda_detail::kLinearTile)),
                    static_cast<unsigned int>(cuda_detail::DivUp(spec.m, cuda_detail::kLinearTile)),
                    static_cast<unsigned int>(batch_count));
    BatchedMatMulKernelCuda<T><<<grid, block, 0, cuda_detail::InferenceStream()>>>(a.get(), b.get(), out.get(), spec);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

template <DataType dtype>
int RunMatMul(feather::operators::MatMulParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    if (a_dims.size() < 2 || b_dims.size() < 2 || a_dims.back() != b_dims[b_dims.size() - 2]) {
        return -1;
    }
    if (b_dims.size() != 2) {
        return RunBatchedMatMul<dtype>(param);
    }
    const int64_t m = cuda_detail::ComputeProduct(a_dims, 0, a_dims.size() - 1);
    const int64_t k = a_dims.back();
    const int64_t n = b_dims.back();
    cuda_detail::DeviceBuffer<T> a;
    cuda_detail::DeviceBuffer<T> b;
    cuda_detail::DeviceBuffer<T> out;
    if (cuda_detail::CopyTensorToDevice(param->a.get(), &a) != 0 ||
        cuda_detail::CopyTensorToDevice(param->b.get(), &b) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    if (cuda_detail::LaunchCublasMatMul<dtype>(a.get(), b.get(), out.get(), m, k, n, 1.0f, false) != 0 ||
        cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

}  // namespace

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunMatMul<DataType::FP32>(static_cast<feather::operators::MatMulParam*>(param_), "CUDA::MatMul::FP32");
}

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunMatMul<DataType::FP16>(static_cast<feather::operators::MatMulParam*>(param_), "CUDA::MatMul::FP16");
}

template <>
int32_t MatMulKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunMatMul<DataType::BF16>(static_cast<feather::operators::MatMulParam*>(param_), "CUDA::MatMul::BF16");
}

void EnsureCudaMatMulKernelsRegistered() { (void)g_cuda_matmul_kernels_registered; }

}  // namespace kernel
}  // namespace feather
