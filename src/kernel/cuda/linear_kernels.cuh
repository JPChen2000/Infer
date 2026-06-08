#ifndef FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH
#define FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH

#include "src/kernel/cuda/kernel_io.cuh"

namespace feather {
namespace kernel {
namespace cuda_detail {

constexpr int kLinearTile = 16;

template <typename T>
__global__ void MatMulTiledKernelCuda(const T* a, const T* b, const T* bias, T* out, int64_t m, int64_t k, int64_t n,
                                      int bias_mode) {
    __shared__ float a_tile[kLinearTile][kLinearTile];
    __shared__ float b_tile[kLinearTile][kLinearTile];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kLinearTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kLinearTile + threadIdx.x;
    float sum = 0.0f;

    for (int64_t tile = 0; tile < k; tile += kLinearTile) {
        const int64_t a_col = tile + threadIdx.x;
        const int64_t b_row = tile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? ReadDevice(a, row * k + a_col) : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? ReadDevice(b, b_row * n + col) : 0.0f;
        __syncthreads();

        for (int i = 0; i < kLinearTile; ++i) {
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        const int64_t idx = row * n + col;
        if (bias != nullptr) {
            sum += ReadDevice(bias, bias_mode == 1 ? col : idx);
        }
        WriteDevice(out, idx, sum);
    }
}

template <typename T>
inline void LaunchMatMulKernelCuda(const T* a, const T* b, const T* bias, T* out, int64_t m, int64_t k, int64_t n,
                                   int bias_mode) {
    dim3 block(kLinearTile, kLinearTile);
    dim3 grid(static_cast<unsigned int>(DivUp(n, kLinearTile)), static_cast<unsigned int>(DivUp(m, kLinearTile)));
    MatMulTiledKernelCuda<T><<<grid, block, 0, InferenceStream()>>>(a, b, bias, out, m, k, n, bias_mode);
}

}  // namespace cuda_detail
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_CUDA_LINEAR_KERNELS_CUH
