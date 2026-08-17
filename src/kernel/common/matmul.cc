#include "src/kernel/matmul.h"

#include <functional>
#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

bool g_matmul_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeMatMulCommon(feather::operators::MatMulParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    const auto a_dims = param->a->dims().data();
    const auto b_dims = param->b->dims().data();
    const auto out_dims = param->out->dims().data();
    const size_t a_rank = a_dims.size();
    const size_t b_rank = b_dims.size();
    const size_t out_rank = out_dims.size();
    if (a_rank < 2 || b_rank < 2 || out_rank < 2) {
        return -1;
    }
    const int64_t m = a_dims[a_rank - 2];
    const int64_t k = a_dims[a_rank - 1];
    const int64_t n = b_dims[b_rank - 1];
    param->out->set_data_type(dtype);

    const size_t batch_rank = out_rank - 2;
    const int64_t batch_count =
        batch_rank == 0 ? 1 : std::accumulate(out_dims.begin(), out_dims.begin() + batch_rank, int64_t{1},
                                              std::multiplies<int64_t>());
    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    std::vector<int64_t> batch_dims(out_dims.begin(), out_dims.begin() + static_cast<int64_t>(batch_rank));
    const auto batch_strides = ComputeStrides(batch_dims);
    std::vector<int64_t> batch_coords(batch_rank, 0);

    for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t remaining = batch;
        for (size_t axis = 0; axis < batch_rank; ++axis) {
            batch_coords[axis] = remaining / batch_strides[axis];
            remaining %= batch_strides[axis];
        }
        int64_t a_batch_offset = 0;
        const size_t a_batch_rank = a_rank - 2;
        const size_t a_gap = batch_rank - a_batch_rank;
        for (size_t axis = 0; axis < a_batch_rank; ++axis) {
            const int64_t coord = a_dims[axis] == 1 ? 0 : batch_coords[axis + a_gap];
            a_batch_offset += coord * a_strides[axis];
        }
        int64_t b_batch_offset = 0;
        const size_t b_batch_rank = b_rank - 2;
        const size_t b_gap = batch_rank - b_batch_rank;
        for (size_t axis = 0; axis < b_batch_rank; ++axis) {
            const int64_t coord = b_dims[axis] == 1 ? 0 : batch_coords[axis + b_gap];
            b_batch_offset += coord * b_strides[axis];
        }
        const int64_t out_batch_offset = batch * m * n;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t t = 0; t < k; ++t) {
                    sum += TensorIO<dtype>::Read(param->a.get(), a_batch_offset + i * a_strides[a_rank - 2] + t) *
                           TensorIO<dtype>::Read(param->b.get(), b_batch_offset + t * b_strides[b_rank - 2] + j);
                }
                TensorIO<dtype>::Write(param->out.get(), out_batch_offset + i * n + j, sum);
            }
        }
    }
    return 0;
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::MatMul::FP32");
    return ComputeMatMulCommon<DataType::FP32>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::MatMul::FP16");
    return ComputeMatMulCommon<DataType::FP16>(static_cast<feather::operators::MatMulParam*>(param_));
}

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP32> MatMulCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, MatMul, MatMulCommonFP32Kernel);

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP16> MatMulCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, MatMul, MatMulCommonFP16Kernel);

void EnsureCommonMatMulKernelsRegistered() { (void)g_matmul_kernels_registered; }

void EnsureMatMulKernelsRegistered() {
    EnsureCommonMatMulKernelsRegistered();
    EnsureX86MatMulKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
