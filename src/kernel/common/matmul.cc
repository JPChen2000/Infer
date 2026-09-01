#include "src/kernel/matmul.h"

#include <functional>
#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/common/int8_kernel_utils.h"
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
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::BF16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E4M3, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E5M2, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::FP8E5M2>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT8, "MatMul",
                                                []() { return std::make_unique<MatMulKernel<DeviceType::COMMON, DataType::INT8>>(); });
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

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::MatMul::BF16");
    return ComputeMatMulCommon<DataType::BF16>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <DataType dtype>
int32_t ComputeCommonFp8MatMul(feather::operators::MatMulParam* param) {
    return ComputeMatMulCommon<dtype>(param);
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::MatMul::FP8E4M3");
    return ComputeCommonFp8MatMul<DataType::FP8E4M3>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::MatMul::FP8E5M2");
    return ComputeCommonFp8MatMul<DataType::FP8E5M2>(static_cast<feather::operators::MatMulParam*>(param_));
}

template <>
int32_t MatMulKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::MatMul::INT8");
    auto* param = static_cast<feather::operators::MatMulParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const auto& out_dims = param->out->dims().data();
    const size_t a_rank = a_dims.size();
    const size_t b_rank = b_dims.size();
    const size_t out_rank = out_dims.size();
    if (a_rank < 2 || b_rank < 2 || out_rank < 2) {
        return -1;
    }
    const int64_t m = a_dims[a_rank - 2];
    const int64_t k = a_dims[a_rank - 1];
    const int64_t b_k = b_dims[b_rank - 2];
    const int64_t n = b_dims[b_rank - 1];
    if (m <= 0 || k <= 0 || n <= 0 || b_k != k || out_rank != std::max(a_rank, b_rank)) {
        return -1;
    }
    const size_t batch_rank = out_rank - 2;
    const size_t a_batch_rank = a_rank - 2;
    const size_t b_batch_rank = b_rank - 2;
    if (a_batch_rank > batch_rank || b_batch_rank > batch_rank) {
        return -1;
    }
    for (size_t axis = 0; axis < batch_rank; ++axis) {
        const int64_t a_dim = axis < batch_rank - a_batch_rank ? 1 : a_dims[axis - (batch_rank - a_batch_rank)];
        const int64_t b_dim = axis < batch_rank - b_batch_rank ? 1 : b_dims[axis - (batch_rank - b_batch_rank)];
        const int64_t expected = std::max(a_dim, b_dim);
        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) || out_dims[axis] != expected) {
            return -1;
        }
    }
    if (out_dims[out_rank - 2] != m || out_dims[out_rank - 1] != n) {
        return -1;
    }

    int8_detail::QuantizationView input_quantization;
    int8_detail::QuantizationView weight_quantization;
    int8_detail::QuantizationView output_quantization;
    if (!int8_detail::BuildInputQuantizationView(param->a, &input_quantization) ||
        !int8_detail::BuildWeightQuantizationView(param->b, static_cast<int64_t>(b_rank - 1), n,
                                                   &weight_quantization) ||
        !int8_detail::BuildOutputQuantizationView(param->out, &output_quantization)) {
        return -1;
    }

    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    const int64_t batch_count = batch_rank == 0
                                    ? 1
                                    : std::accumulate(out_dims.begin(), out_dims.begin() + batch_rank, int64_t{1},
                                                      std::multiplies<int64_t>());
    const auto batch_strides = ComputeStrides(std::vector<int64_t>(out_dims.begin(), out_dims.begin() + batch_rank));
    std::vector<int64_t> batch_coords(batch_rank, 0);
    const int8_t* lhs = param->a->data<int8_t>();
    const int8_t* rhs = param->b->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t remaining = batch;
        for (size_t axis = 0; axis < batch_rank; ++axis) {
            batch_coords[axis] = remaining / batch_strides[axis];
            remaining %= batch_strides[axis];
        }
        int64_t a_batch_offset = 0;
        const size_t a_gap = batch_rank - a_batch_rank;
        for (size_t axis = 0; axis < a_batch_rank; ++axis) {
            const int64_t coordinate = a_dims[axis] == 1 ? 0 : batch_coords[axis + a_gap];
            a_batch_offset += coordinate * a_strides[axis];
        }
        int64_t b_batch_offset = 0;
        const size_t b_gap = batch_rank - b_batch_rank;
        for (size_t axis = 0; axis < b_batch_rank; ++axis) {
            const int64_t coordinate = b_dims[axis] == 1 ? 0 : batch_coords[axis + b_gap];
            b_batch_offset += coordinate * b_strides[axis];
        }
        const int64_t output_batch_offset = batch * m * n;
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t channel = 0; channel < n; ++channel) {
                int64_t accumulator = 0;
                const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
                for (int64_t index = 0; index < k; ++index) {
                    accumulator += static_cast<int64_t>(static_cast<int32_t>(lhs[a_batch_offset + row * a_strides[a_rank - 2] + index]) -
                                                        input_quantization.zero_point) *
                                  (static_cast<int32_t>(rhs[b_batch_offset + index * b_strides[b_rank - 2] + channel]) -
                                   weight_zero_point);
                    if (!int8_detail::FitsInt32(accumulator)) {
                        return -1;
                    }
                }
                const double scale = static_cast<double>(input_quantization.scale) *
                                     weight_quantization.scale_for(static_cast<size_t>(channel));
                if (!int8_detail::QuantizeAccumulator(
                        accumulator, scale, output_quantization,
                        &output[output_batch_offset + row * n + channel])) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP32> MatMulCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, MatMul, MatMulCommonFP32Kernel);

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::FP16> MatMulCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, MatMul, MatMulCommonFP16Kernel);

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::BF16> MatMulCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, MatMul, MatMulCommonBF16Kernel);

typedef feather::kernel::MatMulKernel<DeviceType::COMMON, DataType::INT8> MatMulCommonINT8Kernel;
REGISTER_KERNEL(COMMON, INT8, MatMul, MatMulCommonINT8Kernel);

void EnsureCommonMatMulKernelsRegistered() { (void)g_matmul_kernels_registered; }

void EnsureMatMulKernelsRegistered() {
    EnsureCommonMatMulKernelsRegistered();
    EnsureX86MatMulKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
