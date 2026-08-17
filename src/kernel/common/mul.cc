#include "src/kernel/mul.h"

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

bool InferBroadcastShape(const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
                         std::vector<int64_t>* out_dims) {
    if (out_dims == nullptr) {
        return false;
    }
    const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
    out_dims->assign(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
        const int64_t lhs_dim = i < out_rank - lhs_dims.size() ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
        const int64_t rhs_dim = i < out_rank - rhs_dims.size() ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return false;
        }
        (*out_dims)[i] = std::max(lhs_dim, rhs_dim);
    }
    return true;
}

int64_t ComputeBroadcastOffset(const std::vector<int64_t>& out_coords, const std::vector<int64_t>& in_dims,
                               const std::vector<int64_t>& in_strides) {
    const size_t rank_gap = out_coords.size() - in_dims.size();
    int64_t offset = 0;
    for (size_t i = 0; i < in_dims.size(); ++i) {
        const int64_t coord = in_dims[i] == 1 ? 0 : out_coords[i + rank_gap];
        offset += coord * in_strides[i];
    }
    return offset;
}

bool g_mul_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Mul",
                                                []() { return std::make_unique<MulKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Mul",
                                                []() { return std::make_unique<MulKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Mul",
                                                []() { return std::make_unique<MulKernel<DeviceType::COMMON, DataType::BF16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT64, "Mul",
                                                []() { return std::make_unique<MulKernel<DeviceType::COMMON, DataType::INT64>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeMulKernel(feather::operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        param->lhs->data_type() != dtype || param->rhs->data_type() != dtype) {
        return -1;
    }

    std::vector<int64_t> out_dims;
    if (!InferBroadcastShape(param->lhs->dims().data(), param->rhs->dims().data(), &out_dims)) {
        return -1;
    }
    param->out->set_data_type(dtype);
    const auto out_strides = ComputeStrides(out_dims);
    const auto lhs_strides = ComputeStrides(param->lhs->dims().data());
    const auto rhs_strides = ComputeStrides(param->rhs->dims().data());
    std::vector<int64_t> out_coords(out_dims.size(), 0);

    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        int64_t remaining = linear;
        for (size_t axis = 0; axis < out_dims.size(); ++axis) {
            out_coords[axis] = remaining / out_strides[axis];
            remaining %= out_strides[axis];
        }
        const int64_t lhs_offset = ComputeBroadcastOffset(out_coords, param->lhs->dims().data(), lhs_strides);
        const int64_t rhs_offset = ComputeBroadcastOffset(out_coords, param->rhs->dims().data(), rhs_strides);
        if constexpr (dtype == DataType::INT64) {
            param->out->mutable_data<int64_t>()[linear] =
                param->lhs->data<int64_t>()[lhs_offset] * param->rhs->data<int64_t>()[rhs_offset];
        } else {
            const float lhs = TensorIO<dtype>::Read(param->lhs.get(), lhs_offset);
            const float rhs = TensorIO<dtype>::Read(param->rhs.get(), rhs_offset);
            TensorIO<dtype>::Write(param->out.get(), linear, lhs * rhs);
        }
    }
    return 0;
}

template <>
int32_t MulKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Mul::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeMulKernel<DataType::FP32>(param);
}

template <>
int32_t MulKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Mul::FP16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeMulKernel<DataType::FP16>(param);
}

template <>
int32_t MulKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Mul::BF16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeMulKernel<DataType::BF16>(param);
}

template <>
int32_t MulKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Mul::INT64");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeMulKernel<DataType::INT64>(param);
}

typedef feather::kernel::MulKernel<DeviceType::COMMON, DataType::FP32> MulCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Mul, MulCommonFP32Kernel);

typedef feather::kernel::MulKernel<DeviceType::COMMON, DataType::FP16> MulCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Mul, MulCommonFP16Kernel);

void EnsureCommonMulKernelsRegistered() { (void)g_mul_kernels_registered; }

void EnsureMulKernelsRegistered() {
    EnsureCommonMulKernelsRegistered();
    EnsureX86MulKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
