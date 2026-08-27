#include "src/kernel/add.h"

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/fp8_host.h"
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

bool g_add_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Add",
                                                []() { return std::make_unique<AddKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Add",
                                                []() { return std::make_unique<AddKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Add",
                                                []() { return std::make_unique<AddKernel<DeviceType::COMMON, DataType::BF16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E4M3, "Add",
                                                []() { return std::make_unique<AddKernel<DeviceType::COMMON, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E5M2, "Add",
                                                []() { return std::make_unique<AddKernel<DeviceType::COMMON, DataType::FP8E5M2>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeAddKernel(feather::operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
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
        const float lhs = TensorIO<dtype>::Read(param->lhs.get(), lhs_offset);
        const float rhs = TensorIO<dtype>::Read(param->rhs.get(), rhs_offset);
        TensorIO<dtype>::Write(param->out.get(), linear, lhs + rhs);
    }
    return 0;
}

template <>
int32_t AddKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Add::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeAddKernel<DataType::FP32>(param);
}

template <>
int32_t AddKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Add::FP16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeAddKernel<DataType::FP16>(param);
}

template <>
int32_t AddKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Add::BF16");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    return ComputeAddKernel<DataType::BF16>(param);
}

template <DataType dtype>
int32_t ComputeAddFp8(feather::operators::BinaryParam* param) {
    return fp8_host::Binary<dtype, fp8_host::BinaryOp::kAdd>(param);
}

template <>
int32_t AddKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::Add::FP8E4M3");
    return ComputeAddFp8<DataType::FP8E4M3>(static_cast<feather::operators::BinaryParam*>(param_));
}

template <>
int32_t AddKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::Add::FP8E5M2");
    return ComputeAddFp8<DataType::FP8E5M2>(static_cast<feather::operators::BinaryParam*>(param_));
}

typedef feather::kernel::AddKernel<DeviceType::COMMON, DataType::FP32> AddCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Add, AddCommonFP32Kernel);

typedef feather::kernel::AddKernel<DeviceType::COMMON, DataType::FP16> AddCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Add, AddCommonFP16Kernel);

void EnsureCommonAddKernelsRegistered() { (void)g_add_kernels_registered; }

void EnsureAddKernelsRegistered() {
    EnsureCommonAddKernelsRegistered();
    EnsureX86AddKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
