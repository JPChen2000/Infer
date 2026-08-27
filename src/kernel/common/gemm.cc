#include "src/kernel/gemm.h"

#include <functional>
#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool IsVectorBias(const Tensor* bias, int64_t n) {
    if (bias == nullptr || !bias->IsInitialized() || bias->dims().empty() ||
        bias->dims()[bias->dims().size() - 1] != n) {
        return false;
    }
    for (size_t index = 0; index + 1 < bias->dims().size(); ++index) {
        if (bias->dims()[index] != 1) {
            return false;
        }
    }
    return bias->numel() == n;
}

bool g_gemm_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::BF16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E4M3, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E5M2, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::FP8E5M2>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeGemmCommon(feather::operators::GemmParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    if (param->trans_a) {
        return -1;
    }
    const auto a_dims = param->a->dims().data();
    const int64_t k = a_dims.back();
    const int64_t m = param->a->numel() / k;
    const int64_t n = param->trans_b ? param->b->dims()[0] : param->b->dims()[1];
    param->out->set_data_type(dtype);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                const int64_t a_offset = param->trans_a ? t * m + i : i * k + t;
                const int64_t b_offset = param->trans_b ? j * k + t : t * n + j;
                sum += TensorIO<dtype>::Read(param->a.get(), a_offset) *
                       TensorIO<dtype>::Read(param->b.get(), b_offset);
            }
            sum *= param->alpha;
            if (param->bias != nullptr && param->bias->IsInitialized()) {
                if (IsVectorBias(param->bias.get(), n)) {
                    sum += param->beta * TensorIO<dtype>::Read(param->bias.get(), j);
                } else {
                    sum += param->beta * TensorIO<dtype>::Read(param->bias.get(), i * n + j);
                }
            }
            TensorIO<dtype>::Write(param->out.get(), i * n + j, sum);
        }
    }

    return 0;
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Gemm::FP32");
    return ComputeGemmCommon<DataType::FP32>(static_cast<feather::operators::GemmParam*>(param_));
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Gemm::FP16");
    return ComputeGemmCommon<DataType::FP16>(static_cast<feather::operators::GemmParam*>(param_));
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Gemm::BF16");
    return ComputeGemmCommon<DataType::BF16>(static_cast<feather::operators::GemmParam*>(param_));
}

template <DataType dtype>
int32_t ComputeCommonFp8Gemm(feather::operators::GemmParam* param) {
    return ComputeGemmCommon<dtype>(param);
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::Gemm::FP8E4M3");
    return ComputeCommonFp8Gemm<DataType::FP8E4M3>(static_cast<feather::operators::GemmParam*>(param_));
}

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::Gemm::FP8E5M2");
    return ComputeCommonFp8Gemm<DataType::FP8E5M2>(static_cast<feather::operators::GemmParam*>(param_));
}

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP32> GemmCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Gemm, GemmCommonFP32Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP16> GemmCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Gemm, GemmCommonFP16Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::BF16> GemmCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Gemm, GemmCommonBF16Kernel);

void EnsureCommonGemmKernelsRegistered() { (void)g_gemm_kernels_registered; }

void EnsureGemmKernelsRegistered() {
    EnsureCommonGemmKernelsRegistered();
    EnsureX86GemmKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
