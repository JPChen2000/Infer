#include "src/kernel/gemm.h"

#include <functional>
#include <numeric>

#include "src/kernel/common/kernel_io.h"
#include "src/kernel/common/int8_kernel_utils.h"
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
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT8, "Gemm",
                                                []() { return std::make_unique<GemmKernel<DeviceType::COMMON, DataType::INT8>>(); });
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

template <>
int32_t GemmKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::Gemm::INT8");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 || param->trans_a ||
        !std::isfinite(param->alpha) || !std::isfinite(param->beta)) {
        return -1;
    }
    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const int64_t k = a_dims.back();
    const int64_t rows = param->a->numel() / k;
    const int64_t b_k = param->trans_b ? b_dims[1] : b_dims[0];
    const int64_t channels = param->trans_b ? b_dims[0] : b_dims[1];
    std::vector<int64_t> expected_output = a_dims;
    expected_output.back() = channels;
    if (k <= 0 || rows <= 0 || channels <= 0 || b_k != k || param->out->dims().data() != expected_output) {
        return -1;
    }

    int8_detail::QuantizationView input_quantization;
    int8_detail::QuantizationView weight_quantization;
    int8_detail::QuantizationView output_quantization;
    const int64_t weight_axis = param->trans_b ? 0 : 1;
    if (!int8_detail::BuildInputQuantizationView(param->a, &input_quantization) ||
        !int8_detail::BuildWeightQuantizationView(param->b, weight_axis, channels, &weight_quantization) ||
        !int8_detail::BuildOutputQuantizationView(param->out, &output_quantization) ||
        !int8_detail::ValidateLinearBias(param->bias, rows, channels)) {
        return -1;
    }

    const int8_t* lhs = param->a->data<int8_t>();
    const int8_t* rhs = param->b->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            int64_t dot = 0;
            const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
            for (int64_t index = 0; index < k; ++index) {
                const int64_t rhs_offset = param->trans_b ? channel * k + index : index * channels + channel;
                dot += static_cast<int64_t>(static_cast<int32_t>(lhs[row * k + index]) - input_quantization.zero_point) *
                       (static_cast<int32_t>(rhs[rhs_offset]) - weight_zero_point);
                if (!int8_detail::FitsInt32(dot)) {
                    return -1;
                }
            }
            const int32_t bias = int8_detail::ReadLinearBias(param->bias, row, channel, channels);
            if (!int8_detail::FitsInt32(bias)) {
                return -1;
            }
            const double accumulator = static_cast<double>(param->alpha) * static_cast<double>(dot) +
                                       static_cast<double>(param->beta) * static_cast<double>(bias);
            const double scale = static_cast<double>(input_quantization.scale) *
                                 weight_quantization.scale_for(static_cast<size_t>(channel));
            if (!int8_detail::QuantizeReal(accumulator * scale, output_quantization,
                                           &output[row * channels + channel])) {
                return -1;
            }
        }
    }
    return 0;
}

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP32> GemmCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Gemm, GemmCommonFP32Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::FP16> GemmCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Gemm, GemmCommonFP16Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::BF16> GemmCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Gemm, GemmCommonBF16Kernel);

typedef feather::kernel::GemmKernel<DeviceType::COMMON, DataType::INT8> GemmCommonINT8Kernel;
REGISTER_KERNEL(COMMON, INT8, Gemm, GemmCommonINT8Kernel);

void EnsureCommonGemmKernelsRegistered() { (void)g_gemm_kernels_registered; }

void EnsureGemmKernelsRegistered() {
    EnsureCommonGemmKernelsRegistered();
    EnsureX86GemmKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
