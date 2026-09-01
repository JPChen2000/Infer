#include "core/tensor.h"
#include "util/logger.h"
#include "src/kernel/common/kernel_io.h"
#include "src/kernel/common/int8_kernel_utils.h"
#include "src/kernel/fp8_host.h"
#include "util/timer.h"
#include "util/types.h"
#include "src/kernel/fc.h"
#include "src/operator/params.h"
using feather::DataType;
using feather::Tensor;
using feather::operators::FcParam;

namespace feather {
namespace kernel {

namespace {
bool g_fc_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E4M3, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP8E5M2, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::FP8E5M2>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT8, "FC",
                                               []() { return std::make_unique<FcKernel<DeviceType::COMMON, DataType::INT8>>(); });
    return true;
}();
}  // namespace

template <DataType dtype>
int32_t ComputeFcCommon(feather::operators::FcParam* param) {
    int64_t m = 0;
    int64_t n = 0;
    int64_t c = 0;
    if (!fp8_host::ValidateFc<dtype>(param, &m, &n, &c)) return -1;

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n; ++k) {
                sum += TensorIO<dtype>::Read(param->input.get(), i * n + k) *
                       TensorIO<dtype>::Read(param->w.get(), k * c + j);
            }
            if (param->bias != nullptr) {
                if (param->bias->dims().size() == 1) {
                    sum += TensorIO<dtype>::Read(param->bias.get(), j);
                } else {
                    sum += TensorIO<dtype>::Read(param->bias.get(), i * c + j);
                }
            }
            TensorIO<dtype>::Write(param->out.get(), i * c + j, sum);
        }
    }

    return 0;
}

template<>
int32_t FcKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::FC::FP32");
    return ComputeFcCommon<DataType::FP32>(static_cast<FcParam*>(param_));
}

template<>
int32_t FcKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::FC::FP16");
    return ComputeFcCommon<DataType::FP16>(static_cast<FcParam*>(param_));
}

template <DataType dtype>
int32_t ComputeCommonFp8Fc(feather::operators::FcParam* param) {
    return ComputeFcCommon<dtype>(param);
}

template <>
int32_t FcKernel<DeviceType::COMMON, DataType::FP8E4M3>::compute() {
    AutoTimer timer("Common::FC::FP8E4M3");
    return ComputeCommonFp8Fc<DataType::FP8E4M3>(static_cast<FcParam*>(param_));
}

template <>
int32_t FcKernel<DeviceType::COMMON, DataType::FP8E5M2>::compute() {
    AutoTimer timer("Common::FC::FP8E5M2");
    return ComputeCommonFp8Fc<DataType::FP8E5M2>(static_cast<FcParam*>(param_));
}

template <>
int32_t FcKernel<DeviceType::COMMON, DataType::INT8>::compute() {
    AutoTimer timer("Common::FC::INT8");
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr ||
        param->input->dims().size() != 2 || param->w->dims().size() != 2 ||
        param->input->dims()[1] != param->w->dims()[0]) {
        return -1;
    }
    const int64_t rows = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t channels = param->w->dims()[1];
    if (rows <= 0 || k <= 0 || channels <= 0 || param->out->dims().data() != std::vector<int64_t>{rows, channels}) {
        return -1;
    }

    int8_detail::QuantizationView input_quantization;
    int8_detail::QuantizationView weight_quantization;
    int8_detail::QuantizationView output_quantization;
    if (!int8_detail::BuildInputQuantizationView(param->input, &input_quantization) ||
        !int8_detail::BuildWeightQuantizationView(param->w, 1, channels, &weight_quantization) ||
        !int8_detail::BuildOutputQuantizationView(param->out, &output_quantization) ||
        !int8_detail::ValidateLinearBias(param->bias, rows, channels)) {
        return -1;
    }

    const int8_t* input = param->input->data<int8_t>();
    const int8_t* weight = param->w->data<int8_t>();
    int8_t* output = param->out->mutable_data<int8_t>();
    const int32_t input_zero_point = input_quantization.zero_point;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            int64_t accumulator = 0;
            const int32_t weight_zero_point = weight_quantization.zero_point_for(static_cast<size_t>(channel));
            for (int64_t index = 0; index < k; ++index) {
                accumulator += static_cast<int64_t>(static_cast<int32_t>(input[row * k + index]) - input_zero_point) *
                                (static_cast<int32_t>(weight[index * channels + channel]) - weight_zero_point);
                if (!int8_detail::FitsInt32(accumulator)) {
                    return -1;
                }
            }
            if (!int8_detail::AddInt32Bias(&accumulator,
                                           int8_detail::ReadLinearBias(param->bias, row, channel, channels))) {
                return -1;
            }
            const double scale = static_cast<double>(input_quantization.scale) *
                                 weight_quantization.scale_for(static_cast<size_t>(channel));
            if (!int8_detail::QuantizeAccumulator(accumulator, scale, output_quantization,
                                                  &output[row * channels + channel])) {
                return -1;
            }
        }
    }
    return 0;
}
typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP32> FcCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, FC, FcCommonFP32Kernel);

typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP16> FcCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, FC, FcCommonFP16Kernel);

typedef feather::kernel::FcKernel<DeviceType::COMMON, DataType::INT8> FcCommonINT8Kernel;
REGISTER_KERNEL(COMMON, INT8, FC, FcCommonINT8Kernel);

void EnsureCommonFcKernelsRegistered() { (void)g_fc_kernels_registered; }

void EnsureFcKernelsRegistered() {
    EnsureCommonFcKernelsRegistered();
    EnsureX86FcKernelsRegistered();
}
}  // namespace kernel
}  // namespace feather
