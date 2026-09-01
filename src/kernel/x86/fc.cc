#include "src/kernel/fc.h"

#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp8.h"
#include "src/kernel/x86/linear_fp32.h"
#include "src/kernel/fp8_host.h"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

void EnsureX86Int8KernelsRegistered();

namespace {

constexpr int64_t kFp8PackedRhsMinimumMacs = 1 << 18;

bool g_fc_x86_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP32, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP16, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP8E4M3, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP8E5M2, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
    return true;
}();

template <DataType dtype>
bool ShouldUsePackedFp8Fc(const feather::operators::FcParam* param, int64_t m, int64_t k, int64_t n) {
    return param != nullptr && param->w != nullptr && param->w->is_immutable() && m == 1 && k > 0 && n >= 64 &&
           k >= kFp8PackedRhsMinimumMacs / n;
}

template <DataType dtype>
int32_t ComputeFp8Fc(feather::operators::FcParam* param, x86::PackedFp8Rhs* packed_rhs,
                     x86::Fp8LinearWorkspace* workspace) {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    if (!fp8_host::ValidateFc<dtype>(param, &m, &k, &n)) return -1;
    const auto bias_type =
        param->bias == nullptr
            ? x86::LinearBiasType::kNone
            : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector : x86::LinearBiasType::kMatrix);
    const auto* lhs = static_cast<const uint8_t*>(param->input->raw_data());
    const auto* rhs = static_cast<const uint8_t*>(param->w->raw_data());
    auto* out = static_cast<uint8_t*>(param->out->raw_data());
    const auto* bias = param->bias == nullptr ? nullptr : static_cast<const uint8_t*>(param->bias->raw_data());
    const float bias_scale = param->bias == nullptr ? 1.0f : param->bias->quantization_scale();
    if (ShouldUsePackedFp8Fc<dtype>(param, m, k, n) && packed_rhs != nullptr &&
        packed_rhs->Matches(dtype, rhs, k, n, param->w->mutation_version())) {
        return x86::ComputeLinearRowMajorX86Fp8PackedRhs(
            dtype, lhs, param->input->quantization_scale(), rhs, param->w->quantization_scale(), *packed_rhs, bias,
            bias_scale, m, k, n, bias_type, out, param->out->quantization_scale(), 1.0f, 1.0f,
            param->w->mutation_version(), workspace);
    }
    return x86::ComputeLinearRowMajorX86Fp8(
        dtype, lhs, param->input->quantization_scale(), rhs, param->w->quantization_scale(), bias, bias_scale, m, k,
        n, bias_type, out, param->out->quantization_scale());
}

template <DataType dtype>
int32_t PrepareFp8Fc(feather::operators::FcParam* param, x86::PackedFp8Rhs* packed_rhs) {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    if (param == nullptr || !fp8_host::ValidateFc<dtype>(param, &m, &k, &n) ||
        !ShouldUsePackedFp8Fc<dtype>(param, m, k, n) || packed_rhs == nullptr) {
        return 0;
    }
    (void)packed_rhs->Pack(dtype, static_cast<const uint8_t*>(param->w->raw_data()), k, n,
                           param->w->mutation_version());
    return 0;
}

}  // namespace

template <>
int32_t FcKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::FC::FP32");
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    const float* input = param->input->data<float>();
    const float* weight = param->w->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    const auto bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);

    return x86::ComputeLinearRowMajorX86Fp32(input, weight, bias, m, k, n, bias_type, param->out->mutable_data<float>());
}

template <>
int32_t FcKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::FC::FP16");
    auto* param = static_cast<feather::operators::FcParam*>(param_);
    if (param == nullptr || param->input == nullptr || param->w == nullptr || param->out == nullptr) {
        return -1;
    }

    const uint16_t* input = param->input->data<uint16_t>();
    const uint16_t* weight = param->w->data<uint16_t>();
    const uint16_t* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<uint16_t>() : nullptr;
    const int64_t m = param->input->dims()[0];
    const int64_t k = param->input->dims()[1];
    const int64_t n = param->w->dims()[1];
    const auto bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
                        : (param->bias->dims().size() == 1 ? x86::LinearBiasType::kVector
                                                           : x86::LinearBiasType::kMatrix);

    return x86::ComputeLinearRowMajorX86Fp16(input, weight, bias, m, k, n, bias_type,
                                             param->out->mutable_data<uint16_t>());
}

int32_t FcKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::FC::FP8E4M3");
    return ComputeFp8Fc<DataType::FP8E4M3>(static_cast<feather::operators::FcParam*>(param_), &packed_rhs_, &workspace_);
}

int32_t FcKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::FC::FP8E5M2");
    return ComputeFp8Fc<DataType::FP8E5M2>(static_cast<feather::operators::FcParam*>(param_), &packed_rhs_, &workspace_);
}

int32_t FcKernel<DeviceType::X86, DataType::FP8E4M3>::Prepare() {
    return PrepareFp8Fc<DataType::FP8E4M3>(static_cast<feather::operators::FcParam*>(param_), &packed_rhs_);
}

int32_t FcKernel<DeviceType::X86, DataType::FP8E5M2>::Prepare() {
    return PrepareFp8Fc<DataType::FP8E5M2>(static_cast<feather::operators::FcParam*>(param_), &packed_rhs_);
}

void EnsureX86FcKernelsRegistered() {
    (void)g_fc_x86_kernels_registered;
    EnsureX86Int8KernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
