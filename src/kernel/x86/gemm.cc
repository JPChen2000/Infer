#include "src/kernel/gemm.h"

#include "src/operator/params.h"
#include "src/kernel/common/kernel_io.h"
#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp16.h"
#include "src/kernel/x86/linear_fp32.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

constexpr int64_t kBf16LogitsPackMinimumOutput = 32768;

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

template <DataType dtype>
int32_t ComputeGeneralGemm(feather::operators::GemmParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 || param->trans_a ||
        param->a->data_type() != dtype || param->b->data_type() != dtype || param->out->data_type() != dtype) {
        return -1;
    }
    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    if (k <= 0) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    const int64_t b_k = param->trans_b ? param->b->dims()[1] : param->b->dims()[0];
    const int64_t n = param->trans_b ? param->b->dims()[0] : param->b->dims()[1];
    if (k != b_k || param->out->numel() != m * n) {
        return -1;
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                const int64_t a_offset = i * k + t;
                const int64_t b_offset = param->trans_b ? j * k + t : t * n + j;
                sum += TensorIO<dtype>::Read(param->a.get(), a_offset) * TensorIO<dtype>::Read(param->b.get(), b_offset);
            }
            sum *= param->alpha;
            if (param->bias != nullptr && param->bias->IsInitialized()) {
                const int64_t bias_offset = IsVectorBias(param->bias.get(), n) ? j : i * n + j;
                sum += param->beta * TensorIO<dtype>::Read(param->bias.get(), bias_offset);
            }
            TensorIO<dtype>::Write(param->out.get(), i * n + j, sum);
        }
    }
    return 0;
}

}  // namespace

int32_t GemmKernel<DeviceType::X86, DataType::BF16>::Prepare() {
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || param->a->data_type() != DataType::BF16 ||
        param->b->data_type() != DataType::BF16 || !param->trans_b ||
        param->trans_a || param->a->dims().size() < 2 || param->b->dims().size() != 2 ||
        !param->b->is_immutable()) {
        return 0;
    }
    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    const int64_t n = param->b->dims()[0];
    if (k <= 0 || param->a->numel() <= 0 || param->a->numel() % k != 0 ||
        n < kBf16LogitsPackMinimumOutput || param->b->dims()[1] != k) {
        return 0;
    }
    const int64_t m = param->a->numel() / k;
    if (m != 1) {
        return 0;
    }
    // Packing is an optimization. If allocation fails, compute() will use the
    // direct transposed-RHS path instead of making graph finalization fail.
    (void)packed_transposed_rhs_.Pack(param->b->data<uint16_t>(), k, n, param->b->mutation_version());
    return 0;
}

template <>
int32_t GemmKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Gemm::FP32");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || !param->out->IsInitialized()) {
        return -1;
    }

    if (param->trans_a || param->trans_b || param->alpha != 1.0f || param->beta != 1.0f) {
        return ComputeGeneralGemm<DataType::FP32>(param);
    }

    const float* lhs = param->a->data<float>();
    const float* rhs = param->b->data<float>();
    const float* bias = param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<float>() : nullptr;
    float* out = param->out->mutable_data<float>();

    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    if (k <= 0) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    const int64_t n = param->b->dims()[1];

    const x86::LinearBiasType bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
        : (IsVectorBias(param->bias.get(), n) ? x86::LinearBiasType::kVector : x86::LinearBiasType::kMatrix);
    return x86::ComputeLinearRowMajorX86Fp32(lhs, rhs, bias, m, k, n, bias_type, out);
}

template <>
int32_t GemmKernel<DeviceType::X86, DataType::FP16>::compute() {
    AutoTimer timer("X86::Gemm::FP16");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }

    if (param->trans_a || param->trans_b || param->alpha != 1.0f || param->beta != 1.0f) {
        return ComputeGeneralGemm<DataType::FP16>(param);
    }

    const uint16_t* lhs = param->a->data<uint16_t>();
    const uint16_t* rhs = param->b->data<uint16_t>();
    const uint16_t* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<uint16_t>() : nullptr;
    uint16_t* out = param->out->mutable_data<uint16_t>();

    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    if (k <= 0) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    const int64_t n = param->b->dims()[1];

    const x86::LinearBiasType bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
        : (IsVectorBias(param->bias.get(), n) ? x86::LinearBiasType::kVector : x86::LinearBiasType::kMatrix);
    return x86::ComputeLinearRowMajorX86Fp16(lhs, rhs, bias, m, k, n, bias_type, out);
}

int32_t GemmKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Gemm::BF16");
    auto* param = static_cast<feather::operators::GemmParam*>(param_);
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr) {
        return -1;
    }
    if (param->trans_a || param->a->data_type() != DataType::BF16 || param->b->data_type() != DataType::BF16 ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2) {
        return -1;
    }

    const auto& a_dims = param->a->dims().data();
    const auto& b_dims = param->b->dims().data();
    const int64_t k = a_dims.back();
    const int64_t b_k = param->trans_b ? b_dims[1] : b_dims[0];
    const int64_t n = param->trans_b ? b_dims[0] : b_dims[1];
    if (k <= 0 || param->a->numel() <= 0 || param->a->numel() % k != 0 || n <= 0 || k != b_k) {
        return -1;
    }
    const int64_t m = param->a->numel() / k;
    if (m <= 0 || param->out->numel() != m * n) {
        return -1;
    }

    param->out->set_data_type(DataType::BF16);

    const uint16_t* bias =
        param->bias != nullptr && param->bias->IsInitialized() ? param->bias->data<uint16_t>() : nullptr;
    const x86::LinearBiasType bias_type =
        bias == nullptr ? x86::LinearBiasType::kNone
        : (IsVectorBias(param->bias.get(), n) ? x86::LinearBiasType::kVector : x86::LinearBiasType::kMatrix);
    auto* out = static_cast<uint16_t*>(param->out->raw_data());
    if (param->trans_b) {
        const x86::PackedBf16TransposedRhs* packed_rhs = nullptr;
        const bool packed_ok = m == 1 && n >= kBf16LogitsPackMinimumOutput && param->b->is_immutable() &&
                               packed_transposed_rhs_.Pack(param->b->data<uint16_t>(), k, n,
                                                           param->b->mutation_version());
        if (packed_ok) {
            packed_rhs = &packed_transposed_rhs_;
        }
        if (packed_rhs != nullptr &&
            packed_rhs->Matches(param->b->data<uint16_t>(), k, n, param->b->mutation_version())) {
            return x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhs(
                param->a->data<uint16_t>(), param->b->data<uint16_t>(), *packed_rhs, bias, m, k, n, bias_type,
                param->alpha, param->beta, out, &workspace_, param->b->mutation_version());
        }
        return x86::ComputeLinearRowMajorX86Bf16TransposedRhs(param->a->data<uint16_t>(), param->b->data<uint16_t>(),
                                                              bias, m, k, n, bias_type, param->alpha, param->beta, out,
                                                              &workspace_);
    }
    if (param->alpha != 1.0f || param->beta != 1.0f) {
        return ComputeGeneralGemm<DataType::BF16>(param);
    }
    return x86::ComputeLinearRowMajorX86Bf16(param->a->data<uint16_t>(), param->b->data<uint16_t>(), bias, m, k, n,
                                             bias_type, out, &workspace_);
}

void EnsureX86GemmKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
