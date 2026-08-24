#include "src/kernel/qwen_gemm_argmax.h"

#include "util/bf16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool IsNaN(uint16_t bits) { return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0; }

bool Validate(const operators::QwenGemmArgmaxParam* param, int64_t* k, int64_t* n) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || !param->out->IsInitialized() ||
        param->a->data_type() != DataType::BF16 || param->b->data_type() != DataType::BF16 ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 || k == nullptr || n == nullptr) {
        return false;
    }
    *k = param->a->dims()[param->a->dims().size() - 1];
    *n = param->b->dims()[0];
    return *k > 0 && *n > 0 && param->a->numel() == *k && param->b->dims()[1] == *k && param->out->numel() == 1;
}

}  // namespace

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>::Prepare() {
    auto* param = static_cast<operators::QwenGemmArgmaxParam*>(param_);
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate(param, &k, &n) || !param->b->is_immutable()) {
        return 0;
    }
    (void)packed_rhs_.Pack(static_cast<const uint16_t*>(param->b->raw_data()), k, n,
                           param->b->mutation_version());
    return 0;
}

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::QwenGemmArgmax::BF16");
    auto* param = static_cast<operators::QwenGemmArgmaxParam*>(param_);
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate(param, &k, &n)) {
        return -1;
    }
    const auto* lhs = static_cast<const uint16_t*>(param->a->raw_data());
    const auto* rhs = static_cast<const uint16_t*>(param->b->raw_data());
    int64_t token = 0;
    if (packed_rhs_.Matches(rhs, k, n, param->b->mutation_version())) {
        if (x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmax(
                lhs, rhs, packed_rhs_, k, n, &token, &workspace_, param->b->mutation_version()) != 0) {
            return -1;
        }
    } else {
        float best_value = 0.0f;
        int64_t best = -1;
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int64_t row = 0; row < k; ++row) {
                sum += BFloat16ToFloat(lhs[row]) * BFloat16ToFloat(rhs[col * k + row]);
            }
            const uint16_t rounded = FloatToBFloat16(sum);
            if (!IsNaN(rounded)) {
                const float value = BFloat16ToFloat(rounded);
                if (best < 0 || value > best_value) {
                    best = col;
                    best_value = value;
                }
            }
        }
        token = best < 0 ? 0 : best;
    }
    param->out->set_data_type(DataType::INT64);
    param->out->mutable_data<int64_t>()[0] = token;
    return 0;
}

void EnsureX86QwenGemmArgmaxKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::BF16, "QwenGemmArgmax", []() {
            return std::make_unique<QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>>();
        });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
