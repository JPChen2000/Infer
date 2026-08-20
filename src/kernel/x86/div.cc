#include "src/kernel/div.h"

#include <immintrin.h>

#include <memory>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool TryComputeLastDimensionScalarBroadcastFp32(feather::operators::BinaryParam* param) {
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr ||
        !param->lhs->IsInitialized() || !param->rhs->IsInitialized() || !param->out->IsInitialized() ||
        param->lhs->data_type() != DataType::FP32 || param->rhs->data_type() != DataType::FP32) {
        return false;
    }

    const auto& lhs_dims = param->lhs->dims().data();
    const auto& rhs_dims = param->rhs->dims().data();
    const auto& out_dims = param->out->dims().data();
    if (out_dims.empty() || lhs_dims != out_dims || rhs_dims.size() != out_dims.size() || rhs_dims.back() != 1 ||
        out_dims.back() <= 0 || param->out->numel() % out_dims.back() != 0) {
        return false;
    }
    for (size_t axis = 0; axis + 1 < out_dims.size(); ++axis) {
        if (rhs_dims[axis] != out_dims[axis]) {
            return false;
        }
    }

    const int64_t inner = out_dims.back();
    const int64_t rows = param->out->numel() / inner;
    if (param->rhs->numel() != rows || param->lhs->numel() != param->out->numel()) {
        return false;
    }

    const float* lhs = param->lhs->data<float>();
    const float* rhs = param->rhs->data<float>();
    float* out = static_cast<float*>(param->out->raw_data());
    for (int64_t row = 0; row < rows; ++row) {
        const __m256 divisor = _mm256_set1_ps(rhs[row]);
        const float* lhs_row = lhs + row * inner;
        float* out_row = out + row * inner;
        int64_t col = 0;
        for (; col + 8 <= inner; col += 8) {
            _mm256_storeu_ps(out_row + col, _mm256_div_ps(_mm256_loadu_ps(lhs_row + col), divisor));
        }
        for (; col < inner; ++col) {
            out_row[col] = lhs_row[col] / rhs[row];
        }
    }
    param->out->set_data_type(DataType::FP32);
    return true;
}

}  // namespace

template <>
int32_t DivKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Div::FP32");
    auto* param = static_cast<feather::operators::BinaryParam*>(param_);
    if (TryComputeLastDimensionScalarBroadcastFp32(param)) {
        return 0;
    }
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs / rhs; });
}

void EnsureX86DivKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Div",
            []() { return std::make_unique<DivKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
