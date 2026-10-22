// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/kernel/qwen_gemm_argmax.h"

#include <cmath>
#include <limits>

#include "util/bf16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool IsNaN(uint16_t bits) { return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0; }

int64_t SelectGreedy(const uint16_t* lhs, const uint16_t* rhs, int64_t k, int64_t n) {
    int64_t best = -1;
    float best_value = 0.0f;
    for (int64_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t row = 0; row < k; ++row) {
            sum += BFloat16ToFloat(lhs[row]) * BFloat16ToFloat(rhs[col * k + row]);
        }
        const uint16_t rounded = FloatToBFloat16(sum);
        if (IsNaN(rounded)) {
            continue;
        }
        const float value = BFloat16ToFloat(rounded);
        if (best < 0 || value > best_value) {
            best = col;
            best_value = value;
        }
    }
    return best < 0 ? 0 : best;
}

template <DeviceType device>
int32_t Compute(operators::QwenGemmArgmaxParam* param) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || !param->out->IsInitialized() ||
        param->a->data_type() != DataType::BF16 || param->b->data_type() != DataType::BF16 ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 ||
        param->a->dims()[param->a->dims().size() - 1] != param->b->dims()[1] ||
        param->a->numel() != param->a->dims()[param->a->dims().size() - 1] || param->b->dims()[0] <= 0 ||
        param->out->numel() != 1) {
        return -1;
    }
    const int64_t k = param->a->dims()[param->a->dims().size() - 1];
    const int64_t n = param->b->dims()[0];
    param->out->set_data_type(DataType::INT64);
    param->out->mutable_data<int64_t>()[0] =
        SelectGreedy(static_cast<const uint16_t*>(param->a->raw_data()), static_cast<const uint16_t*>(param->b->raw_data()), k, n);
    return 0;
}

bool g_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "QwenGemmArgmax", []() {
        return std::make_unique<QwenGemmArgmaxKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t QwenGemmArgmaxKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::QwenGemmArgmax::BF16");
    return Compute<DeviceType::COMMON>(ParamAs<operators::QwenGemmArgmaxParam>());
}

void EnsureCommonQwenGemmArgmaxKernelsRegistered() { (void)g_registered; }

void EnsureQwenGemmArgmaxKernelsRegistered() {
    EnsureCommonQwenGemmArgmaxKernelsRegistered();
    EnsureX86QwenGemmArgmaxKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenGemmArgmaxKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
