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
#include <cstdint>

#include "util/bf16.h"
#include "util/fp8.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool IsNaN(uint16_t bits) { return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0; }

bool IsValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

template <DataType dtype>
bool Validate(const operators::QwenGemmArgmaxParam* param, int64_t* k, int64_t* n) {
    if (param == nullptr || param->a == nullptr || param->b == nullptr || param->out == nullptr ||
        !param->a->IsInitialized() || !param->b->IsInitialized() || !param->out->IsInitialized() ||
        param->a->data_type() != dtype || param->b->data_type() != dtype ||
        param->a->dims().size() < 2 || param->b->dims().size() != 2 || k == nullptr || n == nullptr) {
        return false;
    }
    if (!IsValidScale(param->output_scale)) {
        return false;
    }
    if constexpr (dtype == DataType::FP8E4M3 || dtype == DataType::FP8E5M2) {
        if (!HasCompatiblePerTensorQuantization(param->a->quantization()) ||
            !HasCompatiblePerTensorQuantization(param->b->quantization()) ||
            !IsValidScale(param->a->quantization_scale()) || !IsValidScale(param->b->quantization_scale())) {
            return false;
        }
    }
    *k = param->a->dims()[param->a->dims().size() - 1];
    *n = param->b->dims()[0];
    return *k > 0 && *n > 0 && param->a->numel() == *k && param->b->dims()[1] == *k && param->out->numel() == 1;
}

template <DataType dtype>
int32_t ComputeFp8(operators::QwenGemmArgmaxParam* param, x86::PackedFp8TransposedRhs* packed_rhs,
                   x86::Fp8LinearWorkspace* workspace) {
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate<dtype>(param, &k, &n)) {
        return -1;
    }
    int64_t token = 0;
    if (x86::ComputeLinearRowMajorX86Fp8TransposedRhsArgmax(
            dtype, static_cast<const uint8_t*>(param->a->raw_data()), param->a->quantization_scale(),
            static_cast<const uint8_t*>(param->b->raw_data()), param->b->quantization_scale(), packed_rhs, k, n,
            param->output_scale, &token, workspace, param->b->mutation_version()) != 0) {
        return -1;
    }
    param->out->set_data_type(DataType::INT64);
    param->out->mutable_data<int64_t>()[0] = token;
    return 0;
}

template <DataType dtype>
int32_t PrepareFp8(operators::QwenGemmArgmaxParam* param, x86::PackedFp8TransposedRhs* packed_rhs) {
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate<dtype>(param, &k, &n) || !param->b->is_immutable()) {
        return 0;
    }
    (void)packed_rhs->Pack(dtype, static_cast<const uint8_t*>(param->b->raw_data()), k, n,
                           param->b->mutation_version());
    return 0;
}

}  // namespace

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>::Prepare() {
    auto* param = ParamAs<operators::QwenGemmArgmaxParam>();
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate<DataType::BF16>(param, &k, &n) || !param->b->is_immutable()) {
        return 0;
    }
    (void)packed_rhs_.Pack(static_cast<const uint16_t*>(param->b->raw_data()), k, n,
                           param->b->mutation_version());
    return 0;
}

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::QwenGemmArgmax::BF16");
    auto* param = ParamAs<operators::QwenGemmArgmaxParam>();
    int64_t k = 0;
    int64_t n = 0;
    if (!Validate<DataType::BF16>(param, &k, &n)) {
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



int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E4M3>::Prepare() {
    return PrepareFp8<DataType::FP8E4M3>(ParamAs<operators::QwenGemmArgmaxParam>(), &packed_rhs_);
}

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::QwenGemmArgmax::FP8E4M3");
    return ComputeFp8<DataType::FP8E4M3>(ParamAs<operators::QwenGemmArgmaxParam>(), &packed_rhs_,
                                         &workspace_);
}

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E5M2>::Prepare() {
    return PrepareFp8<DataType::FP8E5M2>(ParamAs<operators::QwenGemmArgmaxParam>(), &packed_rhs_);
}

int32_t QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::QwenGemmArgmax::FP8E5M2");
    return ComputeFp8<DataType::FP8E5M2>(ParamAs<operators::QwenGemmArgmaxParam>(), &packed_rhs_,
                                         &workspace_);
}

}  // namespace kernel
}  // namespace feather
