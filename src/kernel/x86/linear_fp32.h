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

#ifndef FEATHER_KERNEL_X86_LINEAR_FP32_H
#define FEATHER_KERNEL_X86_LINEAR_FP32_H

#include <cstddef>
#include <cstdint>

namespace feather {
namespace kernel {
namespace x86 {

enum class LinearBiasType {
    kNone,
    kVector,
    kMatrix,
};

size_t Fp32LinearWorkerCount(int64_t m, int64_t k, int64_t n);

// Large row-major matrices use a small row block so the RHS vector is reused
// across several output rows.  Small and single-row workloads keep the
// existing path to avoid paying the blocking overhead.
bool Fp32LinearUsesBlockedKernel(int64_t m, int64_t k, int64_t n);

int32_t ComputeLinearRowMajorX86Fp32(const float* lhs, const float* rhs, const float* bias, int64_t m, int64_t k,
                                     int64_t n, LinearBiasType bias_type, float* out);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_FP32_H
