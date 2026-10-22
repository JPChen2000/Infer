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

#ifndef FEATHER_KERNEL_X86_LINEAR_FP16_H
#define FEATHER_KERNEL_X86_LINEAR_FP16_H

#include <cstdint>

#include "src/kernel/x86/linear_fp32.h"

namespace feather {
namespace kernel {
namespace x86 {

int32_t ComputeLinearRowMajorX86Fp16(const uint16_t* lhs, const uint16_t* rhs, const uint16_t* bias, int64_t m,
                                     int64_t k, int64_t n, LinearBiasType bias_type, uint16_t* out);

}  // namespace x86
}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_X86_LINEAR_FP16_H
