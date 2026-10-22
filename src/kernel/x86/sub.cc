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

#include "src/kernel/sub.h"

#include "src/kernel/common/elementwise_broadcast.h"
#include "src/kernel/x86/elementwise.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Sub::FP32");
    auto* param = ParamAs<feather::operators::BinaryParam>();
    if (param == nullptr || param->lhs == nullptr || param->rhs == nullptr || param->out == nullptr) {
        return -1;
    }
    if (x86::elementwise_detail::TryComputeScalarBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kSub>(param) ||
        x86::elementwise_detail::TryComputeLastDimensionBroadcastFp32<
            x86::elementwise_detail::BinaryOperation::kSub>(param)) {
        return 0;
    }
    return common_detail::RunBinary<DataType::FP32>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                                    [](float lhs, float rhs) { return lhs - rhs; });
}

template <DataType dtype>
int32_t ComputeX86Fp8Sub(feather::operators::BinaryParam* param) {
    if (param == nullptr) return -1;
    return common_detail::RunBinary<dtype>(param->out.get(), param->lhs.get(), param->rhs.get(),
                                           [](float lhs, float rhs) { return lhs - rhs; });
}

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP8E4M3>::compute() {
    AutoTimer timer("X86::Sub::FP8E4M3");
    return ComputeX86Fp8Sub<DataType::FP8E4M3>(ParamAs<feather::operators::BinaryParam>());
}

template <>
int32_t SubKernel<DeviceType::X86, DataType::FP8E5M2>::compute() {
    AutoTimer timer("X86::Sub::FP8E5M2");
    return ComputeX86Fp8Sub<DataType::FP8E5M2>(ParamAs<feather::operators::BinaryParam>());
}



}  // namespace kernel
}  // namespace feather
