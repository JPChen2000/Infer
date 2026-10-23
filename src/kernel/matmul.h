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

#ifndef FEATHER_KERNEL_MATMUL_H
#define FEATHER_KERNEL_MATMUL_H

#include "core/kernel.h"
#include "src/operator/params.h"
#include "src/kernel/x86/linear.h"
#include "src/kernel/x86/linear_fp8.h"

namespace feather {
namespace kernel {

void EnsureCommonMatMulKernelsRegistered();
void EnsureX86MatMulKernelsRegistered();
void EnsureMatMulKernelsRegistered();

inline Status MatMulOutputSupport(const operators::MatMulParam* param, DataType dtype) {
    if (!param || !param->out)
        return Status::Error(StatusCode::kInvalidArgument, "MatMul requires a bound output");
    if (param->out->data_type() != dtype)
        return Status::Error(StatusCode::kUnsupported, "MatMul output dtype is incompatible with this candidate");
    return Status::Ok();
}

template <DeviceType dev, DataType dtype>
class MatMulKernel : public KernelBase {
   public:
    Status Supports() const override { return MatMulOutputSupport(ParamAs<operators::MatMulParam>(), dtype); }
    int32_t compute() override;
};

template <>
class MatMulKernel<DeviceType::X86, DataType::BF16> : public KernelBase {
   public:
    Status Supports() const override { return MatMulOutputSupport(ParamAs<operators::MatMulParam>(), DataType::BF16); }
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedBf16Rhs packed_rhs_;
    x86::Bf16LinearWorkspace workspace_;
};

template <>
class MatMulKernel<DeviceType::X86, DataType::FP8E4M3> : public KernelBase {
   public:
    Status Supports() const override { return MatMulOutputSupport(ParamAs<operators::MatMulParam>(), DataType::FP8E4M3); }
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

template <>
class MatMulKernel<DeviceType::X86, DataType::FP8E5M2> : public KernelBase {
   public:
    Status Supports() const override { return MatMulOutputSupport(ParamAs<operators::MatMulParam>(), DataType::FP8E5M2); }
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_MATMUL_H
