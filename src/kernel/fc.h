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

#ifndef FEATHER_KERNEL_FC_COMPUTE_H
#define FEATHER_KERNEL_FC_COMPUTE_H
#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/x86/linear_fp8.h"
#include "util/logger.h"
using feather::Tensor;
namespace feather {
namespace kernel {

void EnsureCommonFcKernelsRegistered();
void EnsureFcKernelsRegistered();
void EnsureX86FcKernelsRegistered();

template <DeviceType dev, DataType dtype>
class FcKernel : public KernelBase {
   public:
    int32_t compute();
};

template <>
class FcKernel<DeviceType::X86, DataType::FP8E4M3> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};

template <>
class FcKernel<DeviceType::X86, DataType::FP8E5M2> : public KernelBase {
   public:
    int32_t Prepare() override;
    int32_t compute() override;

   private:
    x86::PackedFp8Rhs packed_rhs_;
    x86::Fp8LinearWorkspace workspace_;
};
}  // namespace kernel
}  // namespace feather
// template class feather::kernel::FcCUDAKernel<DataType::FP32>;
#endif  // FEATHER_KERNEL_FC_COMPUTE_H
