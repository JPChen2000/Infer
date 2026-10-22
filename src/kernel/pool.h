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

#ifndef FEATHER_KERNEL_POOL_H
#define FEATHER_KERNEL_POOL_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaPoolBackend {
    kUnknown,
    kFallback,
    kCudnn,
};

namespace detail {
inline thread_local CudaPoolBackend g_last_cuda_pool_backend = CudaPoolBackend::kUnknown;
}

inline CudaPoolBackend LastCudaPoolBackend() { return detail::g_last_cuda_pool_backend; }
inline void ResetLastCudaPoolBackend() { detail::g_last_cuda_pool_backend = CudaPoolBackend::kUnknown; }
inline void SetLastCudaPoolBackend(CudaPoolBackend backend) { detail::g_last_cuda_pool_backend = backend; }

void EnsureCommonPoolKernelsRegistered();
void EnsureX86PoolKernelsRegistered();
void EnsurePoolKernelsRegistered();

template <DeviceType dev, DataType dtype>
class AvgPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

template <DeviceType dev, DataType dtype>
class MaxPoolKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_POOL_H
