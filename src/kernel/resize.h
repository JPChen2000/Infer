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

#ifndef FEATHER_KERNEL_RESIZE_H
#define FEATHER_KERNEL_RESIZE_H

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

enum class CudaResizeBackend {
    kUnknown,
    kGeneric,
    kDirect4D,
};

namespace detail {
inline thread_local CudaResizeBackend g_last_cuda_resize_backend = CudaResizeBackend::kUnknown;
}

inline CudaResizeBackend LastCudaResizeBackend() { return detail::g_last_cuda_resize_backend; }
inline void ResetLastCudaResizeBackend() { detail::g_last_cuda_resize_backend = CudaResizeBackend::kUnknown; }
inline void SetLastCudaResizeBackend(CudaResizeBackend backend) { detail::g_last_cuda_resize_backend = backend; }

void EnsureCommonResizeKernelsRegistered();
void EnsureX86ResizeKernelsRegistered();
void EnsureResizeKernelsRegistered();

template <DeviceType dev, DataType dtype>
class ResizeKernel : public KernelBase {
   public:
    int32_t compute() override;
};

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_RESIZE_H
