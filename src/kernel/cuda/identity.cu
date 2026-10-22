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

#include "src/kernel/identity.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_identity_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Identity",
                              []() { return std::make_unique<IdentityKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Identity",
                              []() { return std::make_unique<IdentityKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Identity",
                              []() { return std::make_unique<IdentityKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int RunIdentity(feather::operators::UnaryParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    // Identity keeps independent host Tensor storage while sharing the CUDA
    // allocation with its input. This removes a decode-time device-to-device copy.
    return cuda_detail::RunDeviceAlias<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunIdentity<DataType::FP32>(ParamAs<feather::operators::UnaryParam>(),
                                      "CUDA::Identity::FP32");
}

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunIdentity<DataType::FP16>(ParamAs<feather::operators::UnaryParam>(),
                                      "CUDA::Identity::FP16");
}

template <>
int32_t IdentityKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunIdentity<DataType::BF16>(ParamAs<feather::operators::UnaryParam>(),
                                       "CUDA::Identity::BF16");
}

void EnsureCudaIdentityKernelsRegistered() { (void)g_cuda_identity_kernels_registered; }

}  // namespace kernel
}  // namespace feather
