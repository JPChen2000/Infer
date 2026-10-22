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

#include "src/kernel/tanh.h"

#include <memory>

#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_tanh_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Tanh", []() {
        return std::make_unique<TanhKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t TanhKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP32, 5>(
        ParamAs<feather::operators::UnaryParam>(), "CUDA::Tanh::FP32");
}

template <>
int32_t TanhKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunUnaryElementwise<DataType::FP16, 5>(
        ParamAs<feather::operators::UnaryParam>(), "CUDA::Tanh::FP16");
}

void EnsureCudaTanhKernelsRegistered() { (void)g_cuda_tanh_kernels_registered; }

}  // namespace kernel
}  // namespace feather
