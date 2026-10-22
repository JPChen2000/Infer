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

#include "src/kernel/pow.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/elementwise.cuh"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_pow_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Pow",
                              []() { return std::make_unique<PowKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Pow",
                              []() { return std::make_unique<PowKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return cuda_detail::RunPow<DataType::FP32>(ParamAs<feather::operators::PowParam>(),
                                              "CUDA::Pow::FP32");
}

template <>
int32_t PowKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return cuda_detail::RunPow<DataType::FP16>(ParamAs<feather::operators::PowParam>(),
                                              "CUDA::Pow::FP16");
}

void EnsureCudaPowKernelsRegistered() { (void)g_cuda_pow_kernels_registered; }

}  // namespace kernel
}  // namespace feather
