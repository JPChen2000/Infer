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

#include "src/kernel/flatten.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_flatten_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Flatten",
                              []() { return std::make_unique<FlattenKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Flatten",
                              []() { return std::make_unique<FlattenKernel<DeviceType::CUDA, DataType::FP16>>(); });
    return true;
}();

template <DataType dtype>
int RunFlatten(feather::operators::FlattenParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunFlatten<DataType::FP32>(ParamAs<feather::operators::FlattenParam>(),
                                     "CUDA::Flatten::FP32");
}

template <>
int32_t FlattenKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunFlatten<DataType::FP16>(ParamAs<feather::operators::FlattenParam>(),
                                     "CUDA::Flatten::FP16");
}

void EnsureCudaFlattenKernelsRegistered() { (void)g_cuda_flatten_kernels_registered; }

}  // namespace kernel
}  // namespace feather
