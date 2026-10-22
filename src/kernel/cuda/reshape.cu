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

#include "src/kernel/reshape.h"

#include <memory>

#include "core/kernel.h"
#include "src/kernel/cuda/kernel_io.cuh"
#include "src/operator/params.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_reshape_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Reshape",
                              []() { return std::make_unique<ReshapeKernel<DeviceType::CUDA, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int RunReshape(feather::operators::ReshapeParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }
    // Reshape is a contiguous view: host metadata remains per Tensor, but the
    // CUDA allocation is shared with the input instead of copied.
    return cuda_detail::RunDeviceAlias<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunReshape<DataType::FP32>(ParamAs<feather::operators::ReshapeParam>(),
                                     "CUDA::Reshape::FP32");
}

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunReshape<DataType::FP16>(ParamAs<feather::operators::ReshapeParam>(),
                                     "CUDA::Reshape::FP16");
}

template <>
int32_t ReshapeKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunReshape<DataType::BF16>(ParamAs<feather::operators::ReshapeParam>(),
                                      "CUDA::Reshape::BF16");
}

void EnsureCudaReshapeKernelsRegistered() { (void)g_cuda_reshape_kernels_registered; }

}  // namespace kernel
}  // namespace feather
