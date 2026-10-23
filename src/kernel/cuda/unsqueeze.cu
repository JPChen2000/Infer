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

#include "src/kernel/unsqueeze.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_cuda_unsqueeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::BF16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::INT64, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::CUDA, DataType::INT64>>();
    });
    return true;
}();

template <DataType dtype>
int32_t RunUnsqueeze(feather::operators::AxesParam* param, const char* timer_name) {
    AutoTimer timer(timer_name);
    using T = cuda_detail::StorageT<dtype>;
    if (param == nullptr || param->input == nullptr || param->out == nullptr || param->input->data_type() != dtype ||
        param->input->numel() != param->out->numel()) {
        return -1;
    }
    return cuda_detail::RunDeviceCopy<T>(param->input.get(), param->out.get());
}

}  // namespace

template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    return RunUnsqueeze<DataType::FP32>(ParamAs<feather::operators::AxesParam>(),
                                        "CUDA::Unsqueeze::FP32");
}

template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::FP16>::compute() {
    return RunUnsqueeze<DataType::FP16>(ParamAs<feather::operators::AxesParam>(),
                                        "CUDA::Unsqueeze::FP16");
}

template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::BF16>::compute() {
    return RunUnsqueeze<DataType::BF16>(ParamAs<feather::operators::AxesParam>(),
                                        "CUDA::Unsqueeze::BF16");
}

template <>
int32_t UnsqueezeKernel<DeviceType::CUDA, DataType::INT64>::compute() {
    return RunUnsqueeze<DataType::INT64>(ParamAs<feather::operators::AxesParam>(), "CUDA::Unsqueeze::INT64");
}

void EnsureCudaUnsqueezeKernelsRegistered() { (void)g_cuda_unsqueeze_kernels_registered; }

}  // namespace kernel
}  // namespace feather
