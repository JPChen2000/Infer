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

#include "src/kernel/resize.h"

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_resize_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Resize",
                                                []() { return std::make_unique<ResizeKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Resize",
                                                []() { return std::make_unique<ResizeKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Resize",
                                                []() { return std::make_unique<ResizeKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeResizeKernel(feather::operators::ResizeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    param->out->set_data_type(dtype);
    return ComputeResizeHost(param,
        [&](int64_t index) { return TensorIO<dtype>::Read(param->input.get(), index); },
        [&](int64_t index, float value) { TensorIO<dtype>::Write(param->out.get(), index, value); });
}

template <>
int32_t ResizeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Resize::FP32");
    return ComputeResizeKernel<DataType::FP32>(ParamAs<feather::operators::ResizeParam>());
}

template <>
int32_t ResizeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Resize::FP16");
    return ComputeResizeKernel<DataType::FP16>(ParamAs<feather::operators::ResizeParam>());
}

template <>
int32_t ResizeKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Resize::BF16");
    return ComputeResizeKernel<DataType::BF16>(ParamAs<feather::operators::ResizeParam>());
}

typedef feather::kernel::ResizeKernel<DeviceType::COMMON, DataType::FP32> ResizeCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Resize, ResizeCommonFP32Kernel);

typedef feather::kernel::ResizeKernel<DeviceType::COMMON, DataType::FP16> ResizeCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Resize, ResizeCommonFP16Kernel);

void EnsureCommonResizeKernelsRegistered() { (void)g_resize_kernels_registered; }

void EnsureResizeKernelsRegistered() {
    EnsureCommonResizeKernelsRegistered();
    EnsureX86ResizeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
