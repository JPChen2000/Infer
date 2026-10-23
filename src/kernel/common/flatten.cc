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

#include <cstring>

#include "util/types.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_flatten_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Flatten",
                                                []() { return std::make_unique<FlattenKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Flatten",
                                                []() { return std::make_unique<FlattenKernel<DeviceType::COMMON, DataType::FP16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeFlattenKernel(feather::operators::FlattenParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    std::memcpy(param->out->mutable_data(param->out->memory_size()), param->input->raw_data(), bytes);
    return 0;
}

template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Flatten::FP32");
    return ComputeFlattenKernel<DataType::FP32>(ParamAs<feather::operators::FlattenParam>());
}

template <>
int32_t FlattenKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Flatten::FP16");
    return ComputeFlattenKernel<DataType::FP16>(ParamAs<feather::operators::FlattenParam>());
}

typedef feather::kernel::FlattenKernel<DeviceType::COMMON, DataType::FP32> FlattenCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Flatten, FlattenCommonFP32Kernel);

typedef feather::kernel::FlattenKernel<DeviceType::COMMON, DataType::FP16> FlattenCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Flatten, FlattenCommonFP16Kernel);

void EnsureFlattenKernelsRegistered() {
    (void)g_flatten_kernels_registered;
    EnsureX86FlattenKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
