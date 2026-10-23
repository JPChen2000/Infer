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

#include <cstring>

#include "src/kernel/common/kernel_io.h"
#include "util/timer.h"
#include "util/types.h"

namespace feather {
namespace kernel {

namespace {

bool g_identity_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "Identity",
                                                []() { return std::make_unique<IdentityKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

}  // namespace

template <DataType dtype>
int32_t ComputeIdentityKernel(feather::operators::UnaryParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(param->input->numel()) * DataTypeBytes(dtype);
    param->out->set_data_type(dtype);
    if (param->input->IsInitialized() && param->out->IsInitialized() &&
        param->input->raw_data() == param->out->raw_data()) {
        return 0;
    }
    std::memcpy(param->out->mutable_data(param->out->memory_size()), param->input->raw_data(), bytes);
    return 0;
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Identity::FP32");
    return ComputeIdentityKernel<DataType::FP32>(ParamAs<feather::operators::UnaryParam>());
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Identity::FP16");
    return ComputeIdentityKernel<DataType::FP16>(ParamAs<feather::operators::UnaryParam>());
}

template <>
int32_t IdentityKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Identity::BF16");
    return ComputeIdentityKernel<DataType::BF16>(ParamAs<feather::operators::UnaryParam>());
}

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::FP32> IdentityCommonFP32Kernel;
REGISTER_KERNEL(COMMON, FP32, Identity, IdentityCommonFP32Kernel);

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::FP16> IdentityCommonFP16Kernel;
REGISTER_KERNEL(COMMON, FP16, Identity, IdentityCommonFP16Kernel);

typedef feather::kernel::IdentityKernel<DeviceType::COMMON, DataType::BF16> IdentityCommonBF16Kernel;
REGISTER_KERNEL(COMMON, BF16, Identity, IdentityCommonBF16Kernel);

void EnsureIdentityKernelsRegistered() {
    (void)g_identity_kernels_registered;
    EnsureX86IdentityKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
