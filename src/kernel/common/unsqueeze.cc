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

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_unsqueeze_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "Unsqueeze", []() {
        return std::make_unique<UnsqueezeKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Unsqueeze::FP32");
    return common_tensor_detail::CopyTensor(ParamAs<feather::operators::AxesParam>());
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Unsqueeze::FP16");
    return common_tensor_detail::CopyTensor(ParamAs<feather::operators::AxesParam>());
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Unsqueeze::BF16");
    return common_tensor_detail::CopyTensor(ParamAs<feather::operators::AxesParam>());
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Unsqueeze::INT64");
    return common_tensor_detail::CopyTensor(ParamAs<feather::operators::AxesParam>());
}

template <>
int32_t UnsqueezeKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::Unsqueeze::BOOL");
    return common_tensor_detail::CopyTensor(ParamAs<feather::operators::AxesParam>());
}

void EnsureCommonUnsqueezeKernelsRegistered() { (void)g_common_unsqueeze_kernels_registered; }

void EnsureUnsqueezeKernelsRegistered() { EnsureCommonUnsqueezeKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
