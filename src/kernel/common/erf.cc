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

#include "src/kernel/erf.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_erf_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Erf", []() {
        return std::make_unique<ErfKernel<DeviceType::COMMON, DataType::BF16>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t ErfKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Erf::FP32");
    auto* param = ParamAs<feather::operators::UnaryParam>();
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP32>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::erf(value); });
}

template <>
int32_t ErfKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Erf::FP16");
    auto* param = ParamAs<feather::operators::UnaryParam>();
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::FP16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::erf(value); });
}

template <>
int32_t ErfKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Erf::BF16");
    auto* param = ParamAs<feather::operators::UnaryParam>();
    if (param == nullptr) {
        return -1;
    }
    return common_detail::RunUnary<DataType::BF16>(param->out.get(), param->input.get(),
                                                   [](float value) { return std::erf(value); });
}

void EnsureErfKernelsRegistered() {
    (void)g_erf_kernels_registered;
    EnsureX86ErfKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
