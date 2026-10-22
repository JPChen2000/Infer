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

#include "src/kernel/exp.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_exp_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Exp", []() { return std::make_unique<ExpKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunExp(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::exp(value); });
}

}  // namespace

template <> int32_t ExpKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Exp::FP32");
    return RunExp<DataType::FP32>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t ExpKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Exp::FP16");
    return RunExp<DataType::FP16>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t ExpKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Exp::BF16");
    return RunExp<DataType::BF16>(ParamAs<feather::operators::UnaryParam>());
}

void EnsureExpKernelsRegistered() {
    (void)g_exp_kernels_registered;
    EnsureX86ExpKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
