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

#include "src/kernel/sin.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_sin_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Sin", []() { return std::make_unique<SinKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunSin(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::sin(value); });
}

}  // namespace

template <> int32_t SinKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Sin::FP32");
    return RunSin<DataType::FP32>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t SinKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Sin::FP16");
    return RunSin<DataType::FP16>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t SinKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Sin::BF16");
    return RunSin<DataType::BF16>(ParamAs<feather::operators::UnaryParam>());
}

void EnsureSinKernelsRegistered() { (void)g_sin_kernels_registered; }

}  // namespace kernel
}  // namespace feather
