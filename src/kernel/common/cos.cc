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

#include "src/kernel/cos.h"

#include <cmath>

#include "src/kernel/common/elementwise_broadcast.h"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool g_cos_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::FP32>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::FP16>>(); });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BF16, "Cos", []() { return std::make_unique<CosKernel<DeviceType::COMMON, DataType::BF16>>(); });
    return true;
}();

template <DataType dtype>
int32_t RunCos(feather::operators::UnaryParam* param) {
    return param == nullptr ? -1 : common_detail::RunUnary<dtype>(param->out.get(), param->input.get(),
                                                                    [](float value) { return std::cos(value); });
}

}  // namespace

template <> int32_t CosKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::Cos::FP32");
    return RunCos<DataType::FP32>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t CosKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::Cos::FP16");
    return RunCos<DataType::FP16>(ParamAs<feather::operators::UnaryParam>());
}
template <> int32_t CosKernel<DeviceType::COMMON, DataType::BF16>::compute() {
    AutoTimer timer("Common::Cos::BF16");
    return RunCos<DataType::BF16>(ParamAs<feather::operators::UnaryParam>());
}

void EnsureCosKernelsRegistered() { (void)g_cos_kernels_registered; }

}  // namespace kernel
}  // namespace feather
