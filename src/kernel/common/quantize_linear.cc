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

#include "src/kernel/quantize_linear.h"

#include "src/kernel/common/quantize_linear_utils.h"

namespace feather {
namespace kernel {
namespace {

bool g_quantize_linear_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP32, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::FP16, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::BF16, "QuantizeLinear",
                                                []() { return std::make_unique<CommonQuantizeLinearKernel>(); });
    return true;
}();

}  // namespace

int32_t CommonQuantizeLinearKernel::compute() {
    auto* param = ParamAs<operators::QuantizeLinearParam>();
    if (param == nullptr || param->input == nullptr ||
        (param->input->data_type() != DataType::FP32 && param->input->data_type() != DataType::FP16 &&
         param->input->data_type() != DataType::BF16)) return -1;
    return quantize_linear_detail::ComputeQuantizeLinear(param);
}

void EnsureQuantizeLinearKernelsRegistered() { (void)g_quantize_linear_kernels_registered; }

}  // namespace kernel
}  // namespace feather
