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

#include "src/kernel/dequantize_linear.h"

#include "src/kernel/common/quantize_linear_utils.h"

namespace feather {
namespace kernel {
namespace {

bool g_dequantize_linear_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT8, "DequantizeLinear",
                                                []() { return std::make_unique<CommonDequantizeLinearKernel>(); });
    return true;
}();

}  // namespace

int32_t CommonDequantizeLinearKernel::compute() {
    auto* param = ParamAs<operators::DequantizeLinearParam>();
    if (param == nullptr || param->input == nullptr || param->input->data_type() != DataType::INT8) return -1;
    return quantize_linear_detail::ComputeDequantizeLinear(param);
}

void EnsureDequantizeLinearKernelsRegistered() { (void)g_dequantize_linear_kernels_registered; }

}  // namespace kernel
}  // namespace feather
