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

#include "util/timer.h"

namespace feather {
namespace kernel {

template <>
int32_t ExpKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Exp::FP32");
    auto* param = ParamAs<feather::operators::UnaryParam>();
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 ||
        (param->out->data_type() != DataType::UNKNOWN && param->out->data_type() != DataType::FP32) ||
        param->input->dims().data() != param->out->dims().data()) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    for (int64_t index = 0; index < param->input->numel(); ++index) {
        output[index] = std::exp(input[index]);
    }
    return 0;
}



}  // namespace kernel
}  // namespace feather
