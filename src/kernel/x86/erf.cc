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
#include <memory>

#include "util/threading.h"
#include "util/timer.h"

#if defined(FEATHER_WITH_OPENMP)
#include <omp.h>
#endif

namespace feather {
namespace kernel {

namespace {

constexpr int64_t kErfParallelThreshold = 1 << 15;

}  // namespace

template <>
int32_t ErfKernel<DeviceType::X86, DataType::FP32>::compute() {
    AutoTimer timer("X86::Erf::FP32");
    auto* param = ParamAs<feather::operators::UnaryParam>();
    if (param == nullptr || param->input == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::FP32 || param->out->data_type() != DataType::FP32 ||
        param->input->dims().data() != param->out->dims().data()) {
        return -1;
    }

    param->out->set_data_type(DataType::FP32);
    const float* input = param->input->data<float>();
    float* output = param->out->mutable_data<float>();
    const int64_t numel = param->input->numel();

#if defined(FEATHER_WITH_OPENMP)
    if (numel >= kErfParallelThreshold && !omp_in_parallel()) {
        const size_t worker_count = ThreadCountForWorkItems(numel);
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int64_t index = 0; index < numel; ++index) {
            output[index] = std::erf(input[index]);
        }
        return 0;
    }
#endif
    for (int64_t index = 0; index < numel; ++index) {
        output[index] = std::erf(input[index]);
    }
    return 0;
}



}  // namespace kernel
}  // namespace feather
