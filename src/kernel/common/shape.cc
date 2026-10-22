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

#include "src/kernel/shape.h"

#include <algorithm>
#include <memory>

#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_shape_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(DeviceType::COMMON, DataType::INT64, "Shape", []() {
        return std::make_unique<ShapeKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    return true;
}();

int32_t ComputeShape(feather::operators::ShapeParam* param) {
    if (param == nullptr || param->input == nullptr || param->out == nullptr || !param->out->IsInitialized()) {
        return -1;
    }
    const int64_t rank = static_cast<int64_t>(param->input->dims().size());
    int64_t start = param->start < 0 ? param->start + rank : param->start;
    int64_t end = param->end < 0 ? param->end + rank : param->end;
    start = std::max<int64_t>(0, std::min<int64_t>(start, rank));
    end = std::max<int64_t>(0, std::min<int64_t>(end, rank));
    if (end < start || param->out->numel() != end - start || param->out->memory_size() <
                                                              static_cast<size_t>(end - start) * sizeof(int64_t)) {
        return -1;
    }
    param->out->set_data_type(DataType::INT64);
    auto* output = param->out->mutable_data<int64_t>();
    for (int64_t i = 0; i < end - start; ++i) {
        output[i] = param->input->dims()[static_cast<size_t>(start + i)];
    }
    return 0;
}

}  // namespace

template <>
int32_t ShapeKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::Shape::INT64");
    return ComputeShape(ParamAs<feather::operators::ShapeParam>());
}

void EnsureCommonShapeKernelsRegistered() { (void)g_common_shape_kernels_registered; }

void EnsureShapeKernelsRegistered() { EnsureCommonShapeKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
