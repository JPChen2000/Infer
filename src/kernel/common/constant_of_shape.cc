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

#include "src/kernel/constant_of_shape.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "src/operator/control_tensor.h"
#include "util/fp16.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool g_common_constant_of_shape_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP32, "ConstantOfShape", []() {
        return std::make_unique<ConstantOfShapeKernel<DeviceType::COMMON, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::FP16, "ConstantOfShape", []() {
        return std::make_unique<ConstantOfShapeKernel<DeviceType::COMMON, DataType::FP16>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT32, "ConstantOfShape", []() {
        return std::make_unique<ConstantOfShapeKernel<DeviceType::COMMON, DataType::INT32>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::INT64, "ConstantOfShape", []() {
        return std::make_unique<ConstantOfShapeKernel<DeviceType::COMMON, DataType::INT64>>();
    });
    dispatcher.registerKernel(DeviceType::COMMON, DataType::BOOL, "ConstantOfShape", []() {
        return std::make_unique<ConstantOfShapeKernel<DeviceType::COMMON, DataType::BOOL>>();
    });
    return true;
}();

template <DataType dtype>
int32_t ComputeConstantOfShape(feather::operators::ConstantOfShapeParam* param) {
    if (param == nullptr || param->shape == nullptr || param->out == nullptr || !param->shape->IsInitialized() ||
        !param->out->IsInitialized() || param->output_type != dtype || param->out->data_type() != dtype) {
        return -1;
    }
    std::vector<int64_t> dims;
    if (!feather::operators::ReadIntegerTensor(param->shape, &dims) || dims.empty() ||
        std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim <= 0; })) {
        return -1;
    }
    int64_t expected_numel = 1;
    for (const auto dim : dims) {
        if (expected_numel > std::numeric_limits<int64_t>::max() / dim) {
            return -1;
        }
        expected_numel *= dim;
    }
    if (param->out->dims().data() != dims || param->out->numel() != expected_numel) {
        return -1;
    }

    const int64_t int_value = param->int_value;
    const float float_value = param->use_float_value ? param->float_value : static_cast<float>(int_value);
    if constexpr (dtype == DataType::FP32) {
        std::fill_n(param->out->mutable_data<float>(), expected_numel, float_value);
    } else if constexpr (dtype == DataType::FP16) {
        std::fill_n(param->out->mutable_data<uint16_t>(), expected_numel, FloatToHalf(float_value));
    } else if constexpr (dtype == DataType::INT32) {
        std::fill_n(param->out->mutable_data<int32_t>(), expected_numel, static_cast<int32_t>(int_value));
    } else if constexpr (dtype == DataType::INT64) {
        std::fill_n(param->out->mutable_data<int64_t>(), expected_numel, int_value);
    } else if constexpr (dtype == DataType::BOOL) {
        std::fill_n(param->out->mutable_data<uint8_t>(), expected_numel, int_value == 0 ? 0 : 1);
    } else {
        return -1;
    }
    param->out->set_data_type(dtype);
    return 0;
}

}  // namespace

template <>
int32_t ConstantOfShapeKernel<DeviceType::COMMON, DataType::FP32>::compute() {
    AutoTimer timer("Common::ConstantOfShape::FP32");
    return ComputeConstantOfShape<DataType::FP32>(ParamAs<feather::operators::ConstantOfShapeParam>());
}

template <>
int32_t ConstantOfShapeKernel<DeviceType::COMMON, DataType::FP16>::compute() {
    AutoTimer timer("Common::ConstantOfShape::FP16");
    return ComputeConstantOfShape<DataType::FP16>(ParamAs<feather::operators::ConstantOfShapeParam>());
}

template <>
int32_t ConstantOfShapeKernel<DeviceType::COMMON, DataType::INT32>::compute() {
    AutoTimer timer("Common::ConstantOfShape::INT32");
    return ComputeConstantOfShape<DataType::INT32>(ParamAs<feather::operators::ConstantOfShapeParam>());
}

template <>
int32_t ConstantOfShapeKernel<DeviceType::COMMON, DataType::INT64>::compute() {
    AutoTimer timer("Common::ConstantOfShape::INT64");
    return ComputeConstantOfShape<DataType::INT64>(ParamAs<feather::operators::ConstantOfShapeParam>());
}

template <>
int32_t ConstantOfShapeKernel<DeviceType::COMMON, DataType::BOOL>::compute() {
    AutoTimer timer("Common::ConstantOfShape::BOOL");
    return ComputeConstantOfShape<DataType::BOOL>(ParamAs<feather::operators::ConstantOfShapeParam>());
}

void EnsureCommonConstantOfShapeKernelsRegistered() { (void)g_common_constant_of_shape_kernels_registered; }

void EnsureConstantOfShapeKernelsRegistered() { EnsureCommonConstantOfShapeKernelsRegistered(); }

}  // namespace kernel
}  // namespace feather
