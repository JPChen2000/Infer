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

#include "src/kernel/common/int8_fused.h"

#include <memory>

#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/yolo_decode.h"

namespace feather {
namespace kernel {

namespace {



}  // namespace

template <>
int32_t ShapeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeShapeInt8(ParamAs<feather::operators::ShapeParam>());
}

template <>
int32_t ResizeConcatKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeResizeConcatInt8(ParamAs<feather::operators::ResizeConcatParam>());
}

template <>
int32_t YoloDecodeKernel<DeviceType::X86, DataType::INT8>::compute() {
    return common::ComputeYoloDecodeInt8(ParamAs<feather::operators::YoloDecodeParam>());
}



}  // namespace kernel
}  // namespace feather

