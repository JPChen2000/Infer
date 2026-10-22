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

#ifndef FEATHER_KERNEL_COMMON_INT8_FUSED_H
#define FEATHER_KERNEL_COMMON_INT8_FUSED_H

#include <cstdint>
#include <memory>

#include "core/kernel.h"
#include "src/operator/params.h"

namespace feather {
namespace kernel {

struct Int8Quantization {
    float scale{1.0f};
    int32_t zero_point{0};
};

namespace common {

int32_t ComputeShapeInt8(feather::operators::ShapeParam* param);
int32_t ComputeResizeConcatInt8(feather::operators::ResizeConcatParam* param);
int32_t ComputeYoloDecodeInt8(feather::operators::YoloDecodeParam* param);
Int8Quantization GetInt8Quantization(const std::shared_ptr<Tensor>& tensor);

}  // namespace common

void EnsureCommonInt8FusedKernelsRegistered();
void EnsureX86Int8FusedKernelsRegistered();
void EnsureCudaInt8FusedKernelsRegistered();

}  // namespace kernel
}  // namespace feather

#endif  // FEATHER_KERNEL_COMMON_INT8_FUSED_H
