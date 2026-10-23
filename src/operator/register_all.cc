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

#include "core/operator_registry.h"
#include "src/operator/quantize_linear_op.h"
#include "src/operator/dequantize_linear_op.h"

namespace feather {
namespace operators {

void EnsureAddOperatorRegistered();
void EnsureBatchNormalizationOperatorRegistered();
void EnsureDivOperatorRegistered();
void EnsureErfOperatorRegistered();
void EnsureExpOperatorRegistered();
void EnsureSinOperatorRegistered();
void EnsureCosOperatorRegistered();
void EnsureNegOperatorRegistered();
void EnsureSoftplusOperatorRegistered();
void EnsureConcatOperatorRegistered();
void EnsureConv2DOperatorRegistered();
void EnsureFcOperatorRegistered();
void EnsureGlobalAveragePoolOperatorRegistered();
void EnsureFlattenOperatorRegistered();
void EnsureGemmOperatorRegistered();
void EnsureIdentityOperatorRegistered();
void EnsureMatMulOperatorRegistered();
void EnsureMulOperatorRegistered();
void EnsurePoolOperatorsRegistered();
void EnsurePowOperatorRegistered();
void EnsureReluOperatorRegistered();
void EnsureReshapeOperatorRegistered();
void EnsureResizeOperatorRegistered();
void EnsureResizeConcatOperatorRegistered();
void EnsureSiluOperatorRegistered();
void EnsureSigmoidOperatorRegistered();
void EnsureSliceOperatorRegistered();
void EnsureSoftmaxOperatorRegistered();
void EnsureSplitOperatorRegistered();
void EnsureTransposeOperatorRegistered();
void EnsureUnsqueezeOperatorRegistered();
void EnsureSqueezeOperatorRegistered();
void EnsureCastOperatorRegistered();
void EnsureReduceMeanOperatorRegistered();
void EnsureReduceSumOperatorRegistered();
void EnsureGatherOperatorRegistered();
void EnsureEqualOperatorRegistered();
void EnsureSqrtOperatorRegistered();
void EnsureSubOperatorRegistered();
void EnsureTanhOperatorRegistered();
void EnsureShapeOperatorRegistered();
void EnsureConstantOfShapeOperatorRegistered();
void EnsureExpandOperatorRegistered();
void EnsureWhereOperatorRegistered();
void EnsureQwenGatedDeltaOperatorsRegistered();
void EnsureQwenGatedDeltaSequenceOperatorsRegistered();
void EnsureQwenGemmArgmaxOperatorRegistered();
void EnsureQwenDepthwiseConvStateOperatorRegistered();
void EnsureQwenRmsNormOperatorRegistered();
void EnsureYoloDecodeOperatorRegistered();
void EnsureYoloSelectionOperatorsRegistered();
void EnsureConvTransposeOperatorRegistered();
void EnsureNonMaxSuppressionOperatorRegistered();
void EnsureNormalizationOperatorsRegistered();
void EnsureCachedAttentionOperatorRegistered();

void RegisterBuiltinOperators() {
    EnsureAddOperatorRegistered();
    EnsureBatchNormalizationOperatorRegistered();
    EnsureDivOperatorRegistered();
    EnsureErfOperatorRegistered();
    EnsureExpOperatorRegistered();
    EnsureSinOperatorRegistered();
    EnsureCosOperatorRegistered();
    EnsureNegOperatorRegistered();
    EnsureSoftplusOperatorRegistered();
    EnsureConcatOperatorRegistered();
    EnsureConv2DOperatorRegistered();
    EnsureFcOperatorRegistered();
    EnsureGlobalAveragePoolOperatorRegistered();
    EnsureFlattenOperatorRegistered();
    EnsureGemmOperatorRegistered();
    EnsureIdentityOperatorRegistered();
    EnsureMatMulOperatorRegistered();
    EnsureMulOperatorRegistered();
    EnsurePoolOperatorsRegistered();
    EnsurePowOperatorRegistered();
    EnsureReluOperatorRegistered();
    EnsureReshapeOperatorRegistered();
    EnsureResizeOperatorRegistered();
    EnsureResizeConcatOperatorRegistered();
    EnsureSiluOperatorRegistered();
    EnsureSigmoidOperatorRegistered();
    EnsureSliceOperatorRegistered();
    EnsureSoftmaxOperatorRegistered();
    EnsureSplitOperatorRegistered();
    EnsureTransposeOperatorRegistered();
    EnsureUnsqueezeOperatorRegistered();
    EnsureSqueezeOperatorRegistered();
    EnsureCastOperatorRegistered();
    EnsureQuantizeLinearOperatorRegistered();
    EnsureDequantizeLinearOperatorRegistered();
    EnsureReduceMeanOperatorRegistered();
    EnsureReduceSumOperatorRegistered();
    EnsureGatherOperatorRegistered();
    EnsureEqualOperatorRegistered();
    EnsureSqrtOperatorRegistered();
    EnsureSubOperatorRegistered();
    EnsureTanhOperatorRegistered();
    EnsureShapeOperatorRegistered();
    EnsureConstantOfShapeOperatorRegistered();
    EnsureExpandOperatorRegistered();
    EnsureWhereOperatorRegistered();
    EnsureQwenGatedDeltaOperatorsRegistered();
    EnsureQwenGatedDeltaSequenceOperatorsRegistered();
    EnsureQwenGemmArgmaxOperatorRegistered();
    EnsureQwenDepthwiseConvStateOperatorRegistered();
    EnsureQwenRmsNormOperatorRegistered();
    EnsureYoloDecodeOperatorRegistered();
    EnsureYoloSelectionOperatorsRegistered();
    EnsureConvTransposeOperatorRegistered();
    EnsureNonMaxSuppressionOperatorRegistered();
    EnsureNormalizationOperatorsRegistered();
    EnsureCachedAttentionOperatorRegistered();
}

void EnsureBuiltinOperatorsRegistered() { RegisterBuiltinOperators(); }

}  // namespace operators
}  // namespace feather
