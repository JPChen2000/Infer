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

#include "core/kernel.h"
#include "src/kernel/common/int8_fused.h"

namespace feather {
namespace kernel {

void EnsureCudaAddKernelsRegistered();
void EnsureCudaBatchNormalizationKernelsRegistered();
void EnsureCudaDivKernelsRegistered();
void EnsureCudaErfKernelsRegistered();
void EnsureCudaExpKernelsRegistered();
void EnsureCudaSinKernelsRegistered();
void EnsureCudaCosKernelsRegistered();
void EnsureCudaNegKernelsRegistered();
void EnsureCudaSoftplusKernelsRegistered();
void EnsureCudaConcatKernelsRegistered();
void EnsureCudaConv2DKernelsRegistered();
void EnsureCudaConvTransposeKernelsRegistered();
void EnsureCudaNonMaxSuppressionKernelsRegistered();
void EnsureCudaNormalizationKernelsRegistered();
void EnsureCudaCachedAttentionKernelsRegistered();
void EnsureCudaFcKernelsRegistered();
void EnsureCudaInt8KernelsRegistered();
void EnsureCudaStandardInt8KernelsRegistered();
void EnsureCudaFlattenKernelsRegistered();
void EnsureCudaGemmKernelsRegistered();
void EnsureCudaGlobalAveragePoolKernelsRegistered();
void EnsureCudaIdentityKernelsRegistered();
void EnsureCudaMatMulKernelsRegistered();
void EnsureCudaMulKernelsRegistered();
void EnsureCudaPoolKernelsRegistered();
void EnsureCudaPowKernelsRegistered();
void EnsureCudaReluKernelsRegistered();
void EnsureCudaReshapeKernelsRegistered();
void EnsureCudaResizeKernelsRegistered();
void EnsureCudaResizeConcatKernelsRegistered();
void EnsureCudaSiluKernelsRegistered();
void EnsureCudaSigmoidKernelsRegistered();
void EnsureCudaSliceKernelsRegistered();
void EnsureCudaSoftmaxKernelsRegistered();
void EnsureCudaSplitKernelsRegistered();
void EnsureCudaSqrtKernelsRegistered();
void EnsureCudaSubKernelsRegistered();
void EnsureCudaTanhKernelsRegistered();
void EnsureCudaUnsqueezeKernelsRegistered();
void EnsureCudaSqueezeKernelsRegistered();
void EnsureCudaCastKernelsRegistered();
void EnsureCudaReduceMeanKernelsRegistered();
void EnsureCudaReduceSumKernelsRegistered();
void EnsureCudaGatherKernelsRegistered();
void EnsureCudaYoloTopKKernelsRegistered();
void EnsureCudaYoloSelectionKernelsRegistered();
void EnsureCudaEqualKernelsRegistered();
void EnsureCudaExpandKernelsRegistered();
void EnsureCudaWhereKernelsRegistered();
void EnsureCudaTransposeKernelsRegistered();
void EnsureCudaYoloDecodeKernelsRegistered();
void EnsureCudaQwenRmsNormKernelsRegistered();
void EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
void EnsureCudaQwenGatedDeltaKernelsRegistered();
void EnsureCudaQwenGatedDeltaSequenceKernelsRegistered();
void EnsureCudaQwenGemmArgmaxKernelsRegistered();
void EnsureCudaFp8KernelsRegistered();
void EnsureCudaInt8FusedKernelsRegistered();

void EnsureCudaKernelsRegistered() {
    EnsureCudaAddKernelsRegistered();
    EnsureCudaBatchNormalizationKernelsRegistered();
    EnsureCudaDivKernelsRegistered();
    EnsureCudaErfKernelsRegistered();
    EnsureCudaExpKernelsRegistered();
    EnsureCudaSinKernelsRegistered();
    EnsureCudaCosKernelsRegistered();
    EnsureCudaNegKernelsRegistered();
    EnsureCudaSoftplusKernelsRegistered();
    EnsureCudaConcatKernelsRegistered();
    EnsureCudaConv2DKernelsRegistered();
    EnsureCudaConvTransposeKernelsRegistered();
    EnsureCudaNonMaxSuppressionKernelsRegistered();
    EnsureCudaNormalizationKernelsRegistered();
    EnsureCudaCachedAttentionKernelsRegistered();
    EnsureCudaFcKernelsRegistered();
    EnsureCudaInt8KernelsRegistered();
    EnsureCudaStandardInt8KernelsRegistered();
    EnsureCudaFlattenKernelsRegistered();
    EnsureCudaGemmKernelsRegistered();
    EnsureCudaGlobalAveragePoolKernelsRegistered();
    EnsureCudaIdentityKernelsRegistered();
    EnsureCudaMatMulKernelsRegistered();
    EnsureCudaMulKernelsRegistered();
    EnsureCudaPoolKernelsRegistered();
    EnsureCudaPowKernelsRegistered();
    EnsureCudaReluKernelsRegistered();
    EnsureCudaReshapeKernelsRegistered();
    EnsureCudaResizeKernelsRegistered();
    EnsureCudaResizeConcatKernelsRegistered();
    EnsureCudaSiluKernelsRegistered();
    EnsureCudaSigmoidKernelsRegistered();
    EnsureCudaSliceKernelsRegistered();
    EnsureCudaSoftmaxKernelsRegistered();
    EnsureCudaSplitKernelsRegistered();
    EnsureCudaSqrtKernelsRegistered();
    EnsureCudaSubKernelsRegistered();
    EnsureCudaTanhKernelsRegistered();
    EnsureCudaUnsqueezeKernelsRegistered();
    EnsureCudaSqueezeKernelsRegistered();
    EnsureCudaCastKernelsRegistered();
    EnsureCudaReduceMeanKernelsRegistered();
    EnsureCudaReduceSumKernelsRegistered();
    EnsureCudaGatherKernelsRegistered();
    EnsureCudaYoloTopKKernelsRegistered();
    EnsureCudaYoloSelectionKernelsRegistered();
    EnsureCudaEqualKernelsRegistered();
    EnsureCudaExpandKernelsRegistered();
    EnsureCudaWhereKernelsRegistered();
    EnsureCudaTransposeKernelsRegistered();
    EnsureCudaYoloDecodeKernelsRegistered();
    EnsureCudaQwenRmsNormKernelsRegistered();
    EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
    EnsureCudaQwenGatedDeltaKernelsRegistered();
    EnsureCudaQwenGatedDeltaSequenceKernelsRegistered();
    EnsureCudaQwenGemmArgmaxKernelsRegistered();
    EnsureCudaFp8KernelsRegistered();
    EnsureCudaInt8FusedKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
