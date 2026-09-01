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
void EnsureCudaEqualKernelsRegistered();
void EnsureCudaExpandKernelsRegistered();
void EnsureCudaWhereKernelsRegistered();
void EnsureCudaTransposeKernelsRegistered();
void EnsureCudaYoloDecodeKernelsRegistered();
void EnsureCudaQwenRmsNormKernelsRegistered();
void EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
void EnsureCudaQwenGatedDeltaKernelsRegistered();
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
    EnsureCudaEqualKernelsRegistered();
    EnsureCudaExpandKernelsRegistered();
    EnsureCudaWhereKernelsRegistered();
    EnsureCudaTransposeKernelsRegistered();
    EnsureCudaYoloDecodeKernelsRegistered();
    EnsureCudaQwenRmsNormKernelsRegistered();
    EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
    EnsureCudaQwenGatedDeltaKernelsRegistered();
    EnsureCudaQwenGemmArgmaxKernelsRegistered();
    EnsureCudaFp8KernelsRegistered();
    EnsureCudaInt8FusedKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
