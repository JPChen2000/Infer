#include "core/operator_registry.h"

namespace feather {
namespace operators {

void EnsureAddOperatorRegistered();
void EnsureBatchNormalizationOperatorRegistered();
void EnsureDivOperatorRegistered();
void EnsureErfOperatorRegistered();
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
void EnsureGatherOperatorRegistered();
void EnsureEqualOperatorRegistered();
void EnsureSqrtOperatorRegistered();
void EnsureSubOperatorRegistered();
void EnsureTanhOperatorRegistered();
void EnsureShapeOperatorRegistered();
void EnsureExpandOperatorRegistered();
void EnsureWhereOperatorRegistered();
void EnsureYoloDecodeOperatorRegistered();

void EnsureBuiltinOperatorsRegistered() {
    EnsureAddOperatorRegistered();
    EnsureBatchNormalizationOperatorRegistered();
    EnsureDivOperatorRegistered();
    EnsureErfOperatorRegistered();
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
    EnsureReduceMeanOperatorRegistered();
    EnsureGatherOperatorRegistered();
    EnsureEqualOperatorRegistered();
    EnsureSqrtOperatorRegistered();
    EnsureSubOperatorRegistered();
    EnsureTanhOperatorRegistered();
    EnsureShapeOperatorRegistered();
    EnsureExpandOperatorRegistered();
    EnsureWhereOperatorRegistered();
    EnsureYoloDecodeOperatorRegistered();
}

}  // namespace operators
}  // namespace feather
