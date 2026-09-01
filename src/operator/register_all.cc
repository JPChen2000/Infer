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
void EnsureQwenGemmArgmaxOperatorRegistered();
void EnsureQwenDepthwiseConvStateOperatorRegistered();
void EnsureQwenRmsNormOperatorRegistered();
void EnsureYoloDecodeOperatorRegistered();

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
    EnsureQwenGemmArgmaxOperatorRegistered();
    EnsureQwenDepthwiseConvStateOperatorRegistered();
    EnsureQwenRmsNormOperatorRegistered();
    EnsureYoloDecodeOperatorRegistered();
}

void EnsureBuiltinOperatorsRegistered() { RegisterBuiltinOperators(); }

}  // namespace operators
}  // namespace feather
