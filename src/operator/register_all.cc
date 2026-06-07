#include "core/operator_registry.h"

namespace feather {
namespace operators {

void EnsureAddOperatorRegistered();
void EnsureConcatOperatorRegistered();
void EnsureConv2DOperatorRegistered();
void EnsureFcOperatorRegistered();
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
void EnsureSigmoidOperatorRegistered();
void EnsureSliceOperatorRegistered();
void EnsureSoftmaxOperatorRegistered();
void EnsureSplitOperatorRegistered();
void EnsureTransposeOperatorRegistered();

void EnsureBuiltinOperatorsRegistered() {
    EnsureAddOperatorRegistered();
    EnsureConcatOperatorRegistered();
    EnsureConv2DOperatorRegistered();
    EnsureFcOperatorRegistered();
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
    EnsureSigmoidOperatorRegistered();
    EnsureSliceOperatorRegistered();
    EnsureSoftmaxOperatorRegistered();
    EnsureSplitOperatorRegistered();
    EnsureTransposeOperatorRegistered();
}

}  // namespace operators
}  // namespace feather
