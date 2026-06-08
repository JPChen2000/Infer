#include "core/kernel.h"

namespace feather {
namespace kernel {

void EnsureCudaAddKernelsRegistered();
void EnsureCudaConcatKernelsRegistered();
void EnsureCudaConv2DKernelsRegistered();
void EnsureCudaFcKernelsRegistered();
void EnsureCudaFlattenKernelsRegistered();
void EnsureCudaGemmKernelsRegistered();
void EnsureCudaIdentityKernelsRegistered();
void EnsureCudaMatMulKernelsRegistered();
void EnsureCudaMulKernelsRegistered();
void EnsureCudaPoolKernelsRegistered();
void EnsureCudaPowKernelsRegistered();
void EnsureCudaReluKernelsRegistered();
void EnsureCudaReshapeKernelsRegistered();
void EnsureCudaResizeKernelsRegistered();
void EnsureCudaSiluKernelsRegistered();
void EnsureCudaSigmoidKernelsRegistered();
void EnsureCudaSliceKernelsRegistered();
void EnsureCudaSoftmaxKernelsRegistered();
void EnsureCudaSplitKernelsRegistered();
void EnsureCudaTransposeKernelsRegistered();
void EnsureCudaYoloDecodeKernelsRegistered();

void EnsureCudaKernelsRegistered() {
    EnsureCudaAddKernelsRegistered();
    EnsureCudaConcatKernelsRegistered();
    EnsureCudaConv2DKernelsRegistered();
    EnsureCudaFcKernelsRegistered();
    EnsureCudaFlattenKernelsRegistered();
    EnsureCudaGemmKernelsRegistered();
    EnsureCudaIdentityKernelsRegistered();
    EnsureCudaMatMulKernelsRegistered();
    EnsureCudaMulKernelsRegistered();
    EnsureCudaPoolKernelsRegistered();
    EnsureCudaPowKernelsRegistered();
    EnsureCudaReluKernelsRegistered();
    EnsureCudaReshapeKernelsRegistered();
    EnsureCudaResizeKernelsRegistered();
    EnsureCudaSiluKernelsRegistered();
    EnsureCudaSigmoidKernelsRegistered();
    EnsureCudaSliceKernelsRegistered();
    EnsureCudaSoftmaxKernelsRegistered();
    EnsureCudaSplitKernelsRegistered();
    EnsureCudaTransposeKernelsRegistered();
    EnsureCudaYoloDecodeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
