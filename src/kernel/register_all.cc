#include "core/kernel.h"

#include "src/kernel/add.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/fc.h"
#include "src/kernel/flatten.h"
#include "src/kernel/gemm.h"
#include "src/kernel/identity.h"
#include "src/kernel/matmul.h"
#include "src/kernel/mul.h"
#include "src/kernel/pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/reshape.h"
#include "src/kernel/relu.h"
#include "src/kernel/resize.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/slice.h"
#include "src/kernel/softmax.h"
#include "src/kernel/split.h"
#include "src/kernel/transpose.h"

namespace feather {
namespace kernel {

void EnsureBuiltinKernelsRegistered() {
    EnsureAddKernelsRegistered();
    EnsureConcatKernelsRegistered();
    EnsureConv2DKernelsRegistered();
    EnsureFcKernelsRegistered();
    EnsureFlattenKernelsRegistered();
    EnsureGemmKernelsRegistered();
    EnsureIdentityKernelsRegistered();
    EnsureMatMulKernelsRegistered();
    EnsureMulKernelsRegistered();
    EnsurePoolKernelsRegistered();
    EnsurePowKernelsRegistered();
    EnsureReshapeKernelsRegistered();
    EnsureReluKernelsRegistered();
    EnsureResizeKernelsRegistered();
    EnsureSigmoidKernelsRegistered();
    EnsureSliceKernelsRegistered();
    EnsureSoftmaxKernelsRegistered();
    EnsureSplitKernelsRegistered();
    EnsureTransposeKernelsRegistered();
}

}  // namespace kernel
}  // namespace feather
