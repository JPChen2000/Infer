#include "core/kernel.h"

#include "src/kernel/add.h"
#include "src/kernel/batch_normalization.h"
#include "src/kernel/div.h"
#include "src/kernel/erf.h"
#include "src/kernel/exp.h"
#include "src/kernel/sin.h"
#include "src/kernel/cos.h"
#include "src/kernel/neg.h"
#include "src/kernel/softplus.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/fc.h"
#include "src/kernel/global_average_pool.h"
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
#include "src/kernel/resize_concat.h"
#include "src/kernel/silu.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/slice.h"
#include "src/kernel/softmax.h"
#include "src/kernel/split.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/sub.h"
#include "src/kernel/tanh.h"
#include "src/kernel/transpose.h"
#include "src/kernel/unsqueeze.h"
#include "src/kernel/squeeze.h"
#include "src/kernel/cast.h"
#include "src/kernel/reduce_mean.h"
#include "src/kernel/reduce_sum.h"
#include "src/kernel/gather.h"
#include "src/kernel/equal.h"
#include "src/kernel/shape.h"
#include "src/kernel/constant_of_shape.h"
#include "src/kernel/expand.h"
#include "src/kernel/where.h"
#include "src/kernel/yolo_decode.h"

namespace feather {
namespace kernel {

#ifdef FEATHER_WITH_CUDA
void EnsureCudaKernelsRegistered();
#endif

void EnsureBuiltinKernelsRegistered() {
    EnsureAddKernelsRegistered();
    EnsureBatchNormalizationKernelsRegistered();
    EnsureDivKernelsRegistered();
    EnsureErfKernelsRegistered();
    EnsureExpKernelsRegistered();
    EnsureSinKernelsRegistered();
    EnsureCosKernelsRegistered();
    EnsureNegKernelsRegistered();
    EnsureSoftplusKernelsRegistered();
    EnsureConcatKernelsRegistered();
    EnsureConv2DKernelsRegistered();
    EnsureFcKernelsRegistered();
    EnsureGlobalAveragePoolKernelsRegistered();
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
    EnsureResizeConcatKernelsRegistered();
    EnsureSiluKernelsRegistered();
    EnsureSigmoidKernelsRegistered();
    EnsureSliceKernelsRegistered();
    EnsureSoftmaxKernelsRegistered();
    EnsureSplitKernelsRegistered();
    EnsureSqrtKernelsRegistered();
    EnsureSubKernelsRegistered();
    EnsureTanhKernelsRegistered();
    EnsureTransposeKernelsRegistered();
    EnsureUnsqueezeKernelsRegistered();
    EnsureSqueezeKernelsRegistered();
    EnsureCastKernelsRegistered();
    EnsureReduceMeanKernelsRegistered();
    EnsureReduceSumKernelsRegistered();
    EnsureGatherKernelsRegistered();
    EnsureEqualKernelsRegistered();
    EnsureShapeKernelsRegistered();
    EnsureConstantOfShapeKernelsRegistered();
    EnsureExpandKernelsRegistered();
    EnsureWhereKernelsRegistered();
    EnsureYoloDecodeKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaKernelsRegistered();
#endif
}

}  // namespace kernel
}  // namespace feather
