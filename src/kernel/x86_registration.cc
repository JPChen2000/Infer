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

// ISA-neutral registration and CPU feature gate for the native backend.
#include "src/kernel/add.h"
#include "src/kernel/batch_normalization.h"
#include "src/kernel/cast.h"
#include "src/kernel/common/int8_fused.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/cos.h"
#include "src/kernel/div.h"
#include "src/kernel/equal.h"
#include "src/kernel/erf.h"
#include "src/kernel/exp.h"
#include "src/kernel/expand.h"
#include "src/kernel/fc.h"
#include "src/kernel/flatten.h"
#include "src/kernel/fp8_host.h"
#include "src/kernel/gather.h"
#include "src/kernel/gemm.h"
#include "src/kernel/global_average_pool.h"
#include "src/kernel/identity.h"
#include "src/kernel/matmul.h"
#include "src/kernel/mul.h"
#include "src/kernel/neg.h"
#include "src/kernel/pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/qwen_depthwise_conv.h"
#include "src/kernel/qwen_gated_delta.h"
#include "src/kernel/qwen_gemm_argmax.h"
#include "src/kernel/qwen_rms_norm.h"
#include "src/kernel/reduce_mean.h"
#include "src/kernel/reduce_sum.h"
#include "src/kernel/relu.h"
#include "src/kernel/reshape.h"
#include "src/kernel/resize.h"
#include "src/kernel/resize_concat.h"
#include "src/kernel/shape.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/silu.h"
#include "src/kernel/sin.h"
#include "src/kernel/slice.h"
#include "src/kernel/softmax.h"
#include "src/kernel/softplus.h"
#include "src/kernel/split.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/squeeze.h"
#include "src/kernel/sub.h"
#include "src/kernel/tanh.h"
#include "src/kernel/transpose.h"
#include "src/kernel/unsqueeze.h"
#include "src/kernel/where.h"
#include "src/kernel/yolo_decode.h"
#include "util/cpu_features.h"

namespace feather {
namespace kernel {
void EnsureX86Int8KernelsRegistered();
void EnsureX86AddKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Add",
            []() { return std::make_unique<AddKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Add",
            []() { return std::make_unique<AddKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Add",
            []() { return std::make_unique<AddKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Add",
            []() { return std::make_unique<AddKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Add",
            []() { return std::make_unique<AddKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86BatchNormalizationKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "BatchNormalization",
            []() { return std::make_unique<BatchNormalizationKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "BatchNormalization",
            []() { return std::make_unique<BatchNormalizationKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86CastKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::BF16, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::BF16>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::FP32, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP32>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::FP8E4M3, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP8E4M3>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::FP8E5M2, "Cast", []() {
            return std::make_unique<CastKernel<DeviceType::X86, DataType::FP8E5M2>>();
        });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ConcatKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::INT8, "Concat",
            []() { return std::make_unique<ConcatKernel<DeviceType::X86, DataType::INT8>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86Conv2DKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Conv2D",
            []() { return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Conv2D",
            []() { return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Conv2D",
            []() { return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86DivKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Div",
            []() { return std::make_unique<DivKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Div",
            []() { return std::make_unique<DivKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Div",
            []() { return std::make_unique<DivKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ErfKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Erf",
            []() { return std::make_unique<ErfKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ExpKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Exp",
            []() { return std::make_unique<ExpKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ExpandKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Expand",
            []() { return std::make_unique<ExpandKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86FcKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool g_fc_x86_kernels_registered = []() {
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP32, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP32>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP16, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP16>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP8E4M3, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
    KernelDispatcher::instance().registerKernel(
        DeviceType::X86, DataType::FP8E5M2, "FC",
        []() { return std::make_unique<FcKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
    return true;
}();
    (void)g_fc_x86_kernels_registered;
    EnsureX86Int8KernelsRegistered();
#endif
}

void EnsureX86FlattenKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Flatten",
            []() { return std::make_unique<FlattenKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Flatten",
            []() { return std::make_unique<FlattenKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86Fp8KernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool g_x86_fp8_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
#define REGISTER_X86_FP8(op, klass)                                                                       \
    dispatcher.registerKernel(DeviceType::X86, DataType::FP8E4M3, op, []() {                           \
        return std::make_unique<klass<DeviceType::X86, DataType::FP8E4M3>>();                           \
    });                                                                                                   \
    dispatcher.registerKernel(DeviceType::X86, DataType::FP8E5M2, op, []() {                           \
        return std::make_unique<klass<DeviceType::X86, DataType::FP8E5M2>>();                           \
    });

    REGISTER_X86_FP8("BatchNormalization", BatchNormalizationKernel)
    REGISTER_X86_FP8("Concat", ConcatKernel)
    REGISTER_X86_FP8("Conv2D", Conv2DKernel)
    REGISTER_X86_FP8("Cos", CosKernel)
    REGISTER_X86_FP8("Erf", ErfKernel)
    REGISTER_X86_FP8("Exp", ExpKernel)
    REGISTER_X86_FP8("Expand", ExpandKernel)
    REGISTER_X86_FP8("Flatten", FlattenKernel)
    REGISTER_X86_FP8("GlobalAveragePool", GlobalAveragePoolKernel)
    REGISTER_X86_FP8("Gather", GatherKernel)
    REGISTER_X86_FP8("Identity", IdentityKernel)
    REGISTER_X86_FP8("AvgPool", AvgPoolKernel)
    REGISTER_X86_FP8("MaxPool", MaxPoolKernel)
    REGISTER_X86_FP8("Neg", NegKernel)
    REGISTER_X86_FP8("Pow", PowKernel)
    REGISTER_X86_FP8("ReduceMean", ReduceMeanKernel)
    REGISTER_X86_FP8("ReduceSum", ReduceSumKernel)
    REGISTER_X86_FP8("ReLU", ReluKernel)
    REGISTER_X86_FP8("Reshape", ReshapeKernel)
    REGISTER_X86_FP8("Resize", ResizeKernel)
    REGISTER_X86_FP8("ResizeConcat", ResizeConcatKernel)
    REGISTER_X86_FP8("Sigmoid", SigmoidKernel)
    REGISTER_X86_FP8("SiLU", SiluKernel)
    REGISTER_X86_FP8("Sin", SinKernel)
    REGISTER_X86_FP8("Slice", SliceKernel)
    REGISTER_X86_FP8("Softmax", SoftmaxKernel)
    REGISTER_X86_FP8("Softplus", SoftplusKernel)
    REGISTER_X86_FP8("Split", SplitKernel)
    REGISTER_X86_FP8("Sqrt", SqrtKernel)
    REGISTER_X86_FP8("Tanh", TanhKernel)
    REGISTER_X86_FP8("Transpose", TransposeKernel)
    REGISTER_X86_FP8("Unsqueeze", UnsqueezeKernel)
    REGISTER_X86_FP8("Squeeze", SqueezeKernel)
    REGISTER_X86_FP8("Where", WhereKernel)
    REGISTER_X86_FP8("Equal", EqualKernel)
#undef REGISTER_X86_FP8
    return true;
}();
    (void)g_x86_fp8_kernels_registered;
#endif
}

void EnsureX86GemmKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Gemm",
            []() { return std::make_unique<GemmKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86IdentityKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Identity",
            []() { return std::make_unique<IdentityKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Identity",
            []() { return std::make_unique<IdentityKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Identity",
            []() { return std::make_unique<IdentityKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86Int8KernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static const bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "FC", []() {
            return std::make_unique<FcKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Gemm", []() {
            return std::make_unique<GemmKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "MatMul", []() {
            return std::make_unique<MatMulKernel<DeviceType::X86, DataType::INT8>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Conv2D", []() {
            return std::make_unique<Conv2DKernel<DeviceType::X86, DataType::INT8>>();
        });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86Int8FusedKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool g_x86_int8_fused_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "Shape", []() {
        return std::make_unique<ShapeKernel<DeviceType::X86, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "ResizeConcat", []() {
        return std::make_unique<ResizeConcatKernel<DeviceType::X86, DataType::INT8>>();
    });
    dispatcher.registerKernel(DeviceType::X86, DataType::INT8, "YoloDecode", []() {
        return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::INT8>>();
    });
    return true;
}();
    (void)g_x86_int8_fused_kernels_registered;
#endif
}

void EnsureX86MatMulKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "MatMul",
            []() { return std::make_unique<MatMulKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86MulKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::BF16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Mul",
            []() { return std::make_unique<MulKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86PoolKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "AvgPool",
            []() { return std::make_unique<AvgPoolKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "AvgPool",
            []() { return std::make_unique<AvgPoolKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "MaxPool",
            []() { return std::make_unique<MaxPoolKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "MaxPool",
            []() { return std::make_unique<MaxPoolKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86PowKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Pow",
            []() { return std::make_unique<PowKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Pow",
            []() { return std::make_unique<PowKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86QwenDepthwiseConvStateKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::BF16, "QwenDepthwiseConvState", []() {
            return std::make_unique<QwenDepthwiseConvStateKernel<DeviceType::X86, DataType::BF16>>();
        });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E4M3,
                                                    "QwenDepthwiseConvState", []() {
                                                        return std::make_unique<
                                                            QwenDepthwiseConvStateKernel<DeviceType::X86,
                                                                                         DataType::FP8E4M3>>();
                                                    });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E5M2,
                                                    "QwenDepthwiseConvState", []() {
                                                        return std::make_unique<
                                                            QwenDepthwiseConvStateKernel<DeviceType::X86,
                                                                                         DataType::FP8E5M2>>();
                                                    });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86QwenGatedDeltaKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        auto& d = KernelDispatcher::instance();
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDeltaState", []() {
            return std::make_unique<QwenGatedDeltaStateKernel<DeviceType::X86, DataType::FP32>>();
        });
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDeltaOutput", []() {
            return std::make_unique<QwenGatedDeltaOutputKernel<DeviceType::X86, DataType::FP32>>();
        });
        d.registerKernel(DeviceType::X86, DataType::FP32, "QwenGatedDelta", []() {
            return std::make_unique<QwenGatedDeltaKernel<DeviceType::X86, DataType::FP32>>();
        });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86QwenGemmArgmaxKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::BF16, "QwenGemmArgmax", []() {
            return std::make_unique<QwenGemmArgmaxKernel<DeviceType::X86, DataType::BF16>>();
        });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E4M3, "QwenGemmArgmax", []() {
            return std::make_unique<QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E4M3>>();
        });
        KernelDispatcher::instance().registerKernel(DeviceType::X86, DataType::FP8E5M2, "QwenGemmArgmax", []() {
            return std::make_unique<QwenGemmArgmaxKernel<DeviceType::X86, DataType::FP8E5M2>>();
        });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86QwenRmsNormKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        auto& dispatcher = KernelDispatcher::instance();
        dispatcher.registerKernel(DeviceType::X86, DataType::FP32, "QwenRmsNorm", []() {
            return std::make_unique<QwenRmsNormKernel<DeviceType::X86, DataType::FP32>>();
        });
        dispatcher.registerKernel(DeviceType::X86, DataType::BF16, "QwenRmsNorm", []() {
            return std::make_unique<QwenRmsNormKernel<DeviceType::X86, DataType::BF16>>();
        });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ReduceMeanKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReduceMean",
            []() { return std::make_unique<ReduceMeanKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ReduceSumKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReduceSum",
            []() { return std::make_unique<ReduceSumKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ReluKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "ReLU",
            []() { return std::make_unique<ReluKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "ReLU",
            []() { return std::make_unique<ReluKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ReshapeKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Reshape",
            []() { return std::make_unique<ReshapeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Reshape",
            []() { return std::make_unique<ReshapeKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86ResizeKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::INT8, "Resize",
            []() { return std::make_unique<ResizeKernel<DeviceType::X86, DataType::INT8>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SigmoidKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sigmoid",
            []() { return std::make_unique<SigmoidKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Sigmoid",
            []() { return std::make_unique<SigmoidKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SiluKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "SiLU",
            []() { return std::make_unique<SiluKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SliceKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Slice",
            []() { return std::make_unique<SliceKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Slice",
            []() { return std::make_unique<SliceKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SoftmaxKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Softmax",
            []() { return std::make_unique<SoftmaxKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SplitKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Split",
            []() { return std::make_unique<SplitKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Split",
            []() { return std::make_unique<SplitKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Split",
            []() { return std::make_unique<SplitKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SqrtKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sqrt",
            []() { return std::make_unique<SqrtKernel<DeviceType::X86, DataType::FP32>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86SubKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E4M3, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP8E4M3>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP8E5M2, "Sub",
            []() { return std::make_unique<SubKernel<DeviceType::X86, DataType::FP8E5M2>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86TransposeKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::FP16>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Transpose",
            []() { return std::make_unique<TransposeKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureX86YoloDecodeKernelsRegistered() {
#ifdef FEATHER_WITH_X86
    if (!HostSupportsX86Kernels()) return;
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP32, "YoloDecode",
            []() { return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::FP32>>(); });
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::FP16, "YoloDecode",
            []() { return std::make_unique<YoloDecodeKernel<DeviceType::X86, DataType::FP16>>(); });
        return true;
    }();
    (void)registered;
#endif
}

void EnsureQwenDepthwiseConvStateKernelsRegistered() {
    EnsureX86QwenDepthwiseConvStateKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
    EnsureCudaQwenDepthwiseConvStateKernelsRegistered();
#endif
}
}  // namespace kernel
}  // namespace feather
