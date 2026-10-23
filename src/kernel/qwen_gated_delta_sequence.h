// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FEATHER_KERNEL_QWEN_GATED_DELTA_SEQUENCE_H
#define FEATHER_KERNEL_QWEN_GATED_DELTA_SEQUENCE_H

#include "core/kernel.h"
#include "src/operator/qwen_gated_delta_sequence_param.h"

namespace feather {
namespace kernel {
void EnsureCommonQwenGatedDeltaSequenceKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
void EnsureCudaQwenGatedDeltaSequenceKernelsRegistered();
#endif
void EnsureQwenGatedDeltaSequenceKernelsRegistered();

template <DeviceType dev, DataType dtype>
class QwenGatedDeltaSequenceKernel : public KernelBase {
 public:
    int32_t compute() override;
};
} // namespace kernel
} // namespace feather
#endif
