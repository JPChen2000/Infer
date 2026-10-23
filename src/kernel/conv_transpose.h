// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_KERNEL_CONV_TRANSPOSE_H
#define FEATHER_KERNEL_CONV_TRANSPOSE_H

#include "core/kernel.h"
#include "src/operator/conv_transpose_param.h"

namespace feather {
namespace kernel {
void EnsureCommonConvTransposeKernelsRegistered();

class CommonConvTransposeKernel final : public KernelBase {
 public:
    int32_t compute() override;
    Status Supports() const override {
        auto* param = ParamAs<operators::ConvTransposeParam>();
        std::vector<int64_t> shape;
        return param && operators::ConvTransposeOutputShape(*param, &shape)
                   ? Status::Ok()
                   : Status::Error(StatusCode::kUnsupported, "ConvTranspose requires valid NCHW FP32 tensors");
    }
};
}  // namespace kernel
}  // namespace feather
#endif
