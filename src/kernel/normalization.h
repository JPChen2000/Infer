// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FEATHER_KERNEL_NORMALIZATION_H
#define FEATHER_KERNEL_NORMALIZATION_H

#include "core/kernel.h"
#include "src/operator/normalization_param.h"

namespace feather {
namespace kernel {

template <DeviceType dev, DataType dtype>
class NormalizationKernel : public KernelBase {
 public:
    Status Supports() const override {
        const auto* param = ParamAs<operators::NormalizationParam>();
        operators::normalization_detail::Shape shape;
        if (!param || !operators::normalization_detail::Describe(*param, &shape) ||
            param->input->data_type() != dtype ||
            (dev == DeviceType::CUDA && param->input->dims().size() > 8))
            return Status::Error(StatusCode::kUnsupported, "unsupported normalization shape, dtype or attributes");
        return Status::Ok();
    }
    int32_t compute() override;
};

void EnsureCommonNormalizationKernelsRegistered();
#ifdef FEATHER_WITH_CUDA
void EnsureCudaNormalizationKernelsRegistered();
#endif
void EnsureNormalizationKernelsRegistered();

}  // namespace kernel
}  // namespace feather
#endif
