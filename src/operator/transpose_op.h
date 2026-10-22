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

#ifndef FEATHER_OPERATOR_TRANSPOSE_OP_H
#define FEATHER_OPERATOR_TRANSPOSE_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/transpose.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class TransposeOp : public OpBase {
   public:
    TransposeOp();
    explicit TransposeOp(const TransposeParam& param);
    TransposeOp(std::string name, const TransposeParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override;
    void RefreshKernelParams() override;

   private:
    void SyncIO();

    TransposeParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_TRANSPOSE_OP_H
