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

#ifndef FEATHER_OPERATOR_MATMUL_OP_H
#define FEATHER_OPERATOR_MATMUL_OP_H

#include <memory>
#include <string>
#include <vector>

#include "core/operator.h"
#include "src/kernel/matmul.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class MatMulOp : public OpBase {
   public:
    MatMulOp();
    explicit MatMulOp(const MatMulParam& param);
    MatMulOp(std::string name, const MatMulParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override;
    int shape_inference_count() const { return shape_inference_count_; }

   private:
    void SyncIO();
    bool ShapeCacheMatches() const;
    void UpdateShapeCache();

    MatMulParam param_;
    std::unique_ptr<KernelBase> kernel_;
    int shape_inference_count_{0};
    bool shape_cache_valid_{false};
    std::vector<int64_t> cached_a_dims_;
    std::vector<int64_t> cached_b_dims_;
    std::vector<int64_t> cached_output_dims_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_MATMUL_OP_H
