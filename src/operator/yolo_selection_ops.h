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

#ifndef FEATHER_OPERATOR_YOLO_SELECTION_OPS_H
#define FEATHER_OPERATOR_YOLO_SELECTION_OPS_H

#include <memory>
#include <string>
#include <vector>

#include "core/operator.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

enum class YoloSelectionKind { kTopK, kGatherElements, kReduceMax, kMod };

struct YoloSelectionParam : ParamBase {
    YoloSelectionKind kind{YoloSelectionKind::kTopK};
    std::shared_ptr<Tensor> data;
    // TopK K, GatherElements indices, ReduceMax axes, or Mod divisor.
    std::shared_ptr<Tensor> control;
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> indices_out;
    int64_t axis{-1};
    bool largest{true};
    bool sorted{true};
    bool keepdims{true};
    bool noop_with_empty_axes{false};
    bool fmod{false};
    std::vector<int64_t> axes;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(data);
        visit(control);
        visit(out);
        visit(indices_out);
    }
};

bool YoloSelectionAxes(const YoloSelectionParam& param, std::vector<int64_t>* axes);
bool YoloSelectionBroadcast(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
                            std::vector<int64_t>* output);
bool IsYoloSelectionInteger(DataType dtype);
bool IsYoloSelectionNumeric(DataType dtype);

class YoloSelectionOp : public OpBase {
   public:
    YoloSelectionOp(std::string name, std::string type, YoloSelectionParam param);
    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override;
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    const KernelBase* kernel() const override { return kernel_.get(); }
    int32_t Run() override;
   private:
    void SyncIO();
    YoloSelectionParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

void EnsureYoloSelectionOperatorsRegistered();

}  // namespace operators
}  // namespace feather
#endif
