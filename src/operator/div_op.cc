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

#include "src/operator/div_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"
#include "src/operator/tensor_op_utils.h"
#include "src/kernel/yolo_selection.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildDivOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }
    BinaryParam param{};
    param.lhs = tensors[node.inputs[0]];
    param.rhs = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<DivOp>(node.name.empty() ? "div" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureDivKernelsRegistered();
    kernel::EnsureYoloSelectionKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Div", {param.lhs, param.rhs, op->outputs().front()});
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_div_op_registered = []() {
    OperatorRegistry::instance().Register("Div", BuildDivOp);
    return true;
}();

}  // namespace

void EnsureDivOperatorRegistered() { (void)g_div_op_registered; }

DivOp::DivOp(std::string name, const BinaryParam& param) : OpBase(std::move(name), "Div"), param_(param) { SyncIO(); }

void DivOp::SyncIO() {
    SetInputs({param_.lhs, param_.rhs});
    SetOutputs({param_.out});
}

int32_t DivOp::CheckShape() const {
    if (param_.lhs && param_.rhs &&
        (param_.lhs->data_type() == DataType::INT64 || param_.rhs->data_type() == DataType::INT64) &&
        param_.lhs->data_type() != param_.rhs->data_type()) return -1;
    return elementwise_detail::CheckBinaryParam(param_);
}

int32_t DivOp::InferOutputShapes() {
    if (param_.lhs && param_.lhs->data_type() == DataType::INT64) {
        if (CheckShape() != 0) return -1;
        std::vector<int64_t> shape;
        if (!YoloSelectionBroadcast(param_.lhs->dims().data(), param_.rhs->dims().data(), &shape)) return -1;
        param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, DataType::INT64);
        if (!param_.out) return -1;
        SyncIO();
        return 0;
    }
    const auto status = elementwise_detail::InferBinaryOutput(&param_);
    if (status == 0) {
        SyncIO();
    }
    return status;
}

void DivOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t DivOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
