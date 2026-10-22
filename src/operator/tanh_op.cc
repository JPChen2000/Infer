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

#include "src/operator/tanh_op.h"

#include <utility>

#include "core/operator_registry.h"
#include "src/operator/elementwise_utils.h"
#include "src/operator/elementwise_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildTanhOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return nullptr;
    }
    UnaryParam param{};
    param.input = tensors[node.inputs[0]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<TanhOp>(node.name.empty() ? "tanh" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureTanhKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Tanh", {param.input, op->outputs().front()}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_tanh_op_registered = []() {
    OperatorRegistry::instance().Register("Tanh", BuildTanhOp);
    return true;
}();

}  // namespace

void EnsureTanhOperatorRegistered() { (void)g_tanh_op_registered; }

TanhOp::TanhOp(std::string name, const UnaryParam& param) : OpBase(std::move(name), "Tanh"), param_(param) { SyncIO(); }

void TanhOp::SyncIO() {
    SetInputs({param_.input});
    SetOutputs({param_.out});
}

int32_t TanhOp::CheckShape() const { return elementwise_detail::CheckUnaryParam(param_); }

int32_t TanhOp::InferOutputShapes() {
    const auto status = elementwise_detail::InferUnaryOutput(&param_);
    if (status == 0) {
        SyncIO();
    }
    return status;
}

void TanhOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t TanhOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
