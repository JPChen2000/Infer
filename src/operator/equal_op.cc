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

#include "src/operator/equal_op.h"

#include "core/operator_registry.h"
#include "src/operator/tensor_op_utils.h"

namespace feather {
namespace operators {

namespace {

std::shared_ptr<OpBase> BuildEqualOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    EqualParam param{};
    param.lhs = tensors[node.inputs[0]];
    param.rhs = tensors[node.inputs[1]];
    param.out = tensors[node.outputs[0]];

    auto op = std::make_shared<EqualOp>(node.name.empty() ? "Equal" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureEqualKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Equal", {param.lhs, param.rhs}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_equal_op_registered = []() {
    OperatorRegistry::instance().Register("Equal", BuildEqualOp);
    return true;
}();

}  // namespace

void EnsureEqualOperatorRegistered() { (void)g_equal_op_registered; }

EqualOp::EqualOp(std::string name, const EqualParam& param) : OpBase(std::move(name), "Equal"), param_(param) {
    SyncIO();
}

void EqualOp::SyncIO() {
    SetInputs({param_.lhs, param_.rhs});
    SetOutputs({param_.out});
}

int32_t EqualOp::CheckShape() const {
    if (param_.lhs == nullptr || param_.rhs == nullptr || param_.out == nullptr ||
        param_.lhs->data_type() != param_.rhs->data_type()) {
        return -1;
    }
    std::vector<int64_t> shape;
    return tensor_op_detail::InferBroadcastShape(param_.lhs->dims().data(), param_.rhs->dims().data(), &shape) ? 0 : -1;
}

int32_t EqualOp::InferOutputShapes() {
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> shape;
    if (!tensor_op_detail::InferBroadcastShape(param_.lhs->dims().data(), param_.rhs->dims().data(), &shape)) {
        return -1;
    }
    param_.out = tensor_op_detail::AllocateOutput(param_.out, shape, DataType::BOOL);
    if (param_.out == nullptr) {
        return -1;
    }
    SyncIO();
    return 0;
}

void EqualOp::AttachKernel(std::unique_ptr<KernelBase> kernel) {
    kernel_ = std::move(kernel);
    if (kernel_ != nullptr) {
        kernel_->SetParamOwner(std::make_shared<std::decay_t<decltype(param_)>>(param_));
    }
}

int32_t EqualOp::Run() {
    if (InferOutputShapes() != 0 || kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
