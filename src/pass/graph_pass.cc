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

#include "pass/graph_pass.h"
#include <unordered_set>

#include "pass/dead_node_elimination_pass.h"
#include "pass/identity_elimination_pass.h"
#include "pass/matmul_add_fusion_pass.h"
#include "pass/qwen_gated_delta_fusion_pass.h"
#include "pass/qwen_depthwise_conv_fusion_pass.h"
#include "pass/qwen_matmul_add_fusion_pass.h"
#include "pass/qwen_gemm_argmax_fusion_pass.h"
#include "pass/qwen_rms_norm_fusion_pass.h"
#include "pass/qwen_state_output_alias_pass.h"
#include "pass/no_op_elimination_pass.h"
#include "pass/resize_concat_fusion_pass.h"
#include "pass/reshape_chain_elimination_pass.h"
#include "pass/sigmoid_mul_fusion_pass.h"
#include "pass/yolo_decode_fusion_pass.h"
#include "core/static_graph.h"
#include "util/logger.h"

namespace feather {

void PassManager::AddPass(std::unique_ptr<GraphPass> pass) {
    if (pass != nullptr) {
        passes_.push_back(std::move(pass));
    }
}

int32_t PassManager::Run(StaticGraph* graph) const { return RunStatus(graph).ok() ? 0 : -1; }

Status PassManager::RunStatus(StaticGraph* graph) const {
    if (!graph) return Status::Error(StatusCode::kInvalidArgument, "null graph for pass pipeline");
    auto status = graph->ValidateStructureStatus();
    if (!status.ok()) return status;
    std::unordered_set<std::string> completed;
    for (const auto& pass : passes_) {
        if (!pass || !pass->SupportsDevice(graph->KernelDevice())) continue;
        for (const auto& requirement : pass->Requires()) {
            if (!completed.count(requirement)) return Status::Error(StatusCode::kBuildFailed,
                "pass=" + pass->name() + " requires prior pass=" + requirement);
        }
        try {
            if (pass->Run(graph) != 0) return Status::Error(StatusCode::kBuildFailed, "pass failed: " + pass->name());
        } catch (const std::exception& error) {
            return Status::Error(StatusCode::kBuildFailed, "pass=" + pass->name() + ": " + error.what());
        }
        status = graph->ValidateStructureStatus();
        if (!status.ok()) return status.WithContext("after pass=" + pass->name());
        completed.insert(pass->name());
    }
    return Status::Ok();
}

std::shared_ptr<PassManager> CreateDefaultPassManager() {
    auto manager = std::make_shared<PassManager>();
    manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    manager->AddPass(std::make_unique<IdentityEliminationPass>());
    manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    manager->AddPass(std::make_unique<NoOpEliminationPass>());
    manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    return manager;
}

std::shared_ptr<PassManager> CreateQwenPassManager() {
    auto manager = std::make_shared<PassManager>();
    manager->AddPass(std::make_unique<SigmoidMulFusionPass>());
    manager->AddPass(std::make_unique<MatMulAddFusionPass>());
    manager->AddPass(std::make_unique<QwenMatMulAddFusionPass>());
    manager->AddPass(std::make_unique<QwenGemmArgmaxFusionPass>());
    manager->AddPass(std::make_unique<QwenGatedDeltaFusionPass>());
    manager->AddPass(std::make_unique<QwenDepthwiseConvFusionPass>());
    manager->AddPass(std::make_unique<QwenRmsNormFusionPass>());
    manager->AddPass(std::make_unique<IdentityEliminationPass>());
    manager->AddPass(std::make_unique<ReshapeChainEliminationPass>());
    manager->AddPass(std::make_unique<NoOpEliminationPass>());
    manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    manager->AddPass(std::make_unique<QwenStateOutputAliasPass>());
    return manager;
}

std::shared_ptr<PassManager> CreateYoloPassManager() {
    auto manager = CreateDefaultPassManager();
    manager->AddPass(std::make_unique<ResizeConcatFusionPass>());
    manager->AddPass(std::make_unique<YoloDecodeFusionPass>());
    manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    return manager;
}

}  // namespace feather
