#include "pass/graph_pass.h"

#include "pass/dead_node_elimination_pass.h"
#include "pass/identity_elimination_pass.h"
#include "pass/matmul_add_fusion_pass.h"
#include "pass/no_op_elimination_pass.h"
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

int32_t PassManager::Run(StaticGraph* graph) const {
    if (graph == nullptr) {
        return -1;
    }
    LOG_INFO("[pass] begin pass_count=%zu active_nodes=%zu", passes_.size(), graph->NodeSize());
    size_t pass_index = 0;
    for (const auto& pass : passes_) {
        ++pass_index;
        if (pass == nullptr) {
            LOG_WARN("[pass] skip index=%zu reason=null", pass_index);
            continue;
        }
        [[maybe_unused]] const auto before_nodes = graph->NodeSize();
        LOG_INFO("[pass] start index=%zu name=%s active_nodes=%zu", pass_index, pass->name().c_str(), before_nodes);
        const auto status = pass->Run(graph);
        [[maybe_unused]] const auto after_nodes = graph->NodeSize();
        if (status != 0) {
            LOG_ERROR("[pass] fail index=%zu name=%s status=%d active_nodes=%zu->%zu", pass_index,
                      pass->name().c_str(), status, before_nodes, after_nodes);
            return status;
        }
        LOG_INFO("[pass] done index=%zu name=%s status=%d active_nodes=%zu->%zu", pass_index,
                 pass->name().c_str(), status, before_nodes, after_nodes);
    }
    LOG_INFO("[pass] end pass_count=%zu active_nodes=%zu", passes_.size(), graph->NodeSize());
    return 0;
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

std::shared_ptr<PassManager> CreateYoloPassManager() {
    auto manager = CreateDefaultPassManager();
    manager->AddPass(std::make_unique<YoloDecodeFusionPass>());
    manager->AddPass(std::make_unique<DeadNodeEliminationPass>());
    return manager;
}

}  // namespace feather
