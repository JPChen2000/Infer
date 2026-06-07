#include "core/graph_pass.h"

#include "core/static_graph.h"

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
    for (const auto& pass : passes_) {
        if (pass == nullptr) {
            return -1;
        }
        const auto status = pass->Run(graph);
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

}  // namespace feather
