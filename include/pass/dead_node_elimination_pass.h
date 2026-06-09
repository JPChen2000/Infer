#ifndef FEATHER_PASS_DEAD_NODE_ELIMINATION_PASS_H
#define FEATHER_PASS_DEAD_NODE_ELIMINATION_PASS_H

#include "pass/graph_pass.h"

namespace feather {

class DeadNodeEliminationPass : public GraphPass {
   public:
    const std::string& name() const override;
    int32_t Run(StaticGraph* graph) override;
};

}  // namespace feather

#endif  // FEATHER_PASS_DEAD_NODE_ELIMINATION_PASS_H
