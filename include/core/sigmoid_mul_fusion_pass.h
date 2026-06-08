#ifndef FEATHER_CORE_SIGMOID_MUL_FUSION_PASS_H
#define FEATHER_CORE_SIGMOID_MUL_FUSION_PASS_H

#include <string>

#include "core/graph_pass.h"

namespace feather {

class SigmoidMulFusionPass : public GraphPass {
   public:
    const std::string& name() const override;
    int32_t Run(StaticGraph* graph) override;
};

}  // namespace feather

#endif  // FEATHER_CORE_SIGMOID_MUL_FUSION_PASS_H
