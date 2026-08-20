#ifndef FEATHER_PASS_QWEN_MATMUL_ADD_FUSION_PASS_H
#define FEATHER_PASS_QWEN_MATMUL_ADD_FUSION_PASS_H

#include "pass/graph_pass.h"

namespace feather {

class QwenMatMulAddFusionPass : public GraphPass {
   public:
    const std::string& name() const override;
    int32_t Run(StaticGraph* graph) override;
};

}  // namespace feather

#endif  // FEATHER_PASS_QWEN_MATMUL_ADD_FUSION_PASS_H
