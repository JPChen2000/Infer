#ifndef FEATHER_PASS_RESIZE_CONCAT_FUSION_PASS_H
#define FEATHER_PASS_RESIZE_CONCAT_FUSION_PASS_H

#include "pass/graph_pass.h"

namespace feather {

class ResizeConcatFusionPass : public GraphPass {
   public:
    const std::string& name() const override;
    int32_t Run(StaticGraph* graph) override;
};

}  // namespace feather

#endif  // FEATHER_PASS_RESIZE_CONCAT_FUSION_PASS_H
