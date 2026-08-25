#ifndef FEATHER_CORE_GRAPH_LOWERING_H
#define FEATHER_CORE_GRAPH_LOWERING_H

#include "core/graph.h"
#include "core/static_graph.h"

namespace feather {

class GraphLowering {
   public:
    int32_t Lower(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const;
    Status LowerStatus(StaticGraph& static_graph, RuntimeGraph* runtime_graph) const;
};

}  // namespace feather

#endif  // FEATHER_CORE_GRAPH_LOWERING_H
