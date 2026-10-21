#ifndef FEATHER_CORE_STATEFUL_SESSION_H
#define FEATHER_CORE_STATEFUL_SESSION_H

#include "core/graph.h"

namespace feather {
struct StateBinding {
    std::string input_name;
    std::string output_name;
    int cache_axis{-1};
};

Status ApplyStateBinding(RuntimeGraph& graph, const StateBinding& binding, DeviceType device);

// Owns state bindings, while storage and backend resources belong to the graph.
class StatefulSession {
 public:
    explicit StatefulSession(RuntimeGraph& graph, DeviceType device = DeviceType::COMMON) : graph_(graph), device_(device) {}
    Status Bind(StateBinding binding);
    Status Advance();
    Status Reset();
 private:
    RuntimeGraph& graph_;
    DeviceType device_;
    std::vector<StateBinding> bindings_;
};
}  // namespace feather
#endif
