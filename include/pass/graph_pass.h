#ifndef FEATHER_PASS_GRAPH_PASS_H
#define FEATHER_PASS_GRAPH_PASS_H

#include <memory>
#include <string>
#include <vector>
#include "core/status.h"
#include "util/types.h"

namespace feather {

class StaticGraph;

class GraphPass {
   public:
    virtual ~GraphPass() = default;
    virtual const std::string& name() const = 0;
    virtual int32_t Run(StaticGraph* graph) = 0;
    virtual std::vector<std::string> Requires() const { return {}; }
    virtual bool SupportsDevice(DeviceType) const { return true; }
};

class PassManager {
   public:
    void AddPass(std::unique_ptr<GraphPass> pass);
    int32_t Run(StaticGraph* graph) const;
    Status RunStatus(StaticGraph* graph) const;
    size_t PassCount() const { return passes_.size(); }

   private:
    std::vector<std::unique_ptr<GraphPass>> passes_;
};

std::shared_ptr<PassManager> CreateDefaultPassManager();
std::shared_ptr<PassManager> CreateYoloPassManager();
std::shared_ptr<PassManager> CreateQwenPassManager();

}  // namespace feather

#endif  // FEATHER_PASS_GRAPH_PASS_H
