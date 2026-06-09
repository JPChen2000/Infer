#ifndef FEATHER_PASS_GRAPH_PASS_H
#define FEATHER_PASS_GRAPH_PASS_H

#include <memory>
#include <string>
#include <vector>

namespace feather {

class StaticGraph;

class GraphPass {
   public:
    virtual ~GraphPass() = default;
    virtual const std::string& name() const = 0;
    virtual int32_t Run(StaticGraph* graph) = 0;
};

class PassManager {
   public:
    void AddPass(std::unique_ptr<GraphPass> pass);
    int32_t Run(StaticGraph* graph) const;
    size_t PassCount() const { return passes_.size(); }

   private:
    std::vector<std::unique_ptr<GraphPass>> passes_;
};

std::shared_ptr<PassManager> CreateDefaultPassManager();
std::shared_ptr<PassManager> CreateYoloPassManager();

}  // namespace feather

#endif  // FEATHER_PASS_GRAPH_PASS_H
