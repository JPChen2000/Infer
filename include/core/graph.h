#ifndef FEATHER_CORE_GRAPH_H
#define FEATHER_CORE_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/operator.h"
#include "util/thread_pool_nv.h"

namespace feather {

struct RuntimeNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<OpBase> owner;
    std::unique_ptr<KernelBase> kernel;
    std::vector<size_t> predecessors;
    std::vector<size_t> successors;
    size_t pending_dependencies{};

    std::string ProfileLabel() const;
    int32_t Run();
};

class RuntimeGraph {
   public:
    RuntimeGraph() = default;

    int32_t load_from_buffer(const char* buffer, size_t size);
    int32_t load_from_path(const std::string& path);

    int32_t Check() const;
    int32_t Run();
    int32_t SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor);
    std::shared_ptr<Tensor> GetTensor(const std::string& name) const;
    const RuntimeNode* GetNode(const std::string& name) const;

    void Clear();
    void AddNode(RuntimeNode node);
    int32_t Finalize();
    size_t NodeSize() const;
    size_t WorkerCount() const;

   private:
    int32_t BuildDependencies();
    void ResetPendingDependencies();
    int32_t RunSerial();

    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::vector<RuntimeNode> nodes_;
    std::unordered_map<std::string, size_t> node_index_by_name_;
    size_t worker_count_{1};
    std::unique_ptr<ThreadPoolNv> thread_pool_;
};

}  // namespace feather

#endif  // FEATHER_CORE_GRAPH_H
