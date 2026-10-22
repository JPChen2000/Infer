// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_CORE_GRAPH_H
#define FEATHER_CORE_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/status.h"
#include "core/backend.h"
#include "core/kernel.h"
#include "util/thread_pool_nv.h"

namespace feather {

#ifdef FEATHER_WITH_CUDA
namespace kernel { namespace cuda_detail { class CudaExecutionContext; } }
#endif

using ValueId = uint32_t;
constexpr ValueId kInvalidValueId = static_cast<ValueId>(-1);

struct RuntimeProfileSummary {
    std::string node_name;
    std::string op_type;
    int64_t call_count{0};
    double total_ms{0.0};
    double avg_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

enum class RuntimeThreadMode {
    kSerialGraph,
    kParallelGraph,
};

struct RuntimeNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<ValueId> input_ids;
    std::vector<ValueId> output_ids;
    std::unique_ptr<KernelBase> kernel;
    DeviceType kernel_device{DeviceType::UNKNOWN};
    std::vector<size_t> predecessors;
    std::vector<size_t> successors;
    size_t pending_dependencies{};

    std::string ProfileLabel() const;
    int32_t Run();
};

class RuntimeGraph {
   public:
    RuntimeGraph();
    explicit RuntimeGraph(std::unique_ptr<BackendContext> backend);
    ~RuntimeGraph();
#ifdef FEATHER_WITH_CUDA
    const std::shared_ptr<kernel::cuda_detail::CudaExecutionContext>& CudaContext() const { return backend_->CudaContext(); }
#endif

    bool UsesMemoryPlan() const { return uses_memory_plan_; }
    BackendContext& Backend() const { return *backend_; }
    const RuntimeNode* GetProducerNode(const std::string& value_name) const;
    int32_t Check() const;
    Status CheckStatus() const;
    int32_t Run();
    Status RunStatus();
    // After Finalize, only compatible external inputs can be rebound. Their
    // Tensor identity stays stable because compiled kernels retain that object.
    int32_t SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor);
    std::shared_ptr<Tensor> GetTensor(const std::string& name) const;
    const RuntimeNode* GetNode(const std::string& name) const;
    bool ProfilingEnabled() const { return profiling_enabled_; }
    const std::vector<RuntimeProfileSummary>& ProfileSummaries() const { return profile_summaries_; }
    void SetProfilingEnabled(bool enabled);
    void RecordNodeProfile(const std::string& node_name, const std::string& op_type, double elapsed_ms);
    RuntimeThreadMode ThreadMode() const { return thread_mode_; }
    void SetThreadMode(RuntimeThreadMode mode);
    size_t ThreadCount() const { return configured_thread_count_; }
    void SetThreadCount(size_t count);
    void SetOutputNames(std::vector<std::string> output_names);

    void Clear();
    void AddNode(RuntimeNode node);
    ValueId GetOrCreateValueId(const std::string& name);
    ValueId GetValueId(const std::string& name) const;
    int32_t Finalize();
    Status FinalizeStatus();
    size_t NodeSize() const;
    size_t WorkerCount() const;

   private:
    friend class CompiledGraph;
    int32_t BuildDependencies();
    void ResetPendingDependencies();
    int32_t RunSerial();
    int32_t RunNode(size_t index);
    int32_t PrepareNodeForRun(const RuntimeNode& node);
    int32_t FinalizeNodeRun(const RuntimeNode& node, int32_t status);
    void ResetRemainingUses();
    void ReleaseUnusedInputs(const RuntimeNode& node);
    bool ShouldKeepTensorDevice(const std::string& value_name, const std::shared_ptr<Tensor>& tensor) const;
    void RefreshOutputTensorPointers();

    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::vector<RuntimeNode> nodes_;
    std::unordered_map<std::string, size_t> node_index_by_name_;
    std::unordered_map<std::string, ValueId> value_id_by_name_;
    std::vector<std::string> value_name_by_id_;
    std::unordered_map<ValueId, size_t> remaining_uses_;
    std::unordered_set<std::string> output_names_;
    std::unordered_set<const Tensor*> output_tensor_ptrs_;
    size_t worker_count_{1};
    std::unique_ptr<ThreadPoolNv> thread_pool_;
    bool profiling_enabled_{false};
    std::vector<RuntimeProfileSummary> profile_summaries_;
    std::unordered_map<std::string, size_t> profile_index_by_node_;
    mutable std::mutex profile_mutex_;
    RuntimeThreadMode thread_mode_{RuntimeThreadMode::kParallelGraph};
    size_t configured_thread_count_{1};
    bool finalized_{false};
    bool uses_memory_plan_{false};
    bool structure_locked_{false};
    struct TensorContract {
        std::string name;
        std::shared_ptr<Tensor> tensor;
        std::shared_ptr<const Tensor> descriptor;
        size_t bytes;
        bool fixed_quantization;
    };
    std::vector<TensorContract> contracts_;
    std::unique_ptr<BackendContext> backend_;
    Status execution_status_;
    mutable std::mutex error_mutex_;
    mutable std::mutex uses_mutex_;
#ifdef FEATHER_WITH_CUDA

#endif
};

}  // namespace feather

#endif  // FEATHER_CORE_GRAPH_H
