#include "core/graph.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

#include "util/threading.h"
#include "util/timer.h"
#include "util/types.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {

namespace {

int32_t SynchronizeRuntimeNodeForProfiling(const RuntimeNode& node) {
#ifdef FEATHER_WITH_CUDA
    if (node.kernel_device == DeviceType::CUDA) {
        return kernel::cuda_detail::SynchronizeInferenceStream();
    }
#endif
    (void)node;
    return 0;
}

#ifdef FEATHER_WITH_CUDA
size_t TensorByteSizeForNode(const Tensor& tensor, const RuntimeNode& node) {
    auto dtype = tensor.data_type();
    if (dtype == DataType::UNKNOWN && node.kernel != nullptr) {
        dtype = node.kernel->data_type();
    }
    const auto element_bytes = DataTypeBytes(dtype);
    if (element_bytes == 0) {
        return tensor.memory_size();
    }
    return static_cast<size_t>(std::max<int64_t>(0, tensor.numel())) * element_bytes;
}
#endif

}  // namespace

RuntimeGraph::RuntimeGraph()
    : thread_mode_(RuntimeThreadMode::kParallelGraph),
      configured_thread_count_(DefaultThreadCount()) {}

std::string RuntimeNode::ProfileLabel() const {
    return "RuntimeNode::" + name + "[" + op_type + "]";
}

int32_t RuntimeNode::Run() {
    if (owner == nullptr || kernel == nullptr) {
        return -1;
    }
    AutoTimer timer(ProfileLabel());
    return kernel->compute();
}

int32_t RuntimeGraph::SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (name.empty() || tensor == nullptr) {
        return -1;
    }
    tensors_[name] = std::move(tensor);
    RefreshOutputTensorPointers();
    return 0;
}

std::shared_ptr<Tensor> RuntimeGraph::GetTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}

const RuntimeNode* RuntimeGraph::GetNode(const std::string& name) const {
    auto it = node_index_by_name_.find(name);
    if (it == node_index_by_name_.end()) {
        return nullptr;
    }
    return &nodes_[it->second];
}

void RuntimeGraph::SetProfilingEnabled(bool enabled) {
    profiling_enabled_ = enabled;
    if (!profiling_enabled_) {
        profile_summaries_.clear();
        profile_index_by_node_.clear();
    } else {
        // Qwen decode has more than a thousand runtime nodes. Reserve once so
        // turning profiling on does not allocate on the per-node hot path.
        profile_summaries_.reserve(nodes_.size());
        profile_index_by_node_.reserve(nodes_.size());
    }
}

void RuntimeGraph::RecordNodeProfile(const std::string& node_name, const std::string& op_type, double elapsed_ms) {
    if (!profiling_enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(profile_mutex_);
    const auto index = profile_index_by_node_.find(node_name);
    if (index == profile_index_by_node_.end()) {
        profile_index_by_node_.emplace(node_name, profile_summaries_.size());
        profile_summaries_.push_back(
            RuntimeProfileSummary{node_name, op_type, 1, elapsed_ms, elapsed_ms, elapsed_ms, elapsed_ms});
        return;
    }

    auto& summary = profile_summaries_[index->second];
    summary.call_count += 1;
    summary.total_ms += elapsed_ms;
    summary.avg_ms = summary.total_ms / static_cast<double>(summary.call_count);
    summary.min_ms = std::min(summary.min_ms, elapsed_ms);
    summary.max_ms = std::max(summary.max_ms, elapsed_ms);
}

void RuntimeGraph::SetThreadCount(size_t count) {
    configured_thread_count_ = std::max<size_t>(1, count);
}

void RuntimeGraph::SetOutputNames(std::vector<std::string> output_names) {
    output_names_.clear();
    for (auto& name : output_names) {
        output_names_.insert(std::move(name));
    }
    RefreshOutputTensorPointers();
}

void RuntimeGraph::Clear() {
    nodes_.clear();
    tensors_.clear();
    node_index_by_name_.clear();
    value_id_by_name_.clear();
    value_name_by_id_.clear();
    remaining_uses_.clear();
    output_names_.clear();
    output_tensor_ptrs_.clear();
    thread_pool_.reset();
    worker_count_ = 1;
    profiling_enabled_ = false;
    profile_summaries_.clear();
    profile_index_by_node_.clear();
}

int32_t RuntimeGraph::Check() const {
    return CheckStatus().ok() ? 0 : -1;
}

Status RuntimeGraph::CheckStatus() const {
    for (const auto& node : nodes_) {
        if (node.owner == nullptr || node.kernel == nullptr) {
            return Status::Error(StatusCode::kBuildFailed, "runtime node has no kernel: " + node.name);
        }
    }
    return Status::Ok();
}

int32_t RuntimeGraph::Run() {
    auto status = Check();
    if (status != 0) {
        return status;
    }
    ResetPendingDependencies();
#ifdef FEATHER_WITH_CUDA
    ResetRemainingUses();
#endif
    return RunSerial();
}

void RuntimeGraph::AddNode(RuntimeNode node) {
    if (node.input_ids.empty()) {
        node.input_ids.reserve(node.inputs.size());
        for (const auto& name : node.inputs) {
            node.input_ids.push_back(GetOrCreateValueId(name));
        }
    }
    if (node.output_ids.empty()) {
        node.output_ids.reserve(node.outputs.size());
        for (const auto& name : node.outputs) {
            node.output_ids.push_back(GetOrCreateValueId(name));
        }
    }
    node_index_by_name_[node.name] = nodes_.size();
    nodes_.push_back(std::move(node));
}

ValueId RuntimeGraph::GetOrCreateValueId(const std::string& name) {
    auto it = value_id_by_name_.find(name);
    if (it != value_id_by_name_.end()) {
        return it->second;
    }
    const auto id = static_cast<ValueId>(value_name_by_id_.size());
    value_id_by_name_.emplace(name, id);
    value_name_by_id_.push_back(name);
    return id;
}

ValueId RuntimeGraph::GetValueId(const std::string& name) const {
    auto it = value_id_by_name_.find(name);
    return it == value_id_by_name_.end() ? kInvalidValueId : it->second;
}

int32_t RuntimeGraph::Finalize() {
    const auto status = BuildDependencies();
    if (status != 0) {
        thread_pool_.reset();
        worker_count_ = 1;
        return status;
    }

    // Prepare immutable-weight kernels before the first timed inference. This
    // keeps one-time packing/reformatting out of decode latency while leaving
    // kernels that do not need preparation unchanged.
    for (const auto& node : nodes_) {
        if (node.kernel != nullptr && node.kernel->Prepare() != 0) {
            thread_pool_.reset();
            worker_count_ = 1;
            return -1;
        }
    }

    if (thread_mode_ == RuntimeThreadMode::kSerialGraph) {
        thread_pool_.reset();
        worker_count_ = 1;
        return 0;
    }

    worker_count_ = std::max<size_t>(1, configured_thread_count_);
    worker_count_ = std::min(worker_count_, nodes_.size());
    if (worker_count_ <= 1) {
        thread_pool_.reset();
        worker_count_ = 1;
        return 0;
    }

    thread_pool_ = std::make_unique<ThreadPoolNv>(worker_count_);
    return 0;
}

size_t RuntimeGraph::NodeSize() const { return nodes_.size(); }

size_t RuntimeGraph::WorkerCount() const { return worker_count_; }

int32_t RuntimeGraph::BuildDependencies() {
    for (auto& node : nodes_) {
        node.predecessors.clear();
        node.successors.clear();
        node.pending_dependencies = 0;
    }

    std::unordered_map<ValueId, size_t> producer_by_value;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto value_id : nodes_[i].output_ids) {
            producer_by_value[value_id] = i;
        }
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto value_id : nodes_[i].input_ids) {
            auto producer_it = producer_by_value.find(value_id);
            if (producer_it == producer_by_value.end()) {
                continue;
            }
            const size_t pred_index = producer_it->second;
            if (pred_index == i) {
                return -1;
            }
            nodes_[i].predecessors.push_back(pred_index);
            nodes_[pred_index].successors.push_back(i);
        }
    }

    for (auto& node : nodes_) {
        node.pending_dependencies = node.predecessors.size();
    }

    return 0;
}

void RuntimeGraph::ResetPendingDependencies() {
    for (auto& node : nodes_) {
        node.pending_dependencies = node.predecessors.size();
    }
}

void RuntimeGraph::ResetRemainingUses() {
    remaining_uses_.clear();
    for (const auto& node : nodes_) {
        for (const auto value_id : node.input_ids) {
            ++remaining_uses_[value_id];
        }
    }
}

bool RuntimeGraph::ShouldKeepTensorDevice(const std::string& value_name, const std::shared_ptr<Tensor>& tensor) const {
    if (tensor == nullptr) {
        return true;
    }
#ifdef FEATHER_WITH_CUDA
    if (output_names_.find(value_name) != output_names_.end()) {
        return true;
    }
    if (output_tensor_ptrs_.find(tensor.get()) != output_tensor_ptrs_.end()) {
        return true;
    }
    if (kernel::cuda_detail::IsTensorDevicePersistent(tensor.get())) {
        return true;
    }
#else
    (void)value_name;
#endif
    return false;
}

void RuntimeGraph::RefreshOutputTensorPointers() {
    output_tensor_ptrs_.clear();
    for (const auto& output_name : output_names_) {
        const auto it = tensors_.find(output_name);
        if (it != tensors_.end() && it->second != nullptr) {
            output_tensor_ptrs_.insert(it->second.get());
        }
    }
}

void RuntimeGraph::ReleaseUnusedInputs(const RuntimeNode& node) {
#ifdef FEATHER_WITH_CUDA
    for (const auto value_id : node.input_ids) {
        auto it = remaining_uses_.find(value_id);
        if (it == remaining_uses_.end()) {
            continue;
        }
        if (it->second == 0) {
            continue;
        }
        --it->second;
        if (it->second != 0) {
            continue;
        }
        if (value_id >= value_name_by_id_.size()) {
            continue;
        }
        const auto& value_name = value_name_by_id_[value_id];
        auto tensor = GetTensor(value_name);
        if (ShouldKeepTensorDevice(value_name, tensor)) {
            continue;
        }
        if (tensor != nullptr) {
            kernel::cuda_detail::ReleaseTensorDevice(tensor.get());
        }
    }
#else
    (void)node;
#endif
}

int32_t RuntimeGraph::PrepareNodeForRun(const RuntimeNode& node) {
#ifdef FEATHER_WITH_CUDA
    if (node.kernel_device == DeviceType::CUDA) {
        return 0;
    }
    for (const auto& input_name : node.inputs) {
        auto tensor = GetTensor(input_name);
        if (tensor == nullptr || !tensor->IsInitialized()) {
            continue;
        }
        const size_t bytes = TensorByteSizeForNode(*tensor, node);
        if (kernel::cuda_detail::SyncTensorToHostIfNeeded(tensor.get(), bytes, tensor->raw_data()) != 0) {
            return -1;
        }
    }
#else
    (void)node;
#endif
    return 0;
}

int32_t RuntimeGraph::FinalizeNodeRun(const RuntimeNode& node, int32_t status) {
#ifdef FEATHER_WITH_CUDA
    if (status != 0 || node.kernel_device == DeviceType::CUDA) {
        return status;
    }
    for (const auto& output_name : node.outputs) {
        auto tensor = GetTensor(output_name);
        if (tensor != nullptr) {
            kernel::cuda_detail::InvalidateTensorDevice(tensor.get());
        }
    }
#else
    (void)node;
#endif
    return status;
}

int32_t RuntimeGraph::RunNode(size_t index) {
    if (index >= nodes_.size()) {
        return -1;
    }
    auto& node = nodes_[index];
    const auto begin = std::chrono::steady_clock::now();
    auto status = PrepareNodeForRun(node);
    if (status == 0) {
        status = node.Run();
    }
    status = FinalizeNodeRun(node, status);
    if (profiling_enabled_ && status == 0) {
        status = SynchronizeRuntimeNodeForProfiling(node);
    }
    if (status == 0) {
        ReleaseUnusedInputs(node);
    }
    const auto end = std::chrono::steady_clock::now();
    RecordNodeProfile(node.name, node.op_type, std::chrono::duration<double, std::milli>(end - begin).count());
    return status;
}

int32_t RuntimeGraph::RunSerial() {
    if (nodes_.size() < 2 || worker_count_ == 1 || thread_pool_ == nullptr) {
        std::queue<size_t> ready;
        size_t completed = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].pending_dependencies == 0) {
                ready.push(i);
            }
        }

        while (!ready.empty()) {
            const size_t index = ready.front();
            ready.pop();
            const auto status = RunNode(index);
            if (status != 0) {
                return status;
            }
            ++completed;
            for (const auto succ_index : nodes_[index].successors) {
                auto& succ = nodes_[succ_index];
                if (succ.pending_dependencies == 0) {
                    return -1;
                }
                --succ.pending_dependencies;
                if (succ.pending_dependencies == 0) {
                    ready.push(succ_index);
                }
            }
        }

        return completed == nodes_.size() ? 0 : -1;
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> ready;
    size_t completed = 0;
    size_t in_flight = 0;
    bool stop = false;
    int32_t error_status = 0;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].pending_dependencies == 0) {
            ready.push(i);
        }
    }
    if (ready.empty() && !nodes_.empty()) {
        return -1;
    }

    auto run_node = [&](int /*tid*/, size_t index) -> int {
        const auto status = RunNode(index);
        {
            std::lock_guard<std::mutex> lock(mutex);
            --in_flight;
            if (status != 0) {
                error_status = status;
                stop = true;
            } else {
                ++completed;
                for (const auto succ_index : nodes_[index].successors) {
                    auto& succ = nodes_[succ_index];
                    if (succ.pending_dependencies == 0) {
                        error_status = -1;
                        stop = true;
                        break;
                    }
                    --succ.pending_dependencies;
                    if (succ.pending_dependencies == 0) {
                        ready.push(succ_index);
                    }
                }
                if (completed == nodes_.size()) {
                    stop = true;
                }
            }
        }
        cv.notify_all();
        return status;
    };

    while (completed < nodes_.size()) {
        std::vector<std::future<int>> futures;
        bool launched = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (error_status != 0) {
                break;
            }
            while (!ready.empty() && in_flight < worker_count_) {
                const size_t index = ready.front();
                ready.pop();
                ++in_flight;
                futures.push_back(thread_pool_->enqueue(run_node, index));
            }
            launched = !futures.empty();
            if (in_flight == 0 && ready.empty()) {
                break;
            }
            if (!launched) {
                cv.wait(lock, [&] { return stop || error_status != 0 || !ready.empty() || in_flight == 0; });
            }
        }

        for (auto& future : futures) {
            const auto status = future.get();
            if (status != 0) {
                error_status = status;
            }
        }
    }

    if (error_status != 0) {
        return error_status;
    }
    return completed == nodes_.size() ? 0 : -1;
}

}  // namespace feather
