#include "core/graph.h"

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include "util/timer.h"

namespace feather {

int32_t RuntimeGraph::load_from_buffer(const char* buffer, size_t size) {
    if (buffer == nullptr || size == 0) {
        return -1;
    }
    return 0;
}

int32_t RuntimeGraph::load_from_path(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return -1;
    }
    return 0;
}

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

void RuntimeGraph::Clear() {
    nodes_.clear();
    tensors_.clear();
    node_index_by_name_.clear();
    thread_pool_.reset();
    worker_count_ = 1;
}

int32_t RuntimeGraph::Check() const {
    for (const auto& node : nodes_) {
        if (node.owner == nullptr || node.kernel == nullptr) {
            return -1;
        }
    }
    return 0;
}

int32_t RuntimeGraph::Run() {
    auto status = Check();
    if (status != 0) {
        return status;
    }
    ResetPendingDependencies();
    return RunSerial();
}

void RuntimeGraph::AddNode(RuntimeNode node) {
    node_index_by_name_[node.name] = nodes_.size();
    nodes_.push_back(std::move(node));
}

int32_t RuntimeGraph::Finalize() {
    const auto status = BuildDependencies();
    if (status != 0) {
        thread_pool_.reset();
        worker_count_ = 1;
        return status;
    }

    worker_count_ = std::max<size_t>(1, std::thread::hardware_concurrency());
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

    std::unordered_map<std::string, size_t> producer_by_value;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& output_name : nodes_[i].outputs) {
            producer_by_value[output_name] = i;
        }
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& input_name : nodes_[i].inputs) {
            auto producer_it = producer_by_value.find(input_name);
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
            auto status = nodes_[index].Run();
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
        const auto status = nodes_[index].Run();
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
