#include "core/static_graph.h"

#include <algorithm>
#include <utility>

#include "core/operator_registry.h"

namespace feather {

namespace {

std::shared_ptr<Tensor> CreateTensorFromValue(const model::ValueDesc& value) {
    if (value.tensor.dims.empty()) {
        return nullptr;
    }
    auto tensor = std::make_shared<Tensor>(value.tensor.dims);
    tensor->set_data_type(value.tensor.data_type);
    tensor->set_layout(value.tensor.layout);
    return tensor;
}

}  // namespace

int32_t StaticGraph::SetModel(const model::ModelDesc& model) {
    model_ = model;
    ClearGraphState();
    if (pass_manager_ == nullptr) {
        pass_manager_ = CreateDefaultPassManager();
    }
    return 0;
}

int32_t StaticGraph::SetTensor(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (name.empty() || tensor == nullptr) {
        return -1;
    }
    tensors_[name] = std::move(tensor);
    return 0;
}

std::shared_ptr<Tensor> StaticGraph::GetTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return it->second;
}

int32_t StaticGraph::Build() {
    ClearGraphState();

    for (const auto& input_name : model_.graph.inputs) {
        if (GetTensor(input_name) == nullptr) {
            return -1;
        }
    }

    for (const auto& value : model_.graph.values) {
        auto it = tensors_.find(value.tensor.name);
        if (it != tensors_.end()) {
            continue;
        }
        if (value.constant) {
            return -1;
        }
        auto tensor = CreateTensorFromValue(value);
        if (tensor == nullptr) {
            return -1;
        }
        tensors_[value.tensor.name] = tensor;
    }

    KernelDeviceScope kernel_device_scope(kernel_device_);
    for (const auto& node : model_.graph.nodes) {
        auto op = OperatorRegistry::instance().Create(node, tensors_);
        if (op == nullptr) {
            return -1;
        }
        const auto& outputs = op->outputs();
        if (node.outputs.size() != outputs.size()) {
            return -1;
        }
        for (size_t i = 0; i < node.outputs.size(); ++i) {
            if (outputs[i] == nullptr) {
                return -1;
            }
            tensors_[node.outputs[i]] = outputs[i];
            producer_by_value_[node.outputs[i]] = node.name;
        }
        for (const auto& input_name : node.inputs) {
            RegisterValueUse(input_name, node.name);
        }
        StaticNode static_node;
        static_node.name = node.name;
        static_node.op_type = node.op_type;
        static_node.inputs = node.inputs;
        static_node.outputs = node.outputs;
        static_node.op = op;
        node_index_by_name_[static_node.name] = nodes_.size();
        nodes_.push_back(std::move(static_node));
        AddOperator(std::move(op));
    }

    return Check();
}

int32_t StaticGraph::Check() const {
    for (const auto& op : operators_) {
        if (op == nullptr || !op->HasKernel()) {
            return -1;
        }
    }
    return 0;
}

int32_t StaticGraph::ApplyPasses() {
    if (pass_manager_ == nullptr) {
        return 0;
    }
    return pass_manager_->Run(this);
}

void StaticGraph::SetPassManager(std::shared_ptr<PassManager> pass_manager) { pass_manager_ = std::move(pass_manager); }

void StaticGraph::AddOperator(std::shared_ptr<OpBase> op) { operators_.push_back(std::move(op)); }

size_t StaticGraph::OperatorSize() const { return operators_.size(); }

size_t StaticGraph::NodeSize() const {
    size_t count = 0;
    for (const auto& node : nodes_) {
        if (!node.removed) {
            ++count;
        }
    }
    return count;
}

const StaticNode* StaticGraph::GetNode(const std::string& name) const {
    auto it = node_index_by_name_.find(name);
    if (it == node_index_by_name_.end()) {
        return nullptr;
    }
    const auto& node = nodes_[it->second];
    if (node.removed) {
        return nullptr;
    }
    return &node;
}

std::string StaticGraph::GetProducer(const std::string& value_name) const {
    auto it = producer_by_value_.find(value_name);
    if (it == producer_by_value_.end()) {
        return "";
    }
    return it->second;
}

std::vector<std::string> StaticGraph::GetUsers(const std::string& value_name) const {
    auto it = users_by_value_.find(value_name);
    if (it == users_by_value_.end()) {
        return {};
    }
    return it->second;
}

bool StaticGraph::RemoveNode(const std::string& node_name) {
    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
        return false;
    }
    auto& node = nodes_[it->second];
    if (node.removed) {
        return false;
    }
    for (const auto& output_name : node.outputs) {
        if (std::find(model_.graph.outputs.begin(), model_.graph.outputs.end(), output_name) != model_.graph.outputs.end()) {
            return false;
        }
        if (!GetUsers(output_name).empty()) {
            return false;
        }
    }

    for (const auto& input_name : node.inputs) {
        UnregisterValueUse(input_name, node_name);
    }
    for (const auto& output_name : node.outputs) {
        auto producer_it = producer_by_value_.find(output_name);
        if (producer_it != producer_by_value_.end() && producer_it->second == node_name) {
            producer_by_value_.erase(producer_it);
        }
        users_by_value_.erase(output_name);
    }
    node.removed = true;
    node.op.reset();
    RebuildActiveOperators();
    return true;
}

bool StaticGraph::ReplaceInputValue(const std::string& node_name, const std::string& from, const std::string& to) {
    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
        return false;
    }
    auto& node = nodes_[it->second];
    if (node.removed) {
        return false;
    }

    bool replaced = false;
    for (auto& input_name : node.inputs) {
        if (input_name != from) {
            continue;
        }
        UnregisterValueUse(from, node_name);
        input_name = to;
        RegisterValueUse(to, node_name);
        replaced = true;
    }
    if (!replaced) {
        return false;
    }
    if (!RebuildNode(it->second)) {
        return false;
    }
    return replaced;
}

bool StaticGraph::ReplaceNodeOp(const std::string& node_name, const std::string& op_type,
                                const std::vector<std::string>& inputs) {
    model::NodeDesc desc;
    desc.name = node_name;
    desc.op_type = op_type;
    desc.inputs = inputs;
    for (const auto& node : model_.graph.nodes) {
        if (node.name != node_name) {
            continue;
        }
        desc.outputs = node.outputs;
        desc.attributes = node.attributes;
        desc.domain = node.domain;
        break;
    }
    return ReplaceNodeDesc(desc);
}

bool StaticGraph::ReplaceNodeDesc(const model::NodeDesc& desc) {
    auto it = node_index_by_name_.find(desc.name);
    if (it == node_index_by_name_.end() || desc.name.empty() || desc.op_type.empty()) {
        return false;
    }
    auto& static_node = nodes_[it->second];
    if (static_node.removed || desc.outputs != static_node.outputs) {
        return false;
    }
    for (const auto& input : desc.inputs) {
        if (GetTensor(input) == nullptr) {
            return false;
        }
    }
    KernelDeviceScope kernel_device_scope(kernel_device_);
    auto op = OperatorRegistry::instance().Create(desc, tensors_);
    if (op == nullptr || op->outputs().size() != static_node.outputs.size()) {
        return false;
    }
    for (size_t i = 0; i < static_node.outputs.size(); ++i) {
        if (op->outputs()[i] == nullptr) {
            return false;
        }
    }

    for (const auto& input : static_node.inputs) {
        UnregisterValueUse(input, desc.name);
    }
    static_node.op_type = desc.op_type;
    static_node.inputs = desc.inputs;
    static_node.op = op;
    for (const auto& input : static_node.inputs) {
        RegisterValueUse(input, desc.name);
    }
    for (size_t i = 0; i < static_node.outputs.size(); ++i) {
        tensors_[static_node.outputs[i]] = op->outputs()[i];
        producer_by_value_[static_node.outputs[i]] = static_node.name;
    }
    for (auto& model_node : model_.graph.nodes) {
        if (model_node.name == desc.name) {
            model_node = desc;
            break;
        }
    }
    RebuildActiveOperators();
    return true;
}

void StaticGraph::ClearGraphState() {
    operators_.clear();
    nodes_.clear();
    node_index_by_name_.clear();
    producer_by_value_.clear();
    users_by_value_.clear();
}

void StaticGraph::RegisterValueUse(const std::string& value_name, const std::string& node_name) {
    auto& users = users_by_value_[value_name];
    if (std::find(users.begin(), users.end(), node_name) == users.end()) {
        users.push_back(node_name);
    }
}

void StaticGraph::UnregisterValueUse(const std::string& value_name, const std::string& node_name) {
    auto it = users_by_value_.find(value_name);
    if (it == users_by_value_.end()) {
        return;
    }
    auto& users = it->second;
    users.erase(std::remove(users.begin(), users.end(), node_name), users.end());
    if (users.empty()) {
        users_by_value_.erase(it);
    }
}

void StaticGraph::RebuildActiveOperators() {
    operators_.clear();
    for (const auto& node : nodes_) {
        if (!node.removed && node.op != nullptr) {
            operators_.push_back(node.op);
        }
    }
}

bool StaticGraph::RebuildNode(size_t node_index) {
    if (node_index >= nodes_.size()) {
        return false;
    }
    auto& static_node = nodes_[node_index];
    if (static_node.removed) {
        return false;
    }

    model::NodeDesc desc;
    desc.name = static_node.name;
    desc.op_type = static_node.op_type;
    desc.inputs = static_node.inputs;
    desc.outputs = static_node.outputs;

    for (const auto& node : model_.graph.nodes) {
        if (node.name != static_node.name) {
            continue;
        }
        desc.attributes = node.attributes;
        desc.domain = node.domain;
        break;
    }

    KernelDeviceScope kernel_device_scope(kernel_device_);
    auto op = OperatorRegistry::instance().Create(desc, tensors_);
    if (op == nullptr) {
        return false;
    }
    static_node.op = op;
    for (size_t i = 0; i < static_node.outputs.size(); ++i) {
        const auto& outputs = op->outputs();
        if (i >= outputs.size() || outputs[i] == nullptr) {
            return false;
        }
        tensors_[static_node.outputs[i]] = outputs[i];
        producer_by_value_[static_node.outputs[i]] = static_node.name;
    }
    RebuildActiveOperators();
    return true;
}

}  // namespace feather
