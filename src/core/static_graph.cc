#include "core/static_graph.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "core/operator_registry.h"

namespace feather {

namespace {

std::shared_ptr<Tensor> CreateTensorFromValue(const model::ValueDesc& value) {
    auto tensor = std::make_shared<Tensor>(value.tensor.dims);
    tensor->set_data_type(value.tensor.data_type);
    tensor->set_layout(value.tensor.layout);
    return tensor;
}

const model::ValueDesc* FindValueDesc(const model::ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) {
            return &value;
        }
    }
    return nullptr;
}

void RestoreDeclaredTensorMetadata(const model::ModelDesc& model, const std::string& name,
                                   const std::shared_ptr<Tensor>& tensor) {
    const auto* value = FindValueDesc(model, name);
    if (value == nullptr || tensor == nullptr) {
        return;
    }
    if (value->tensor.data_type != DataType::UNKNOWN) {
        tensor->set_data_type(value->tensor.data_type);
    }
    if (value->tensor.layout != DataLayout::ND) {
        tensor->set_layout(value->tensor.layout);
    }
}

bool IsBuildTimeEvaluableControlOp(const std::string& op_type) {
    return op_type == "Shape" || op_type == "Unsqueeze" || op_type == "Squeeze" || op_type == "Concat" ||
           op_type == "Cast" || op_type == "ConstantOfShape" || op_type == "Slice" || op_type == "Mul" ||
           op_type == "Equal" || op_type == "Where";
}

bool HasOnlyStaticInputs(const model::NodeDesc& node, const std::unordered_set<std::string>& static_values) {
    if (node.inputs.empty()) {
        return false;
    }
    for (const auto& input : node.inputs) {
        if (input.empty() || static_values.count(input) == 0) {
            return false;
        }
    }
    return true;
}

int32_t EvaluateBuildTimeControlOp(const std::shared_ptr<OpBase>& op) {
    if (op == nullptr) {
        return -1;
    }

    auto runtime_kernel = op->DetachKernel();
    if (runtime_kernel == nullptr) {
        return -1;
    }
    auto control_kernel = KernelDispatcher::instance().create(DeviceType::COMMON, runtime_kernel->data_type(),
                                                               runtime_kernel->layout(), op->type());
    if (control_kernel == nullptr) {
        op->AttachKernel(std::move(runtime_kernel));
        return -1;
    }

    op->AttachKernel(std::move(control_kernel));
    const int32_t status = op->Run();
    (void)op->DetachKernel();
    op->AttachKernel(std::move(runtime_kernel));
    return status;
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

bool StaticGraph::SetValueDataType(const std::string& name, DataType data_type) {
    if (name.empty() || data_type == DataType::UNKNOWN) {
        return false;
    }
    bool found = false;
    for (auto& value : model_.graph.values) {
        if (value.tensor.name == name) {
            value.tensor.data_type = data_type;
            found = true;
            break;
        }
    }
    const auto tensor = GetTensor(name);
    if (tensor != nullptr) {
        tensor->set_data_type(data_type);
        found = true;
    }
    return found;
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

    std::unordered_set<std::string> static_values;

    for (const auto& input_name : model_.graph.inputs) {
        if (GetTensor(input_name) == nullptr) {
            return -1;
        }
    }

    for (const auto& value : model_.graph.values) {
        auto it = tensors_.find(value.tensor.name);
        if (it != tensors_.end()) {
            it->second->set_immutable(value.constant);
            if (value.constant) {
                static_values.insert(value.tensor.name);
            }
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
        if (value.constant) {
            static_values.insert(value.tensor.name);
        }
    }

    for (const auto& node : model_.graph.nodes) {
        auto op = OperatorRegistry::instance().Create(node, tensors_, OperatorRegistry::BuildContext{kernel_device_});
        if (op == nullptr) {
            std::cerr << "StaticGraph::Build failed to create node=" << node.name << " op=" << node.op_type << '\n';
            return -1;
        }
        // Evaluate static control values early so downstream shape inference can
        // consume them. The original node and its input edges remain in the
        // static graph and are executed again by the runtime graph.
        const bool is_shape_node = node.op_type == "Shape";
        const bool can_evaluate_at_build =
            is_shape_node || (IsBuildTimeEvaluableControlOp(node.op_type) && HasOnlyStaticInputs(node, static_values));
        if (can_evaluate_at_build && EvaluateBuildTimeControlOp(op) != 0) {
            std::cerr << "StaticGraph::Build control evaluation failed node=" << node.name << " op=" << node.op_type << '\n';
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
            RestoreDeclaredTensorMetadata(model_, node.outputs[i], outputs[i]);
            tensors_[node.outputs[i]] = outputs[i];
            producer_by_value_[node.outputs[i]] = node.name;
            if (can_evaluate_at_build) {
                static_values.insert(node.outputs[i]);
            }
        }
        for (const auto& input_name : node.inputs) {
            if (!input_name.empty()) {
                RegisterValueUse(input_name, node.name);
            }
        }
        StaticNode static_node;
        static_node.name = node.name;
        static_node.op_type = node.op_type;
        static_node.inputs = node.inputs;
        static_node.outputs = node.outputs;
        static_node.op = op;
        node_index_by_name_[static_node.name] = nodes_.size();
        nodes_.push_back(std::move(static_node));
    }

    return Check();
}

int32_t StaticGraph::Check() const {
    for (const auto& node : nodes_) {
        if (node.removed) {
            continue;
        }
        if (node.op == nullptr || !node.op->HasKernel()) {
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

size_t StaticGraph::OperatorSize() const {
    size_t count = 0;
    for (const auto& node : nodes_) {
        if (!node.removed && node.op != nullptr) {
            ++count;
        }
    }
    return count;
}

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

bool StaticGraph::IsGraphOutputValue(const std::string& value_name) const {
    for (const auto& output_name : model_.graph.outputs) {
        std::string current = output_name;
        std::unordered_set<std::string> visited;
        while (visited.insert(current).second) {
            if (current == value_name) {
                return true;
            }
            const auto alias = output_aliases_.find(current);
            if (alias == output_aliases_.end()) {
                break;
            }
            current = alias->second;
        }
    }
    return false;
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
        if (IsGraphOutputValue(output_name)) {
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
    return true;
}

bool StaticGraph::RemoveNodeWithOutputAlias(const std::string& node_name) {
    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
        return false;
    }
    auto& node = nodes_[it->second];
    if (node.removed || node.inputs.size() != 1 || node.outputs.size() != 1) {
        return false;
    }

    const std::string& input_name = node.inputs[0];
    const std::string& output_name = node.outputs[0];
    if (std::find(model_.graph.outputs.begin(), model_.graph.outputs.end(), output_name) == model_.graph.outputs.end() ||
        !GetUsers(output_name).empty()) {
        return false;
    }
    const auto input = GetTensor(input_name);
    const auto output = GetTensor(output_name);
    if (input == nullptr || output == nullptr || input->dims() != output->dims() ||
        input->data_type() != output->data_type() || input->layout() != output->layout()) {
        return false;
    }

    // Keep the public graph-output name, but let lowering expose the source
    // Tensor directly so an otherwise no-op relay does not become a runtime
    // node.
    tensors_[output_name] = input;
    output_aliases_[output_name] = input_name;
    UnregisterValueUse(input_name, node_name);
    auto producer_it = producer_by_value_.find(output_name);
    if (producer_it != producer_by_value_.end() && producer_it->second == node_name) {
        producer_by_value_.erase(producer_it);
    }
    users_by_value_.erase(output_name);
    node.removed = true;
    node.op.reset();
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
    auto op = OperatorRegistry::instance().Create(desc, tensors_, OperatorRegistry::BuildContext{kernel_device_});
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
        RestoreDeclaredTensorMetadata(model_, static_node.outputs[i], op->outputs()[i]);
        tensors_[static_node.outputs[i]] = op->outputs()[i];
        producer_by_value_[static_node.outputs[i]] = static_node.name;
    }
    for (auto& model_node : model_.graph.nodes) {
        if (model_node.name == desc.name) {
            model_node = desc;
            break;
        }
    }
    return true;
}

bool StaticGraph::ReplaceNodeDescAndAbsorbNode(const model::NodeDesc& desc,
                                                const std::string& absorbed_node_name) {
    auto target_it = node_index_by_name_.find(desc.name);
    auto absorbed_it = node_index_by_name_.find(absorbed_node_name);
    if (target_it == node_index_by_name_.end() || absorbed_it == node_index_by_name_.end() ||
        desc.name.empty() || desc.op_type.empty() || desc.name == absorbed_node_name) {
        return false;
    }

    auto& target = nodes_[target_it->second];
    auto& absorbed = nodes_[absorbed_it->second];
    if (target.removed || absorbed.removed || desc.outputs.size() <= target.outputs.size() ||
        !std::equal(target.outputs.begin(), target.outputs.end(), desc.outputs.begin())) {
        return false;
    }
    const auto appended_begin = desc.outputs.begin() + static_cast<std::ptrdiff_t>(target.outputs.size());
    if (!std::equal(appended_begin, desc.outputs.end(), absorbed.outputs.begin(), absorbed.outputs.end())) {
        return false;
    }
    for (const auto& output_name : absorbed.outputs) {
        if (GetProducer(output_name) != absorbed_node_name) {
            return false;
        }
    }
    for (const auto& input_name : desc.inputs) {
        if (GetTensor(input_name) == nullptr) {
            return false;
        }
    }

    auto model_target = std::find_if(model_.graph.nodes.begin(), model_.graph.nodes.end(),
                                     [&](const model::NodeDesc& node) { return node.name == desc.name; });
    if (model_target == model_.graph.nodes.end()) {
        return false;
    }

    auto op = OperatorRegistry::instance().Create(desc, tensors_, OperatorRegistry::BuildContext{kernel_device_});
    if (op == nullptr || op->outputs().size() != desc.outputs.size()) {
        return false;
    }
    for (const auto& output : op->outputs()) {
        if (output == nullptr) {
            return false;
        }
    }

    for (const auto& input_name : target.inputs) {
        UnregisterValueUse(input_name, desc.name);
    }
    target.op_type = desc.op_type;
    target.inputs = desc.inputs;
    target.outputs = desc.outputs;
    target.op = std::move(op);
    for (const auto& input_name : target.inputs) {
        RegisterValueUse(input_name, desc.name);
    }
    for (size_t index = 0; index < target.outputs.size(); ++index) {
        RestoreDeclaredTensorMetadata(model_, target.outputs[index], target.op->outputs()[index]);
        tensors_[target.outputs[index]] = target.op->outputs()[index];
        producer_by_value_[target.outputs[index]] = target.name;
    }
    *model_target = desc;

    for (const auto& input_name : absorbed.inputs) {
        UnregisterValueUse(input_name, absorbed_node_name);
    }
    absorbed.removed = true;
    absorbed.op.reset();
    return true;
}

void StaticGraph::ClearGraphState() {
    nodes_.clear();
    node_index_by_name_.clear();
    producer_by_value_.clear();
    users_by_value_.clear();
    output_aliases_.clear();
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

    auto op = OperatorRegistry::instance().Create(desc, tensors_, OperatorRegistry::BuildContext{kernel_device_});
    if (op == nullptr) {
        return false;
    }
    static_node.op = op;
    for (size_t i = 0; i < static_node.outputs.size(); ++i) {
        const auto& outputs = op->outputs();
        if (i >= outputs.size() || outputs[i] == nullptr) {
            return false;
        }
        RestoreDeclaredTensorMetadata(model_, static_node.outputs[i], outputs[i]);
        tensors_[static_node.outputs[i]] = outputs[i];
        producer_by_value_[static_node.outputs[i]] = static_node.name;
    }
    return true;
}

}  // namespace feather
