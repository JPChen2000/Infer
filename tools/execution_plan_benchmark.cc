#include <chrono>
#include <iostream>
#include "core/execution_plan.h"

int main() {
    using namespace feather;
    constexpr int nodes = 32, elements = 65536, iterations = 100;
    model::ModelDesc model;
    model.graph.inputs = {"x"}; model.graph.outputs = {"v31"};
    for (int i = -2; i < nodes; ++i) {
        model::ValueDesc value;
        value.tensor.name = i == -2 ? "x" : i == -1 ? "b" : "v" + std::to_string(i);
        value.tensor.dims = {elements}; value.tensor.data_type = DataType::FP32; value.constant = i == -1;
        model.graph.values.push_back(value);
    }
    for (int i = 0; i < nodes; ++i) {
        model::NodeDesc node; node.name = "add" + std::to_string(i); node.op_type = "Add";
        node.inputs = {i == 0 ? "x" : "v" + std::to_string(i - 1), "b"}; node.outputs = {"v" + std::to_string(i)};
        model.graph.nodes.push_back(node);
    }
    StaticGraph graph; graph.SetKernelDevice(DeviceType::COMMON);
    if (graph.SetModel(model) != 0) return 1;
    for (const auto* name : {"x", "b"}) {
        auto tensor = std::make_shared<Tensor>(); tensor->Assign<float>(std::vector<float>(elements, 1), {elements});
        if (graph.SetTensor(name, std::move(tensor)) != 0) return 1;
    }
    if (graph.Build() != 0) return 1;
    std::shared_ptr<const CompiledGraph> compiled;
    auto status = CompiledGraph::Compile(graph, &compiled);
    if (!status.ok()) { std::cerr << status.message() << '\n'; return 1; }
    std::cout << "reuse,mutable_host_bytes,shared_weight_bytes,arena_bytes,mean_us\n";
    for (bool reuse : {false, true}) {
        SessionOptions options; options.reuse_memory = reuse;
        std::unique_ptr<ExecutionSession> session;
        status = compiled->CreateSession(&session, options);
        if (!status.ok()) { std::cerr << status.message() << '\n'; return 1; }
        for (int i = 0; i < 5; ++i) if (!session->Run().ok()) return 1;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) if (!session->Run().ok()) return 1;
        const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
        std::shared_ptr<Tensor> output;
        if (!session->ReadOutput("v31", &output).ok()) return 1;
        for (int i = 0; i < elements; ++i) if (output->data<float>()[i] != nodes + 1) return 1;
        const auto& memory = session->memory_usage();
        std::cout << reuse << ',' << memory.mutable_host_bytes << ',' << memory.shared_weight_bytes << ','
                  << memory.arena_bytes << ',' << elapsed / iterations << '\n';
    }
}
