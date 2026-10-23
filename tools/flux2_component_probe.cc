// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Numerical validation tool: each input/output file contains raw declared dtype.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "core/execution_plan.h"
#include "model/model_io.h"

int main(int argc, char** argv) {
    using namespace feather;
    if (argc < 5) {
        std::cerr << "usage: flux2_component_probe MODEL OUTPUT_DIR common|cuda NAME=RAW ...\n";
        return 2;
    }
    model::ModelLoader loader;
    if (!loader.Load(argv[1])) {
        std::cerr << "cannot load model\n";
        return 1;
    }
    const std::string backend = argv[3];
    if (backend != "common" && backend != "cuda") return 2;
    std::map<std::string, std::string> paths;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto split = arg.find('=');
        if (split == std::string::npos || !paths.emplace(arg.substr(0, split), arg.substr(split + 1)).second) return 2;
    }
    StaticGraph graph;
    graph.SetKernelDevice(backend == "cuda" ? DeviceType::CUDA : DeviceType::COMMON);
    graph.SetFallbackPolicy(KernelFallbackPolicy::kRequireRequested);
    if (graph.SetModel(loader.model()) != 0) return 1;
    for (const auto& value : loader.model().graph.values) {
        const auto& desc = value.tensor;
        if (value.constant) {
            auto weight = loader.CreateWeightTensor(desc.name);
            if (!weight || graph.SetTensor(desc.name, weight) != 0) return 1;
        } else if (paths.count(desc.name)) {
            size_t bytes = 0;
            if (!CheckedTensorBytes(desc.dims, desc.data_type, &bytes)) return 1;
            std::ifstream input(paths.at(desc.name), std::ios::binary | std::ios::ate);
            if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
                std::cerr << "input byte count mismatch: " << desc.name << '\n';
                return 1;
            }
            auto tensor = std::make_shared<Tensor>(bytes);
            tensor->Resize(desc.dims);
            tensor->set_data_type(desc.data_type);
            tensor->set_layout(desc.layout);
            input.seekg(0);
            input.read(static_cast<char*>(tensor->mutable_data(bytes)), bytes);
            if (!input || graph.SetTensor(desc.name, tensor) != 0) return 1;
        }
    }
    for (const auto& name : loader.model().graph.inputs)
        if (!paths.count(name)) {
            std::cerr << "missing input: " << name << '\n';
            return 1;
        }
    const auto check = [](const Status& status) {
        if (!status.ok()) std::cerr << status.message() << '\n';
        return status.ok();
    };
    if (!check(graph.BuildStatus())) return 1;
    std::shared_ptr<const CompiledGraph> compiled;
    if (!check(CompiledGraph::Compile(graph, &compiled))) return 1;
    std::unique_ptr<ExecutionSession> session;
    if (!check(compiled->CreateSession(&session))) return 1;
    // Input values initialized in the static graph seed the session.
    if (!check(session->Run())) return 1;
    std::filesystem::create_directories(argv[2]);
    for (const auto& name : loader.model().graph.outputs) {
        std::shared_ptr<Tensor> tensor;
        if (!check(session->ReadOutput(name, &tensor))) return 1;
        size_t bytes = 0;
        if (!CheckedTensorBytes(tensor->dims().data(), tensor->data_type(), &bytes)) return 1;
        std::ofstream output(std::filesystem::path(argv[2]) / (name + ".bin"), std::ios::binary);
        output.write(static_cast<const char*>(tensor->raw_data()), bytes);
        if (!output) return 1;
    }
    return 0;
}
