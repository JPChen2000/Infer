#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/model_io.h"
#include "quant/static_quantizer.h"

namespace {

void Usage(const char* program) {
    std::cerr << "usage: " << program << " --input MODEL --output MODEL --scales TABLE"
              << " [--non-strict] [--asymmetric] [--per-tensor-weights]"
              << " [--keep-fp32-prefix PREFIX]\n";
}

bool ValueAfter(int argc, char** argv, int* index, std::string* value) {
    if (index == nullptr || value == nullptr || *index + 1 >= argc) return false;
    *value = argv[++*index];
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string scales_path;
    bool strict = true;
    bool symmetric = true;
    bool per_channel_weights = true;
    std::vector<std::string> keep_fp32_prefixes;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input") {
            if (!ValueAfter(argc, argv, &index, &input_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--output") {
            if (!ValueAfter(argc, argv, &index, &output_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--scales") {
            if (!ValueAfter(argc, argv, &index, &scales_path)) { Usage(argv[0]); return 2; }
        } else if (argument == "--non-strict") {
            strict = false;
        } else if (argument == "--asymmetric") {
            symmetric = false;
        } else if (argument == "--per-tensor-weights") {
            per_channel_weights = false;
        } else if (argument == "--keep-fp32-prefix") {
            std::string prefix;
            if (!ValueAfter(argc, argv, &index, &prefix) || prefix.empty()) { Usage(argv[0]); return 2; }
            keep_fp32_prefixes.push_back(std::move(prefix));
        } else {
            Usage(argv[0]);
            return 2;
        }
    }
    if (input_path.empty() || output_path.empty() || scales_path.empty()) { Usage(argv[0]); return 2; }
    feather::ActivationQuantizationTable activations;
    std::vector<std::string> diagnostics;
    if (feather::LoadActivationQuantizationTable(scales_path, &activations, &diagnostics) != 0) {
        for (const auto& item : diagnostics) std::cerr << item << '\n';
        return 3;
    }
    feather::model::ModelLoader loader;
    if (!loader.Load(input_path)) { std::cerr << "failed to load model: " << input_path << '\n'; return 4; }
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> source_weights;
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) continue;
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr) { std::cerr << "failed to load weight: " << value.tensor.name << '\n'; return 5; }
        source_weights[value.tensor.name] = std::move(tensor);
    }
    feather::StaticQuantizationConfig config;
    config.strict = strict;
    config.symmetric = symmetric;
    config.per_channel_weights = per_channel_weights;
    config.keep_fp32_node_prefixes = std::move(keep_fp32_prefixes);
    config.activations = std::move(activations);
    feather::model::ModelDesc output_model;
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> output_weights;
    feather::StaticQuantizationReport report;
    if (feather::StaticQuantizeModel(loader.model(), source_weights, config, &output_model, &output_weights, &report) != 0) {
        for (const auto& item : report.diagnostics) std::cerr << item << '\n';
        return 6;
    }
    feather::model::ModelWriter writer;
    if (!writer.Save(output_path, output_model, output_weights)) {
        std::cerr << "failed to save model: " << output_path << '\n';
        return 7;
    }
    std::cout << "quantized_nodes=" << report.quantized_nodes.size()
              << " skipped_nodes=" << report.skipped_nodes.size() << '\n';
    return 0;
}
