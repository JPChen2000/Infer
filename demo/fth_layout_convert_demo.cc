#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "model/layout_transform.h"
#include "model/model_io.h"

namespace {

void PrintUsage() {
    std::cerr << "usage: fth_layout_convert --input <model.fth> --output <model.fth> --layout nhwc\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string layout = "nhwc";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--layout" && i + 1 < argc) {
            layout = argv[++i];
        } else {
            PrintUsage();
            return 1;
        }
    }

    if (input_path.empty() || output_path.empty() || layout != "nhwc") {
        PrintUsage();
        return 1;
    }

    feather::model::ModelLoader loader;
    if (!loader.Load(input_path)) {
        std::cerr << "failed to load model: " << input_path << '\n';
        return 1;
    }

    auto model = loader.model();
    if (!feather::model::ConvertModelToNhwcInPlace(&model)) {
        std::cerr << "failed to convert model layout\n";
        return 1;
    }

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    for (const auto& value : model.graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        if (tensor == nullptr) {
            std::cerr << "failed to load weight tensor: " << value.tensor.name << '\n';
            return 1;
        }
        weights.emplace(value.tensor.name, std::move(tensor));
    }

    feather::model::ModelWriter writer;
    if (!writer.Save(output_path, model, weights)) {
        std::cerr << "failed to save converted model: " << output_path << '\n';
        return 1;
    }

    std::cout << "saved NHWC model: " << output_path << '\n';
    return 0;
}
