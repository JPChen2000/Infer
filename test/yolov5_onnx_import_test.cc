#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_io.h"
#include "util/fp16.h"

using feather::DataType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelLoader;
using feather::model::ValueDesc;

namespace {

const ValueDesc* FindValueDesc(const feather::model::ModelDesc& model, const std::string& name) {
    for (const auto& value : model.graph.values) {
        if (value.tensor.name == name) {
            return &value;
        }
    }
    return nullptr;
}

std::shared_ptr<Tensor> CreateDeterministicInputTensor(const ValueDesc& value) {
    const auto numel = std::accumulate(value.tensor.dims.begin(), value.tensor.dims.end(), int64_t{1},
                                       std::multiplies<int64_t>());
    std::vector<float> fp32_data(static_cast<size_t>(numel), 0.0f);
    for (size_t i = 0; i < fp32_data.size(); ++i) {
        fp32_data[i] = static_cast<float>((static_cast<int>(i % 251) - 125) / 125.0f);
    }

    auto tensor = std::make_shared<Tensor>();
    if (value.tensor.data_type == DataType::FP16) {
        std::vector<uint16_t> fp16_data(fp32_data.size(), 0);
        for (size_t i = 0; i < fp32_data.size(); ++i) {
            fp16_data[i] = feather::FloatToHalf(fp32_data[i]);
        }
        tensor->Assign<uint16_t>(fp16_data, value.tensor.dims);
        return tensor;
    }

    tensor->Assign<float>(fp32_data, value.tensor.dims);
    return tensor;
}

bool WriteTensorRaw(const std::filesystem::path& path, const Tensor& tensor) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        return false;
    }
    out.write(static_cast<const char*>(tensor.raw_data()), static_cast<std::streamsize>(tensor.memory_size()));
    return out.good();
}

std::vector<float> ReadFloatVectorRaw(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float), 0.0f);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!in.good()) {
        return {};
    }
    return data;
}

std::vector<float> TensorToFloatVector(const Tensor& tensor) {
    std::vector<float> data(static_cast<size_t>(tensor.numel()), 0.0f);
    if (tensor.data_type() == DataType::FP16) {
        const auto* raw = tensor.data<uint16_t>();
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = feather::HalfToFloat(raw[i]);
        }
        return data;
    }
    const auto* raw = tensor.data<float>();
    std::memcpy(data.data(), raw, data.size() * sizeof(float));
    return data;
}

void PrintTensorPreview(const std::string& name, const Tensor& tensor, size_t limit = 12) {
    const auto values = TensorToFloatVector(tensor);
    std::cerr << name << " dims=";
    for (size_t i = 0; i < tensor.dims().size(); ++i) {
        std::cerr << (i == 0 ? "[" : ",") << tensor.dims()[i];
    }
    std::cerr << "] values=";
    for (size_t i = 0; i < std::min(limit, values.size()); ++i) {
        std::cerr << (i == 0 ? "[" : ",") << values[i];
    }
    std::cerr << "]" << std::endl;
}

void PrintTensorRow(const std::string& name, const Tensor& tensor, int64_t row, int64_t width, size_t limit = 8) {
    const auto values = TensorToFloatVector(tensor);
    const auto offset = static_cast<size_t>(row * width);
    if (offset + width > values.size()) {
        return;
    }
    std::cerr << name << " row=" << row << " values=";
    for (size_t i = 0; i < std::min<size_t>(limit, static_cast<size_t>(width)); ++i) {
        std::cerr << (i == 0 ? "[" : ",") << values[offset + i];
    }
    std::cerr << "]" << std::endl;
}

}  // namespace

TEST(yolov5_onnx_import_test, ImportBuildAndLowerYolov5Model) {
    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto script_path = repo_root / "tools" / "onnx_to_feather.py";
    const auto ref_script_path = repo_root / "tools" / "onnx_reference.py";
    const auto onnx_path = repo_root / "third_party" / "models" / "yolov5n.onnx";
    const auto output_path = std::filesystem::temp_directory_path() / "yolov5n_import_test.fth";
    const auto input_raw_path = std::filesystem::temp_directory_path() / "yolov5n_import_test_input.raw";
    const auto output_ref_path = std::filesystem::temp_directory_path() / "yolov5n_import_test_output.raw";

    ASSERT_TRUE(std::filesystem::exists(onnx_path));

    const std::string command = "/home/jarvis/miniconda3/bin/python3 \"" + script_path.string() + "\" --input \"" +
                                onnx_path.string() + "\" --output \"" + output_path.string() + "\"";
    ASSERT_EQ(std::system(command.c_str()), 0);

    ModelLoader loader;
    ASSERT_TRUE(loader.Load(output_path.string()));
    ASSERT_FALSE(loader.model().graph.nodes.empty());

    const auto* input_value = FindValueDesc(loader.model(), loader.model().graph.inputs.front());
    ASSERT_NE(input_value, nullptr);
    auto input_tensor = CreateDeterministicInputTensor(*input_value);
    ASSERT_NE(input_tensor, nullptr);
    ASSERT_TRUE(WriteTensorRaw(input_raw_path, *input_tensor));

    const std::string ref_command =
        "/home/jarvis/miniconda3/bin/python3 \"" + ref_script_path.string() + "\" --model \"" + onnx_path.string() +
        "\" --input-raw \"" + input_raw_path.string() + "\" --output-raw \"" + output_ref_path.string() + "\"";
    ASSERT_EQ(std::system(ref_command.c_str()), 0);
    const auto reference = ReadFloatVectorRaw(output_ref_path);
    ASSERT_FALSE(reference.empty());

    StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);
    ASSERT_EQ(static_graph.SetTensor(input_value->tensor.name, input_tensor), 0);

    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_GT(static_graph.OperatorSize(), 0U);

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_GT(runtime_graph.NodeSize(), 0U);

    ASSERT_EQ(runtime_graph.Run(), 0);
    auto output_tensor = runtime_graph.GetTensor(loader.model().graph.outputs.front());
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 25200, 85}));

    if (const char* debug_yolo = std::getenv("FEATHER_DEBUG_YOLOV5"); debug_yolo != nullptr && std::string(debug_yolo) == "1") {
        const std::vector<std::string> debug_tensors = {
            "/model.24/Sigmoid_output_0",
            "/model.24/Split_output_0",
            "/model.24/Split_output_1",
            "/model.24/Mul_output_0",
            "/model.24/Add_output_0",
            "/model.24/Mul_1_output_0",
            "/model.24/Mul_2_output_0",
            "/model.24/Pow_output_0",
            "/model.24/Mul_3_output_0",
            "/model.24/Concat_output_0",
            "/model.24/Reshape_1_output_0",
            "/model.24/Sigmoid_1_output_0",
            "/model.24/Split_1_output_0",
            "/model.24/Split_1_output_1",
            "/model.24/Mul_4_output_0",
            "/model.24/Add_1_output_0",
            "/model.24/Mul_5_output_0",
            "/model.24/Mul_6_output_0",
            "/model.24/Pow_1_output_0",
            "/model.24/Mul_7_output_0",
            "/model.24/Concat_1_output_0",
            "/model.24/Reshape_3_output_0",
        };
        for (const auto& name : debug_tensors) {
            auto tensor = runtime_graph.GetTensor(name);
            if (tensor != nullptr) {
                PrintTensorPreview(name, *tensor);
            }
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Reshape_3_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Reshape_3_output_0", *tensor, 3858, 85);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Split_1_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Split_1_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Split_1_output_1"); tensor != nullptr) {
            PrintTensorRow("/model.24/Split_1_output_1", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Mul_4_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Mul_4_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Add_1_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Add_1_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Mul_5_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Mul_5_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Mul_6_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Mul_6_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Pow_1_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Pow_1_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("/model.24/Mul_7_output_0"); tensor != nullptr) {
            PrintTensorRow("/model.24/Mul_7_output_0", *tensor, 3858, 2);
        }
        if (auto tensor = runtime_graph.GetTensor("output0"); tensor != nullptr) {
            PrintTensorRow("output0", *tensor, 23058, 85);
        }
    }

    const auto actual = TensorToFloatVector(*output_tensor);
    ASSERT_EQ(actual.size(), reference.size());

    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    size_t max_abs_diff_index = 0;
    std::vector<float> channel_max_diff(85, 0.0f);
    std::vector<float> channel_mean_diff(85, 0.0f);
    float bbox_max_abs_diff = 0.0f;
    float other_max_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_abs_diff_index = i;
        }
        mean_abs_diff += diff;
        const size_t channel = i % 85;
        channel_max_diff[channel] = std::max(channel_max_diff[channel], diff);
        channel_mean_diff[channel] += diff;
        if (channel < 4) {
            bbox_max_abs_diff = std::max(bbox_max_abs_diff, diff);
        } else {
            other_max_abs_diff = std::max(other_max_abs_diff, diff);
        }
    }
    mean_abs_diff /= static_cast<float>(actual.size());
    for (auto& value : channel_mean_diff) {
        value /= static_cast<float>(actual.size() / 85);
    }

    if (max_abs_diff > 0.15f || mean_abs_diff > 0.01f) {
        const size_t anchor_index = max_abs_diff_index / 85;
        const size_t channel_index = max_abs_diff_index % 85;
        std::cerr << "yolov5 diff stats max=" << max_abs_diff << " mean=" << mean_abs_diff
                  << " max_index=" << max_abs_diff_index << " anchor=" << anchor_index
                  << " channel=" << channel_index << " actual=" << actual[max_abs_diff_index]
                  << " ref=" << reference[max_abs_diff_index] << std::endl;
        for (size_t channel = 0; channel < channel_max_diff.size(); ++channel) {
            if (channel_max_diff[channel] < 0.05f && channel_mean_diff[channel] < 0.001f) {
                continue;
            }
            std::cerr << "channel " << channel << " max_diff=" << channel_max_diff[channel]
                      << " mean_diff=" << channel_mean_diff[channel] << std::endl;
        }
    }

    EXPECT_LE(bbox_max_abs_diff, 2.0f);
    EXPECT_LE(other_max_abs_diff, 0.5f);
    EXPECT_LE(mean_abs_diff, 0.0015f);
}
