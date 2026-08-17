#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_io.h"
#include "util/fp16.h"

using feather::DataType;
using feather::DeviceType;
using feather::GraphLowering;
using feather::RuntimeGraph;
using feather::StaticGraph;
using feather::Tensor;
using feather::model::ModelLoader;
using feather::model::ValueDesc;

namespace {

std::filesystem::path RepositoryRoot() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

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
    std::vector<float> data(static_cast<size_t>(numel), 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % 257) - 128) / 128.0f);
    }

    auto tensor = std::make_shared<Tensor>();
    if (value.tensor.data_type == DataType::FP16) {
        std::vector<uint16_t> fp16_data(data.size(), 0);
        for (size_t i = 0; i < data.size(); ++i) {
            fp16_data[i] = feather::FloatToHalf(data[i]);
        }
        tensor->Assign<uint16_t>(fp16_data, value.tensor.dims);
        return tensor;
    }

    tensor->Assign<float>(data, value.tensor.dims);
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

void RunClassificationModelAgainstOnnxRuntime(const std::filesystem::path& onnx_path, const std::string& case_name) {
    if (!std::filesystem::exists(onnx_path)) {
        GTEST_SKIP() << "exported model is not available: " << onnx_path;
    }

    const auto repo_root = RepositoryRoot();
    const auto convert_script = repo_root / "tools" / "onnx_to_feather.py";
    const auto ref_script = repo_root / "tools" / "onnx_reference.py";
    const auto fth_path = onnx_path.parent_path() / (case_name + ".fth");
    const auto input_raw_path = std::filesystem::temp_directory_path() / (case_name + "_input.raw");
    const auto output_ref_path = std::filesystem::temp_directory_path() / (case_name + "_ort.raw");

    const std::string convert_command = "/home/jarvis/miniconda3/bin/python3 \"" + convert_script.string() +
                                        "\" --input \"" + onnx_path.string() + "\" --output \"" +
                                        fth_path.string() + "\"";
    ASSERT_EQ(std::system(convert_command.c_str()), 0);

    ModelLoader loader;
    ASSERT_TRUE(loader.Load(fth_path.string()));
    ASSERT_FALSE(loader.model().graph.nodes.empty());

    const auto* input_value = FindValueDesc(loader.model(), loader.model().graph.inputs.front());
    ASSERT_NE(input_value, nullptr);
    auto input_tensor = CreateDeterministicInputTensor(*input_value);
    ASSERT_NE(input_tensor, nullptr);
    ASSERT_TRUE(WriteTensorRaw(input_raw_path, *input_tensor));

    const std::string ref_command = "/home/jarvis/miniconda3/bin/python3 \"" + ref_script.string() +
                                    "\" --model \"" + onnx_path.string() + "\" --input-raw \"" +
                                    input_raw_path.string() + "\" --output-raw \"" +
                                    output_ref_path.string() + "\"";
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

    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor(loader.model().graph.outputs.front());
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 1000}));

    const auto actual = TensorToFloatVector(*output_tensor);
    ASSERT_EQ(actual.size(), reference.size());

    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        mean_abs_diff += diff;
    }
    mean_abs_diff /= static_cast<float>(actual.size());

    EXPECT_LE(max_abs_diff, 2e-3f) << case_name;
    EXPECT_LE(mean_abs_diff, 2e-4f) << case_name;
}

std::filesystem::path ResolveModelPath(const char* environment_name, const std::filesystem::path& default_path) {
    const char* value = std::getenv(environment_name);
    return value != nullptr && *value != '\0' ? std::filesystem::path(value) : default_path;
}

#ifdef FEATHER_WITH_CUDA

void VerifyClassificationGraphSelectsCudaKernels(const std::filesystem::path& fth_path) {
    if (!std::filesystem::exists(fth_path)) {
        GTEST_SKIP() << "converted classification model is not available: " << fth_path;
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.Load(fth_path.string()));
    ASSERT_EQ(loader.model().graph.inputs.size(), 1U);

    const auto* input_value = FindValueDesc(loader.model(), loader.model().graph.inputs.front());
    ASSERT_NE(input_value, nullptr);

    StaticGraph static_graph;
    static_graph.SetKernelDevice(DeviceType::CUDA);
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);
    ASSERT_EQ(static_graph.SetTensor(input_value->tensor.name, CreateDeterministicInputTensor(*input_value)), 0);
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    ASSERT_EQ(static_graph.Build(), 0);
    ASSERT_EQ(static_graph.ApplyPasses(), 0);
    RuntimeGraph runtime_graph;
    GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);

    for (const auto& static_node : static_graph.nodes()) {
        if (static_node.removed) {
            continue;
        }
        const auto* runtime_node = runtime_graph.GetNode(static_node.name);
        ASSERT_NE(runtime_node, nullptr) << static_node.name;
        EXPECT_EQ(runtime_node->kernel_device, DeviceType::CUDA)
            << "classification node unexpectedly fell back from CUDA: " << static_node.name
            << " (" << static_node.op_type << ")";
    }
}

#endif

}  // namespace

TEST(classification_real_models_test, RunsExportedResNet50OnCpu) {
    RunClassificationModelAgainstOnnxRuntime(
        ResolveModelPath("FEATHER_RESNET50_ONNX",
                         RepositoryRoot() / "models" / "classification" / "gluon_resnet50_v1b_Opset17.onnx"),
        "gluon_resnet50_v1b_Opset17");
}

TEST(classification_real_models_test, RunsExportedRepVggB0OnCpu) {
    RunClassificationModelAgainstOnnxRuntime(
        ResolveModelPath("FEATHER_REPVGG_ONNX",
                         RepositoryRoot() / "models" / "classification" / "repvgg_b0_Opset17.onnx"),
        "repvgg_b0_Opset17");
}

#ifdef FEATHER_WITH_CUDA
TEST(classification_real_models_test, ResNet50SelectsCudaKernelsWithoutHostFallback) {
    VerifyClassificationGraphSelectsCudaKernels(
        RepositoryRoot() / "models" / "classification" / "gluon_resnet50_v1b_Opset17.fth");
}

TEST(classification_real_models_test, RepVggB0SelectsCudaKernelsWithoutHostFallback) {
    VerifyClassificationGraphSelectsCudaKernels(
        RepositoryRoot() / "models" / "classification" / "repvgg_b0_Opset17.fth");
}
#endif
