#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/operator_registry.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_io.h"

namespace {

std::filesystem::path RepositoryRoot() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path ResolveModelPath(const char* environment_name, const std::filesystem::path& default_path) {
    const char* value = std::getenv(environment_name);
    return value != nullptr && *value != '\0' ? std::filesystem::path(value) : default_path;
}

void PrintFirstBuildFailure(const feather::model::ModelDesc& model,
                            feather::OperatorRegistry::TensorMap tensors) {
    feather::KernelDeviceScope scope(feather::DeviceType::X86);
    for (const auto& node : model.graph.nodes) {
        auto op = feather::OperatorRegistry::instance().Create(node, tensors);
        if (op == nullptr) {
            std::cerr << "failed node name=" << node.name << " op=" << node.op_type << " inputs=";
            for (const auto& input : node.inputs) {
                std::cerr << input << "[";
                auto it = tensors.find(input);
                if (it != tensors.end() && it->second != nullptr) {
                    for (size_t i = 0; i < it->second->dims().size(); ++i) {
                        std::cerr << (i == 0 ? "" : ",") << it->second->dims()[i];
                    }
                    std::cerr << " dtype=" << static_cast<int>(it->second->data_type());
                }
                std::cerr << "] ";
            }
            std::cerr << std::endl;
            return;
        }
        const auto& outputs = op->outputs();
        for (size_t i = 0; i < node.outputs.size() && i < outputs.size(); ++i) {
            tensors[node.outputs[i]] = outputs[i];
        }
    }
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

std::vector<float> TensorToFloatVector(const feather::Tensor& tensor) {
    std::vector<float> data(static_cast<size_t>(tensor.numel()), 0.0f);
    if (tensor.data_type() != feather::DataType::FP32) {
        return {};
    }
    std::memcpy(data.data(), tensor.data<float>(), data.size() * sizeof(float));
    return data;
}

size_t CountNodes(const feather::model::ModelDesc& model, const std::string& op_type) {
    size_t count = 0;
    for (const auto& node : model.graph.nodes) {
        if (node.op_type == op_type) {
            ++count;
        }
    }
    return count;
}

const feather::model::NodeDesc* FindFirstNode(const feather::model::ModelDesc& model, const std::string& op_type) {
    for (const auto& node : model.graph.nodes) {
        if (node.op_type == op_type) {
            return &node;
        }
    }
    return nullptr;
}

void RunTinyBertAgainstOnnxRuntime(const std::filesystem::path& onnx_path) {
    if (!std::filesystem::exists(onnx_path)) {
        GTEST_SKIP() << "exported model is not available: " << onnx_path;
    }

    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto convert_script = repo_root / "tools" / "onnx_to_feather.py";
    const auto ref_script = repo_root / "tools" / "onnx_reference.py";
    const auto fth_path = onnx_path.parent_path() / "tiny_random_bert_static.fth";
    const auto input_ids_raw_path = std::filesystem::temp_directory_path() / "tiny_random_bert_static_input_ids.raw";
    const auto attention_mask_raw_path = std::filesystem::temp_directory_path() / "tiny_random_bert_static_attention_mask.raw";
    const auto token_type_ids_raw_path = std::filesystem::temp_directory_path() / "tiny_random_bert_static_token_type_ids.raw";
    const auto output_ref_path = std::filesystem::temp_directory_path() / "tiny_random_bert_static_ort.raw";

    const std::string convert_command = "/home/jarvis/miniconda3/bin/python3 \"" + convert_script.string() +
                                        "\" --input \"" + onnx_path.string() + "\" --output \"" +
                                        fth_path.string() + "\"";
    ASSERT_EQ(std::system(convert_command.c_str()), 0);

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(fth_path.string()));

    auto input_ids = std::make_shared<feather::Tensor>();
    input_ids->Assign<int64_t>({2, 5, 9, 17, 4, 3, 8, 11}, {1, 8});
    auto attention_mask = std::make_shared<feather::Tensor>();
    attention_mask->Assign<int64_t>({1, 1, 1, 1, 1, 1, 1, 1}, {1, 8});
    auto token_type_ids = std::make_shared<feather::Tensor>();
    token_type_ids->Assign<int64_t>({0, 0, 0, 0, 0, 0, 0, 0}, {1, 8});

    ASSERT_TRUE(std::filesystem::exists(ref_script));
    ASSERT_TRUE(std::filesystem::exists(onnx_path));
    ASSERT_TRUE(std::filesystem::exists(convert_script));

    auto write_raw = [](const std::filesystem::path& path, const feather::Tensor& tensor) {
        std::ofstream raw(path, std::ios::binary | std::ios::trunc);
        if (!raw.good()) {
            return false;
        }
        raw.write(static_cast<const char*>(tensor.raw_data()), static_cast<std::streamsize>(tensor.memory_size()));
        raw.close();
        return raw.good();
    };
    ASSERT_TRUE(write_raw(input_ids_raw_path, *input_ids));
    ASSERT_TRUE(write_raw(attention_mask_raw_path, *attention_mask));
    ASSERT_TRUE(write_raw(token_type_ids_raw_path, *token_type_ids));

    const std::string ref_command = "/home/jarvis/miniconda3/bin/python3 \"" + ref_script.string() +
                                    "\" --model \"" + onnx_path.string() + "\" --input-raw \"" +
                                    input_ids_raw_path.string() + "\" --input-raw \"" +
                                    attention_mask_raw_path.string() + "\" --input-raw \"" +
                                    token_type_ids_raw_path.string() + "\" --output-raw \"" +
                                    output_ref_path.string() + "\"";
    ASSERT_EQ(std::system(ref_command.c_str()), 0);
    const auto reference = ReadFloatVectorRaw(output_ref_path);
    ASSERT_FALSE(reference.empty());

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);
    ASSERT_EQ(static_graph.SetTensor("input_ids", input_ids), 0);
    ASSERT_EQ(static_graph.SetTensor("attention_mask", attention_mask), 0);
    ASSERT_EQ(static_graph.SetTensor("token_type_ids", token_type_ids), 0);
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    const int32_t build_status = static_graph.Build();
    if (build_status != 0) {
        feather::OperatorRegistry::TensorMap tensors = static_graph.tensors();
        for (const auto& value : loader.model().graph.values) {
            if (tensors.count(value.tensor.name) != 0 || value.constant) {
                continue;
            }
            auto tensor = std::make_shared<feather::Tensor>(value.tensor.dims);
            tensor->set_data_type(value.tensor.data_type);
            tensors[value.tensor.name] = tensor;
        }
        PrintFirstBuildFailure(loader.model(), tensors);
    }
    ASSERT_EQ(build_status, 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("last_hidden_state");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 8, 32}));

    const auto actual = TensorToFloatVector(*output_tensor);
    ASSERT_FALSE(actual.empty());
    ASSERT_EQ(actual.size(), reference.size());
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        mean_abs_diff += diff;
    }
    mean_abs_diff /= static_cast<float>(actual.size());
    EXPECT_LE(max_abs_diff, 2e-3f);
    EXPECT_LE(mean_abs_diff, 2e-4f);
}

void RunBertTinySpamAgainstOnnxRuntime(const std::filesystem::path& onnx_path) {
    if (!std::filesystem::exists(onnx_path)) {
        GTEST_SKIP() << "exported model is not available: " << onnx_path;
    }

    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto convert_script = repo_root / "tools" / "onnx_to_feather.py";
    const auto ref_script = repo_root / "tools" / "onnx_reference.py";
    const auto fth_path = onnx_path.parent_path() / "bert_tiny_spam_static.fth";
    const auto input_ids_raw_path = std::filesystem::temp_directory_path() / "bert_tiny_spam_input_ids.raw";
    const auto attention_mask_raw_path = std::filesystem::temp_directory_path() / "bert_tiny_spam_attention_mask.raw";
    const auto token_type_ids_raw_path = std::filesystem::temp_directory_path() / "bert_tiny_spam_token_type_ids.raw";
    const auto output_ref_path = std::filesystem::temp_directory_path() / "bert_tiny_spam_ort.raw";

    const std::string convert_command = "/home/jarvis/miniconda3/bin/python3 \"" + convert_script.string() +
                                        "\" --input \"" + onnx_path.string() + "\" --output \"" +
                                        fth_path.string() + "\"";
    ASSERT_EQ(std::system(convert_command.c_str()), 0);

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(fth_path.string()));

    auto input_ids = std::make_shared<feather::Tensor>();
    input_ids->Assign<int64_t>({101, 23156, 999, 2017, 2031, 2180, 1037, 102}, {1, 8});
    auto attention_mask = std::make_shared<feather::Tensor>();
    attention_mask->Assign<int64_t>({1, 1, 1, 1, 1, 1, 1, 1}, {1, 8});
    auto token_type_ids = std::make_shared<feather::Tensor>();
    token_type_ids->Assign<int64_t>({0, 0, 0, 0, 0, 0, 0, 0}, {1, 8});

    auto write_raw = [](const std::filesystem::path& path, const feather::Tensor& tensor) {
        std::ofstream raw(path, std::ios::binary | std::ios::trunc);
        if (!raw.good()) {
            return false;
        }
        raw.write(static_cast<const char*>(tensor.raw_data()), static_cast<std::streamsize>(tensor.memory_size()));
        raw.close();
        return raw.good();
    };
    ASSERT_TRUE(write_raw(input_ids_raw_path, *input_ids));
    ASSERT_TRUE(write_raw(attention_mask_raw_path, *attention_mask));
    ASSERT_TRUE(write_raw(token_type_ids_raw_path, *token_type_ids));

    const std::string ref_command = "/home/jarvis/miniconda3/bin/python3 \"" + ref_script.string() +
                                    "\" --model \"" + onnx_path.string() + "\" --input-raw \"" +
                                    input_ids_raw_path.string() + "\" --input-raw \"" +
                                    attention_mask_raw_path.string() + "\" --input-raw \"" +
                                    token_type_ids_raw_path.string() + "\" --output-raw \"" +
                                    output_ref_path.string() + "\"";
    ASSERT_EQ(std::system(ref_command.c_str()), 0);
    const auto reference = ReadFloatVectorRaw(output_ref_path);
    ASSERT_EQ(reference.size(), 2U);

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);
    ASSERT_EQ(static_graph.SetTensor("input_ids", input_ids), 0);
    ASSERT_EQ(static_graph.SetTensor("attention_mask", attention_mask), 0);
    ASSERT_EQ(static_graph.SetTensor("token_type_ids", token_type_ids), 0);
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    const int32_t build_status = static_graph.Build();
    if (build_status != 0) {
        feather::OperatorRegistry::TensorMap tensors = static_graph.tensors();
        for (const auto& value : loader.model().graph.values) {
            if (tensors.count(value.tensor.name) != 0 || value.constant) {
                continue;
            }
            auto tensor = std::make_shared<feather::Tensor>(value.tensor.dims);
            tensor->set_data_type(value.tensor.data_type);
            tensors[value.tensor.name] = tensor;
        }
        PrintFirstBuildFailure(loader.model(), tensors);
    }
    ASSERT_EQ(build_status, 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("logits");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 2}));

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
    EXPECT_LE(max_abs_diff, 2e-3f);
    EXPECT_LE(mean_abs_diff, 2e-4f);
}

void RunDistilBertSst2AgainstOnnxRuntime(const std::filesystem::path& onnx_path) {
    if (!std::filesystem::exists(onnx_path)) {
        GTEST_SKIP() << "exported model is not available: " << onnx_path;
    }

    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto convert_script = repo_root / "tools" / "onnx_to_feather.py";
    const auto ref_script = repo_root / "tools" / "onnx_reference.py";
    const auto fth_path = onnx_path.parent_path() / "distilbert_sst2_static.fth";
    const auto input_ids_raw_path = std::filesystem::temp_directory_path() / "distilbert_sst2_input_ids.raw";
    const auto attention_mask_raw_path = std::filesystem::temp_directory_path() / "distilbert_sst2_attention_mask.raw";
    const auto output_ref_path = std::filesystem::temp_directory_path() / "distilbert_sst2_ort.raw";

    const std::string convert_command = "/home/jarvis/miniconda3/bin/python3 \"" + convert_script.string() +
                                        "\" --input \"" + onnx_path.string() + "\" --output \"" +
                                        fth_path.string() + "\"";
    ASSERT_EQ(std::system(convert_command.c_str()), 0);

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(fth_path.string()));
    EXPECT_EQ(CountNodes(loader.model(), "Equal"), 1U);
    EXPECT_EQ(CountNodes(loader.model(), "Shape"), 6U);
    EXPECT_EQ(CountNodes(loader.model(), "Expand"), 6U);
    EXPECT_EQ(CountNodes(loader.model(), "Cast"), 6U);
    EXPECT_EQ(CountNodes(loader.model(), "Where"), 6U);
    EXPECT_EQ(CountNodes(loader.model(), "LayerNormalization"), 0U);
    EXPECT_GT(CountNodes(loader.model(), "MatMul"), 0U);
    EXPECT_GT(CountNodes(loader.model(), "ReduceMean"), 0U);
    EXPECT_GT(CountNodes(loader.model(), "Sqrt"), 0U);

    const auto* reshape = FindFirstNode(loader.model(), "Reshape");
    ASSERT_NE(reshape, nullptr);
    ASSERT_EQ(reshape->inputs.size(), 2U);
    EXPECT_FALSE(reshape->inputs[0].empty());
    EXPECT_FALSE(reshape->inputs[1].empty());

    const auto* pow = FindFirstNode(loader.model(), "Pow");
    ASSERT_NE(pow, nullptr);
    ASSERT_EQ(pow->inputs.size(), 2U);
    EXPECT_FALSE(pow->inputs[0].empty());
    EXPECT_FALSE(pow->inputs[1].empty());

    auto input_ids = std::make_shared<feather::Tensor>();
    input_ids->Assign<int64_t>({101, 2023, 2003, 2019, 14153, 3048, 0, 0}, {1, 8});
    auto attention_mask = std::make_shared<feather::Tensor>();
    attention_mask->Assign<int64_t>({1, 1, 1, 1, 1, 1, 0, 0}, {1, 8});

    auto write_raw = [](const std::filesystem::path& path, const feather::Tensor& tensor) {
        std::ofstream raw(path, std::ios::binary | std::ios::trunc);
        if (!raw.good()) {
            return false;
        }
        raw.write(static_cast<const char*>(tensor.raw_data()), static_cast<std::streamsize>(tensor.memory_size()));
        raw.close();
        return raw.good();
    };
    ASSERT_TRUE(write_raw(input_ids_raw_path, *input_ids));
    ASSERT_TRUE(write_raw(attention_mask_raw_path, *attention_mask));

    const std::string ref_command = "/home/jarvis/miniconda3/bin/python3 \"" + ref_script.string() +
                                    "\" --model \"" + onnx_path.string() + "\" --input-raw \"" +
                                    input_ids_raw_path.string() + "\" --input-raw \"" +
                                    attention_mask_raw_path.string() + "\" --output-raw \"" +
                                    output_ref_path.string() + "\"";
    ASSERT_EQ(std::system(ref_command.c_str()), 0);
    const auto reference = ReadFloatVectorRaw(output_ref_path);
    ASSERT_EQ(reference.size(), 2U);

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);
    ASSERT_EQ(static_graph.SetTensor("input_ids", input_ids), 0);
    ASSERT_EQ(static_graph.SetTensor("attention_mask", attention_mask), 0);
    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    const int32_t build_status = static_graph.Build();
    if (build_status != 0) {
        feather::OperatorRegistry::TensorMap tensors = static_graph.tensors();
        for (const auto& value : loader.model().graph.values) {
            if (tensors.count(value.tensor.name) != 0 || value.constant) {
                continue;
            }
            auto tensor = std::make_shared<feather::Tensor>(value.tensor.dims);
            tensor->set_data_type(value.tensor.data_type);
            tensors[value.tensor.name] = tensor;
        }
        PrintFirstBuildFailure(loader.model(), tensors);
    }
    ASSERT_EQ(build_status, 0);

    feather::RuntimeGraph runtime_graph;
    runtime_graph.SetThreadMode(feather::RuntimeThreadMode::kSerialGraph);
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto output_tensor = runtime_graph.GetTensor("logits");
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->dims().data(), std::vector<int64_t>({1, 2}));

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
    EXPECT_LE(max_abs_diff, 2e-3f);
    EXPECT_LE(mean_abs_diff, 2e-4f);
}

}  // namespace

TEST(transformer_real_models_test, RunsTinyRandomBertStaticOnCpu) {
    RunTinyBertAgainstOnnxRuntime(
        ResolveModelPath("FEATHER_TINY_BERT_ONNX",
                         RepositoryRoot() / "models" / "transformer" / "tiny_random_bert_static.onnx"));
}

TEST(transformer_real_models_test, RunsFineTunedBertTinySpamClassifierOnCpu) {
    RunBertTinySpamAgainstOnnxRuntime(
        ResolveModelPath("FEATHER_BERT_TINY_SPAM_ONNX",
                         RepositoryRoot() / "models" / "transformer" / "bert_tiny_spam_static.onnx"));
}

TEST(transformer_real_models_test, RunsDistilBertSst2AttentionMaskOnCpu) {
    RunDistilBertSst2AgainstOnnxRuntime(
        ResolveModelPath("FEATHER_DISTILBERT_SST2_ONNX",
                         RepositoryRoot() / "models" / "transformer" / "distilbert_sst2_static.onnx"));
}
