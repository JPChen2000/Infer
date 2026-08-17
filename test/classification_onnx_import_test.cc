#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/graph.h"
#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "core/tensor.h"
#include "model/model_io.h"

namespace {

bool WriteTinyClassificationOnnxModel(const std::filesystem::path& model_path) {
    const auto script_path = std::filesystem::temp_directory_path() / "feather_tiny_classification_model.py";
    std::ofstream script(script_path, std::ios::trunc);
    if (!script.good()) {
        return false;
    }
    script << R"PY(
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

output_path = sys.argv[1]

input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])
output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])

conv_w = numpy_helper.from_array(np.array([[[[1.0]]], [[[2.0]]]], dtype=np.float32), name="conv.weight")
conv_b = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="conv.bias")
bn_scale = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="bn.scale")
bn_bias = numpy_helper.from_array(np.array([0.0, -1.0], dtype=np.float32), name="bn.bias")
bn_mean = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="bn.mean")
bn_var = numpy_helper.from_array(np.array([1.0, 4.0], dtype=np.float32), name="bn.var")
fc_w = numpy_helper.from_array(np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32), name="fc.weight")
fc_b = numpy_helper.from_array(np.array([0.25, -0.5], dtype=np.float32), name="fc.bias")

nodes = [
    helper.make_node("Conv", ["input", "conv.weight", "conv.bias"], ["conv_out"],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1], name="conv0"),
    helper.make_node("BatchNormalization",
                     ["conv_out", "bn.scale", "bn.bias", "bn.mean", "bn.var"],
                     ["bn_out"], epsilon=1e-5, name="bn0"),
    helper.make_node("Relu", ["bn_out"], ["relu_out"], name="relu0"),
    helper.make_node("GlobalAveragePool", ["relu_out"], ["gap_out"], name="gap0"),
    helper.make_node("Flatten", ["gap_out"], ["flat_out"], axis=1, name="flatten0"),
    helper.make_node("Gemm", ["flat_out", "fc.weight", "fc.bias"], ["logits"],
                     transB=1, alpha=1.0, beta=1.0, name="gemm0"),
]

graph = helper.make_graph(nodes, "tiny_classification_graph", [input_info], [output_info],
                          [conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var, fc_w, fc_b])
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, output_path)
)PY";
    script.close();
    if (!script.good()) {
        return false;
    }
    const std::string command =
        "/home/jarvis/miniconda3/bin/python3 \"" + script_path.string() + "\" \"" + model_path.string() + "\"";
    return std::system(command.c_str()) == 0;
}

}  // namespace

TEST(classification_onnx_import_test, PreservesClassificationOnnxOperatorNames) {
    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto script_path = repo_root / "tools" / "onnx_to_feather.py";
    const auto onnx_path = std::filesystem::temp_directory_path() / "feather_tiny_classification.onnx";
    const auto output_path = std::filesystem::temp_directory_path() / "feather_tiny_classification.fth";

    ASSERT_TRUE(WriteTinyClassificationOnnxModel(onnx_path));
    const std::string command = "/home/jarvis/miniconda3/bin/python3 \"" + script_path.string() + "\" --input \"" +
                                onnx_path.string() + "\" --output \"" + output_path.string() + "\"";
    ASSERT_EQ(std::system(command.c_str()), 0);

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(output_path.string()));

    std::unordered_set<std::string> op_types;
    for (const auto& node : loader.model().graph.nodes) {
        op_types.insert(node.op_type);
    }
    EXPECT_TRUE(op_types.count("Conv"));
    EXPECT_TRUE(op_types.count("BatchNormalization"));
    EXPECT_TRUE(op_types.count("GlobalAveragePool"));
    EXPECT_TRUE(op_types.count("Flatten"));
    EXPECT_TRUE(op_types.count("Gemm"));
}

TEST(classification_onnx_import_test, RunsTinyClassificationGraphOnCpu) {
    const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto script_path = repo_root / "tools" / "onnx_to_feather.py";
    const auto onnx_path = std::filesystem::temp_directory_path() / "feather_tiny_classification_run.onnx";
    const auto output_path = std::filesystem::temp_directory_path() / "feather_tiny_classification_run.fth";

    ASSERT_TRUE(WriteTinyClassificationOnnxModel(onnx_path));
    const std::string command = "/home/jarvis/miniconda3/bin/python3 \"" + script_path.string() + "\" --input \"" +
                                onnx_path.string() + "\" --output \"" + output_path.string() + "\"";
    ASSERT_EQ(std::system(command.c_str()), 0);

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(output_path.string()));

    feather::StaticGraph static_graph;
    ASSERT_EQ(static_graph.SetModel(loader.model()), 0);

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2});
    ASSERT_EQ(static_graph.SetTensor("input", input), 0);

    for (const auto& value : loader.model().graph.values) {
        if (!value.constant) {
            continue;
        }
        auto tensor = loader.CreateWeightTensor(value.tensor.name);
        ASSERT_NE(tensor, nullptr) << value.tensor.name;
        ASSERT_EQ(static_graph.SetTensor(value.tensor.name, tensor), 0);
    }

    ASSERT_EQ(static_graph.Build(), 0);

    feather::RuntimeGraph runtime_graph;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(static_graph, &runtime_graph), 0);
    ASSERT_EQ(runtime_graph.Run(), 0);

    auto logits = runtime_graph.GetTensor("logits");
    ASSERT_NE(logits, nullptr);
    EXPECT_EQ(logits->dims().data(), std::vector<int64_t>({1, 2}));
    ASSERT_EQ(logits->data_type(), feather::DataType::FP32);
    EXPECT_NEAR(logits->data<float>()[0], -1.2500064f, 1e-4f);
    EXPECT_NEAR(logits->data<float>()[1], 8.749982f, 1e-4f);
}
