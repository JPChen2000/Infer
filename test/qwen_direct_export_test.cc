#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

#include "core/graph_lowering.h"
#include "core/static_graph.h"
#include "model/model_io.h"
#include "util/types.h"

namespace {

std::shared_ptr<feather::Tensor> MakeDeclaredTensor(const feather::model::TensorDesc& desc) {
    const size_t bytes = static_cast<size_t>(desc.dims.empty() ? 1 : 1) *
                         static_cast<size_t>(std::max<int64_t>(1, [&desc]() {
                             int64_t count = 1;
                             for (const auto dim : desc.dims) {
                                 count *= dim;
                             }
                             return count;
                         }())) *
                         feather::DataTypeBytes(desc.data_type);
    auto tensor = std::make_shared<feather::Tensor>(bytes);
    tensor->Resize(desc.dims);
    tensor->set_data_type(desc.data_type);
    tensor->set_layout(desc.layout);
    return tensor;
}

}  // namespace

TEST(qwen_direct_export_test, LoadsDirectSafetensorsExportAndBuildsAtomicGraph) {
    const auto path = std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx8.fth");
    if (!std::filesystem::is_regular_file(path)) {
        GTEST_SKIP() << "direct Qwen FTH asset is not present";
    }

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(path.string()));
    const auto& model = loader.model();
    EXPECT_EQ(model.graph.inputs.size(), 51U);
    EXPECT_EQ(model.graph.outputs.size(), 49U);
    EXPECT_EQ(model.graph.nodes.size(), 2613U);

    feather::StaticGraph graph;
    graph.SetKernelDevice(feather::DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    for (const auto& value : model.graph.values) {
        if (value.constant) {
            ASSERT_NE(loader.CreateWeightTensor(value.tensor.name), nullptr) << value.tensor.name;
            ASSERT_EQ(graph.SetTensor(value.tensor.name, loader.CreateWeightTensor(value.tensor.name)), 0);
        }
    }
    for (const auto& input_name : model.graph.inputs) {
        const auto it = std::find_if(model.graph.values.begin(), model.graph.values.end(),
                                     [&input_name](const feather::model::ValueDesc& value) {
                                         return value.tensor.name == input_name;
                                     });
        ASSERT_NE(it, model.graph.values.end()) << input_name;
        ASSERT_EQ(graph.SetTensor(input_name, MakeDeclaredTensor(it->tensor)), 0) << input_name;
    }

    EXPECT_EQ(graph.Build(), 0);
    EXPECT_EQ(graph.NodeSize(), model.graph.nodes.size());
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_direct_export_test, LowersEveryQwenDecodeNodeToCuda) {
    const auto path = std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx128.fth");
    if (!std::filesystem::is_regular_file(path)) {
        GTEST_SKIP() << "context-128 Qwen FTH asset is not present";
    }

    feather::model::ModelLoader loader;
    ASSERT_TRUE(loader.Load(path.string()));
    feather::StaticGraph graph;
    graph.SetKernelDevice(feather::DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(loader.model()), 0);
    for (const auto& value : loader.model().graph.values) {
        if (value.constant) {
            ASSERT_EQ(graph.SetTensor(value.tensor.name, loader.CreateWeightTensor(value.tensor.name)), 0);
        }
    }
    for (const auto& input_name : loader.model().graph.inputs) {
        const auto it = std::find_if(loader.model().graph.values.begin(), loader.model().graph.values.end(),
                                     [&input_name](const feather::model::ValueDesc& value) {
                                         return value.tensor.name == input_name;
                                     });
        ASSERT_NE(it, loader.model().graph.values.end()) << input_name;
        ASSERT_EQ(graph.SetTensor(input_name, MakeDeclaredTensor(it->tensor)), 0) << input_name;
    }
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    feather::RuntimeGraph runtime_graph;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime_graph), 0);
    for (const auto& node : graph.nodes()) {
        if (node.removed) {
            continue;
        }
        const auto* runtime_node = runtime_graph.GetNode(node.name);
        ASSERT_NE(runtime_node, nullptr) << node.name;
        EXPECT_EQ(runtime_node->kernel_device, feather::DeviceType::CUDA)
            << "Qwen node fell back from CUDA: " << node.name << " (" << node.op->type() << ')';
    }
}
#endif
