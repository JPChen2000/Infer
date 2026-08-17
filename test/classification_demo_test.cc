#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "demo/classification_runner.h"
#include "demo/image_io.h"
#include "model/model_io.h"

namespace {

std::filesystem::path WriteTestPpmImage(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "P6\n2 2\n255\n";
    const std::vector<unsigned char> pixels = {
        255, 0,   0,   0,   255, 0,
        0,   0, 255, 255, 255, 255,
    };
    out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    return path;
}

std::shared_ptr<feather::Tensor> MakeTensor(const std::vector<float>& values,
                                            const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

feather::model::ValueDesc MakeValue(const std::string& name, const std::vector<int64_t>& dims,
                                    feather::DataType dtype, feather::DataLayout layout, bool constant) {
    feather::model::ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = dtype;
    value.tensor.layout = layout;
    value.constant = constant;
    return value;
}

bool WriteTinyClassificationModel(const std::filesystem::path& path) {
    feather::model::ModelDesc model;
    model.name = "tiny_classification_demo";
    model.version = 1;
    model.graph.name = "tiny_classification_demo_graph";
    model.graph.inputs = {"input"};
    model.graph.outputs = {"logits"};
    model.graph.values = {
        MakeValue("input", {1, 3, 2, 2}, feather::DataType::FP32, feather::DataLayout::NCHW, false),
        MakeValue("pooled", {1, 3, 1, 1}, feather::DataType::FP32, feather::DataLayout::NCHW, false),
        MakeValue("flat", {1, 3}, feather::DataType::FP32, feather::DataLayout::ND, false),
        MakeValue("weight", {3, 3}, feather::DataType::FP32, feather::DataLayout::ND, true),
        MakeValue("bias", {3}, feather::DataType::FP32, feather::DataLayout::ND, true),
        MakeValue("logits", {1, 3}, feather::DataType::FP32, feather::DataLayout::ND, false),
    };

    feather::model::NodeDesc pool;
    pool.name = "pool";
    pool.op_type = "GlobalAveragePool";
    pool.inputs = {"input"};
    pool.outputs = {"pooled"};

    feather::model::NodeDesc flatten;
    flatten.name = "flatten";
    flatten.op_type = "Flatten";
    flatten.inputs = {"pooled"};
    flatten.outputs = {"flat"};
    flatten.attributes["axis"] = int64_t{1};

    feather::model::NodeDesc gemm;
    gemm.name = "classifier";
    gemm.op_type = "Gemm";
    gemm.inputs = {"flat", "weight", "bias"};
    gemm.outputs = {"logits"};
    gemm.attributes["alpha"] = 1.0f;
    gemm.attributes["beta"] = 1.0f;
    gemm.attributes["transA"] = int64_t{0};
    gemm.attributes["transB"] = int64_t{0};
    model.graph.nodes = {pool, flatten, gemm};

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["weight"] = MakeTensor({
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    }, {3, 3});
    weights["bias"] = MakeTensor({0.25f, 1.0f, 2.0f}, {3});

    feather::model::ModelWriter writer;
    return writer.Save(path.string(), model, weights);
}

}  // namespace

TEST(classification_demo_test, PreprocessImageNetWritesNormalizedNchwTensor) {
    feather::demo::ImageData image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0,   0,   0,   255, 0,
        0,   0, 255, 255, 255, 255,
    };

    feather::demo::ImageNetPreprocessConfig config;
    config.input_size = 2;
    config.resize_shorter_side = 2;
    config.mean = {0.0f, 0.0f, 0.0f};
    config.std = {1.0f, 1.0f, 1.0f};

    feather::Tensor tensor({1, 3, 2, 2});
    tensor.set_layout(feather::DataLayout::NCHW);
    ASSERT_EQ(feather::demo::PreprocessImageNetToTensor(image, 2, 2, config, feather::DataType::FP32, &tensor), 0);

    EXPECT_EQ(tensor.dims().data(), std::vector<int64_t>({1, 3, 2, 2}));
    ASSERT_EQ(tensor.data_type(), feather::DataType::FP32);
    const auto* values = tensor.data<float>();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[3], 1.0f);
    EXPECT_FLOAT_EQ(values[4], 0.0f);
    EXPECT_FLOAT_EQ(values[5], 1.0f);
    EXPECT_FLOAT_EQ(values[6], 0.0f);
    EXPECT_FLOAT_EQ(values[7], 1.0f);
    EXPECT_FLOAT_EQ(values[8], 0.0f);
    EXPECT_FLOAT_EQ(values[9], 0.0f);
    EXPECT_FLOAT_EQ(values[10], 1.0f);
    EXPECT_FLOAT_EQ(values[11], 1.0f);
}

TEST(classification_demo_test, RunsPreparedClassificationModelOnCpuAndRanksTopK) {
    const auto model_path = std::filesystem::temp_directory_path() / "tiny_classification_demo.fth";
    const auto image_path = WriteTestPpmImage(std::filesystem::temp_directory_path() / "tiny_classification_demo.ppm");
    ASSERT_TRUE(WriteTinyClassificationModel(model_path));

    feather::demo::ClassificationRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::ClassificationBackend::kX86), 0);
    EXPECT_NE(runner.DescribeLastBuild().find("backend=x86"), std::string::npos);
    EXPECT_NE(runner.DescribeLastBuild().find("static_nodes="), std::string::npos);
    EXPECT_NE(runner.DescribeLastBuild().find("runtime_nodes="), std::string::npos);

    feather::demo::ImageNetPreprocessConfig config;
    config.input_size = 2;
    config.resize_shorter_side = 2;
    config.mean = {0.0f, 0.0f, 0.0f};
    config.std = {1.0f, 1.0f, 1.0f};

    std::vector<feather::demo::ClassificationResult> results;
    ASSERT_EQ(runner.Run(image_path.string(), config, 2, &results), 0);
    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].class_id, 2);
    EXPECT_EQ(results[1].class_id, 1);
    EXPECT_GT(results[0].probability, results[1].probability);
    EXPECT_NEAR(results[0].logit, 2.0f, 1e-6f);
    EXPECT_NEAR(results[1].logit, 1.0f, 1e-6f);
    EXPECT_NE(runner.DescribeLastRun().find("preprocess_ms="), std::string::npos);
    EXPECT_NE(runner.DescribeLastRun().find("rungraph_ms="), std::string::npos);
}

TEST(classification_demo_test, ParsesClassificationBackends) {
    feather::demo::ClassificationBackend backend = feather::demo::ClassificationBackend::kHost;

    ASSERT_TRUE(feather::demo::ParseClassificationBackend("host", &backend));
    EXPECT_EQ(backend, feather::demo::ClassificationBackend::kHost);
    ASSERT_TRUE(feather::demo::ParseClassificationBackend("common", &backend));
    EXPECT_EQ(backend, feather::demo::ClassificationBackend::kCommon);
    ASSERT_TRUE(feather::demo::ParseClassificationBackend("X86", &backend));
    EXPECT_EQ(backend, feather::demo::ClassificationBackend::kX86);
    ASSERT_TRUE(feather::demo::ParseClassificationBackend("cuda", &backend));
    EXPECT_EQ(backend, feather::demo::ClassificationBackend::kCuda);
}
