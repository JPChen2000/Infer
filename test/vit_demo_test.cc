#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "demo/classification_runner.h"
#include "model/model_io.h"

namespace {

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

std::shared_ptr<feather::Tensor> MakeTensor(const std::vector<float>& values,
                                            const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::filesystem::path WriteVitShapeModel() {
    feather::model::ModelDesc model;
    model.name = "vit_shape_contract";
    model.version = 1;
    model.graph.name = "vit_shape_contract_graph";
    model.graph.inputs = {"pixel_values"};
    model.graph.outputs = {"logits"};
    model.graph.values = {
        MakeValue("pixel_values", {1, 3, 224, 224}, feather::DataType::FP32, feather::DataLayout::NCHW, false),
        MakeValue("pooled", {1, 3, 1, 1}, feather::DataType::FP32, feather::DataLayout::NCHW, false),
        MakeValue("flat", {1, 3}, feather::DataType::FP32, feather::DataLayout::ND, false),
        MakeValue("weight", {3, 3}, feather::DataType::FP32, feather::DataLayout::ND, true),
        MakeValue("bias", {3}, feather::DataType::FP32, feather::DataLayout::ND, true),
        MakeValue("logits", {1, 3}, feather::DataType::FP32, feather::DataLayout::ND, false),
    };

    feather::model::NodeDesc pool;
    pool.name = "pool";
    pool.op_type = "GlobalAveragePool";
    pool.inputs = {"pixel_values"};
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

    const auto path = std::filesystem::temp_directory_path() / "vit_shape_contract.fth";
    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["weight"] = MakeTensor({1.0f, 0.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f},
                                    {3, 3});
    weights["bias"] = MakeTensor({0.0f, 1.0f, 2.0f}, {3});

    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

#ifdef FEATHER_WITH_CUDA

bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::filesystem::path RepositoryRoot() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

#endif

}  // namespace

TEST(vit_demo_test, ReusesClassificationRunnerForVitInputContract) {
    const auto model_path = WriteVitShapeModel();
    feather::demo::ClassificationRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::ClassificationBackend::kCommon), 0)
        << runner.LastError();
    EXPECT_NE(runner.DescribeLastBuild().find("input=pixel_values"), std::string::npos);
    EXPECT_NE(runner.DescribeLastBuild().find("input_shape=[1,3,224,224]"), std::string::npos);
}

#ifdef FEATHER_WITH_CUDA
TEST(vit_demo_test, RealVitCudaDemoProducesFiniteClassificationScores) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const auto model_path = RepositoryRoot() / "models" / "vit" / "vit_base_patch16_224_static.fth";
    const auto image_path = RepositoryRoot() / "third_party" / "test_images" / "bus.jpg";
    if (!std::filesystem::exists(model_path) || !std::filesystem::exists(image_path)) {
        GTEST_SKIP() << "real ViT model or test image is not available";
    }

    feather::demo::ClassificationRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::ClassificationBackend::kCuda), 0)
        << runner.LastError();

    std::vector<feather::demo::ClassificationResult> results;
    ASSERT_EQ(runner.Run(image_path.string(), feather::demo::ImageNetPreprocessConfig(), 5, &results), 0)
        << runner.LastError();
    ASSERT_EQ(results.size(), 5U);
    for (const auto& result : results) {
        EXPECT_TRUE(std::isfinite(result.logit));
        EXPECT_TRUE(std::isfinite(result.probability));
    }
}
#endif
