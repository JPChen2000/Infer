#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "demo/qwen_runner.h"
#include "model/model_io.h"

namespace {

feather::model::ValueDesc MakeValue(const std::string& name, const std::vector<int64_t>& dims,
                                    feather::DataType data_type, bool constant = false) {
    feather::model::ValueDesc value;
    value.tensor.name = name;
    value.tensor.dims = dims;
    value.tensor.data_type = data_type;
    value.constant = constant;
    return value;
}

std::shared_ptr<feather::Tensor> MakeFloatTensor(const std::vector<float>& values, const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::filesystem::path WriteAutoregressiveFixture() {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_fixture";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 4}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1}, feather::DataType::FP32),
        MakeValue("lookup", {4, 4}, feather::DataType::FP32, true),
        MakeValue("logits", {1, 1, 4}, feather::DataType::FP32),
        MakeValue("next_recurrent_state_0", {1}, feather::DataType::FP32),
    };

    feather::model::NodeDesc gather;
    gather.name = "lookup_logits";
    gather.op_type = "Gather";
    gather.inputs = {"lookup", "token_ids"};
    gather.outputs = {"logits"};
    gather.attributes["axis"] = int64_t{0};

    feather::model::NodeDesc state;
    state.name = "next_state";
    state.op_type = "Identity";
    state.inputs = {"recurrent_state_0"};
    state.outputs = {"next_recurrent_state_0"};
    model.graph.nodes = {gather, state};

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["lookup"] = MakeFloatTensor({0.0f, 5.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f, 5.0f,
                                          0.0f, 0.0f, 5.0f, 0.0f,
                                          0.0f, 0.0f, 5.0f, 0.0f},
                                         {4, 4});
    const auto path = std::filesystem::temp_directory_path() / "qwen_runner_fixture.fth";
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

}  // namespace

TEST(qwen_runner_test, GreedilyGeneratesTokensAndPersistsExplicitState) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 3, {2}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(runner.TokensProcessed(), 4);

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
    EXPECT_EQ(runner.TokensProcessed(), 2);
    ASSERT_EQ(runner.Consume({2}), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 3);

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    EXPECT_EQ(runner.TokensProcessed(), 0);
    ASSERT_EQ(runner.Generate({0}, 3, {2}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 3, 2}));
}

TEST(qwen_runner_test, LoadsRealDirectSafetensorsExportWithExplicitStateBindings) {
    const auto model_path = std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx8.fth");
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "direct Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 8);
    EXPECT_EQ(runner.TokensProcessed(), 0);
}

TEST(qwen_runner_test, LoadsPersistentContext128DirectSafetensorsExport) {
    const auto model_path = std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx128.fth");
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "context-128 Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();
    EXPECT_EQ(runner.MaxContext(), 128);
}

TEST(qwen_runner_test, DefaultExportMatchesReferenceFirstDecodeToken) {
    const auto model_path =
        std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx8.fth");
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "default Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({151644}, 1, {}, &generated), 0) << runner.LastError();
    // Verified with the official Qwen3.5 implementation on the same raw
    // safetensors checkpoint. This catches incorrect per-head q/gate packing
    // in the atomic full-attention graph.
    EXPECT_EQ(generated, (std::vector<int64_t>{10748}));
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_runner_test, CudaBackendMatchesCommonForReferenceFirstDecodeToken) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    const auto model_path =
        std::filesystem::path("models/llm/qwen3.5-0.8b/qwen3.5-0.8b_decode_bf16_ctx8.fth");
    if (!std::filesystem::is_regular_file(model_path)) {
        GTEST_SKIP() << "default Qwen FTH asset is not present";
    }

    feather::demo::QwenRunner common_runner;
    ASSERT_EQ(common_runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0)
        << common_runner.LastError();
    std::vector<int64_t> common_generated;
    ASSERT_EQ(common_runner.Generate({151644}, 1, {}, &common_generated), 0) << common_runner.LastError();

    feather::demo::QwenRunner cuda_runner;
    ASSERT_EQ(cuda_runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0)
        << cuda_runner.LastError();
    std::vector<int64_t> cuda_generated;
    ASSERT_EQ(cuda_runner.Generate({151644}, 1, {}, &cuda_generated), 0) << cuda_runner.LastError();

    EXPECT_EQ(common_generated, (std::vector<int64_t>{10748}));
    EXPECT_EQ(cuda_generated, common_generated);
}
#endif
