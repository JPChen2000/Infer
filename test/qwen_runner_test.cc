#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "demo/qwen_demo.h"
#include "demo/qwen_runner.h"
#include "model/model_io.h"
#include "util/bf16.h"

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

std::shared_ptr<feather::Tensor> MakeBf16Tensor(const std::vector<float>& values,
                                                const std::vector<int64_t>& dims) {
    std::vector<feather::BFloat16> storage;
    storage.reserve(values.size());
    for (const float value : values) {
        storage.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<feather::BFloat16>(storage, dims);
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

std::filesystem::path WriteStateTransitionFixture() {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_state_transition_fixture";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 4}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1, 4}, feather::DataType::FP32),
        MakeValue("transition", {4, 4}, feather::DataType::FP32, true),
        MakeValue("bias", {4}, feather::DataType::FP32, true),
        MakeValue("next_state_raw", {1, 4}, feather::DataType::FP32),
        MakeValue("next_recurrent_state_0", {1, 4}, feather::DataType::FP32),
        MakeValue("logits", {1, 1, 4}, feather::DataType::FP32),
    };

    feather::model::NodeDesc transition;
    transition.name = "state_transition";
    transition.op_type = "Gemm";
    transition.inputs = {"recurrent_state_0", "transition", "bias"};
    transition.outputs = {"next_state_raw"};

    feather::model::NodeDesc state_output;
    state_output.name = "next_state";
    state_output.op_type = "Identity";
    state_output.inputs = {"next_state_raw"};
    state_output.outputs = {"next_recurrent_state_0"};

    feather::model::NodeDesc logits;
    logits.name = "state_logits";
    logits.op_type = "Reshape";
    logits.inputs = {"next_state_raw"};
    logits.outputs = {"logits"};
    logits.attributes["shape"] = std::vector<int64_t>{1, 1, 4};
    model.graph.nodes = {transition, state_output, logits};

    std::unordered_map<std::string, std::shared_ptr<feather::Tensor>> weights;
    weights["transition"] = MakeFloatTensor({0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, -2.0f, 2.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f},
                                              {4, 4});
    weights["bias"] = MakeFloatTensor({0.0f, 10.0f, 0.0f, 0.0f}, {4});
    const auto path = std::filesystem::temp_directory_path() / "qwen_runner_state_transition_fixture.fth";
    feather::model::ModelWriter writer;
    EXPECT_TRUE(writer.Save(path.string(), model, weights));
    return path;
}

std::filesystem::path WriteBf16GreedyFixture() {
    feather::model::ModelDesc model;
    model.name = "qwen_runner_bf16_greedy_fixture";
    model.version = 1;
    model.graph.name = "decode";
    model.graph.inputs = {"token_ids", "position_id", "attention_mask", "recurrent_state_0"};
    model.graph.outputs = {"next_recurrent_state_0", "logits"};
    model.graph.values = {
        MakeValue("token_ids", {1, 1}, feather::DataType::INT64),
        MakeValue("position_id", {1}, feather::DataType::INT64),
        MakeValue("attention_mask", {1, 1, 1, 4}, feather::DataType::BF16),
        MakeValue("recurrent_state_0", {1}, feather::DataType::FP32),
        MakeValue("lookup", {2, 4}, feather::DataType::BF16, true),
        MakeValue("logits", {1, 1, 4}, feather::DataType::BF16),
        MakeValue("next_recurrent_state_0", {1}, feather::DataType::FP32),
    };

    feather::model::NodeDesc gather;
    gather.name = "lookup_bf16_logits";
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
    weights["lookup"] = MakeBf16Tensor({std::numeric_limits<float>::quiet_NaN(), 5.0f, 5.0f, 4.0f,
                                         -1.0f, -2.0f, -3.0f, -4.0f},
                                        {2, 4});
    const auto path = std::filesystem::temp_directory_path() / "qwen_runner_bf16_greedy_fixture.fth";
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

TEST(qwen_demo_test, BuffersIncompleteUtf8UntilTheNextChunk) {
    feather::demo::QwenUtf8Stream stream;

    EXPECT_EQ(stream.Push("hello"), "hello");
    EXPECT_TRUE(stream.Push("\xE4").empty());
    EXPECT_EQ(stream.Push("\xBD\xA0"), "\xE4\xBD\xA0");
    EXPECT_TRUE(stream.Finish().empty());

    feather::demo::QwenUtf8Stream emoji;
    EXPECT_TRUE(emoji.Push("\xF0\x9F").empty());
    EXPECT_EQ(emoji.Push("\x98\x80"), "\xF0\x9F\x98\x80");

    feather::demo::QwenUtf8Stream malformed;
    EXPECT_EQ(malformed.Push("\xC0ok"), "\xEF\xBF\xBDok");

    feather::demo::QwenUtf8Stream truncated;
    EXPECT_TRUE(truncated.Push("\xE4").empty());
    EXPECT_EQ(truncated.Finish(), "\xEF\xBF\xBD");
}

TEST(qwen_runner_test, StreamsGeneratedTokensThroughCallback) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> streamed;
    std::vector<int64_t> processed_at_callback;
    ASSERT_EQ(runner.GenerateStream({0}, 3, {2}, [&runner, &streamed, &processed_at_callback](int64_t token_id) {
                  streamed.push_back(token_id);
                  processed_at_callback.push_back(runner.TokensProcessed());
              }),
              0)
        << runner.LastError();
    EXPECT_EQ(streamed, (std::vector<int64_t>{1, 3, 2}));
    EXPECT_EQ(processed_at_callback, (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ(runner.TokensProcessed(), 4);
}

TEST(qwen_runner_test, ExposesRuntimeProfileSummariesWhenEnabled) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    runner.SetRuntimeProfilingEnabled(true);
    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();

    const auto& summaries = runner.RuntimeProfileSummaries();
    ASSERT_FALSE(summaries.empty());
    EXPECT_EQ(summaries[0].call_count, 2);
    EXPECT_EQ(summaries[0].op_type, "Gather");
}

TEST(qwen_runner_test, PreservesOutputWhenPromptValidationFails) {
    const auto model_path = WriteAutoregressiveFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated = {42};
    EXPECT_NE(runner.Generate({}, 1, {}, &generated), 0);
    EXPECT_EQ(generated, (std::vector<int64_t>{42}));
}

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

TEST(qwen_runner_test, PreservesPingPongStateWhenNextStateUsesIdentityView) {
    const auto model_path = WriteStateTransitionFixture();
    for (const auto backend : {feather::demo::QwenBackend::kCommon, feather::demo::QwenBackend::kX86}) {
        feather::demo::QwenRunner runner;
        ASSERT_EQ(runner.Load(model_path.string(), backend), 0) << runner.LastError();

        std::vector<int64_t> generated;
        ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
        // The second decode depends on the first state transition. With the
        // Identity output buffer swapped directly, Gemm becomes in-place and
        // picks token 0 instead of token 2.
        EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));
    }
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_runner_test, CudaPreservesPingPongStateAndResetWhenNextStateUsesIdentityRelay) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    const auto model_path = WriteStateTransitionFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCuda), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));

    ASSERT_EQ(runner.Reset(), 0) << runner.LastError();
    ASSERT_EQ(runner.Generate({0}, 2, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1, 2}));
}
#endif

TEST(qwen_runner_test, SelectsFirstFiniteMaximumFromBf16Logits) {
    const auto model_path = WriteBf16GreedyFixture();
    feather::demo::QwenRunner runner;
    ASSERT_EQ(runner.Load(model_path.string(), feather::demo::QwenBackend::kCommon), 0) << runner.LastError();

    std::vector<int64_t> generated;
    ASSERT_EQ(runner.Generate({0}, 1, {}, &generated), 0) << runner.LastError();
    EXPECT_EQ(generated, (std::vector<int64_t>{1}));
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

TEST(qwen_runner_test, X86BackendMatchesCommonForReferenceFirstDecodeToken) {
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

    feather::demo::QwenRunner x86_runner;
    ASSERT_EQ(x86_runner.Load(model_path.string(), feather::demo::QwenBackend::kX86), 0) << x86_runner.LastError();
    std::vector<int64_t> x86_generated;
    ASSERT_EQ(x86_runner.Generate({151644}, 1, {}, &x86_generated), 0) << x86_runner.LastError();

    EXPECT_EQ(common_generated, (std::vector<int64_t>{10748}));
    EXPECT_EQ(x86_generated, common_generated);
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
