#include <gtest/gtest.h>

#include <memory>
#include <limits>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#include "core/graph_lowering.h"
#endif

#include "core/static_graph.h"
#include "model/model_format.h"
#include "pass/qwen_gemm_argmax_fusion_pass.h"
#include "src/kernel/qwen_gemm_argmax.h"
#include "src/operator/qwen_gemm_argmax_op.h"
#include "util/bf16.h"

namespace {

std::shared_ptr<feather::Tensor> MakeBf16(const std::vector<float>& values,
                                          const std::vector<int64_t>& dims) {
    std::vector<feather::BFloat16> encoded;
    encoded.reserve(values.size());
    for (const float value : values) {
        encoded.push_back(feather::BFloat16{feather::FloatToBFloat16(value)});
    }
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<feather::BFloat16>(encoded, dims);
    return tensor;
}

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

}  // namespace

TEST(qwen_gemm_argmax_test, X86KernelReturnsBf16RoundedGreedyIndex) {
    auto lhs = MakeBf16({1.0f, 2.0f}, {1, 2});
    auto rhs = MakeBf16({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2});
    auto output = std::make_shared<feather::Tensor>(sizeof(int64_t));
    output->Resize({1});
    output->set_data_type(feather::DataType::INT64);

    feather::operators::QwenGemmArgmaxParam param;
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                 feather::DataType::BF16,
                                                                 "QwenGemmArgmax");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->Prepare(), 0);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int64_t>()[0], 1);
}

TEST(qwen_gemm_argmax_test, X86KernelSkipsNaNAndKeepsFirstSignedZeroMaximum) {
    auto lhs = MakeBf16({1.0f}, {1, 1});
    auto rhs = MakeBf16({std::numeric_limits<float>::quiet_NaN(), -0.0f, 0.0f}, {3, 1});
    auto output = std::make_shared<feather::Tensor>(sizeof(int64_t));
    output->Resize({1});
    output->set_data_type(feather::DataType::INT64);

    feather::operators::QwenGemmArgmaxParam param;
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                 feather::DataType::BF16,
                                                                 "QwenGemmArgmax");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->Prepare(), 0);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int64_t>()[0], 1);
}

TEST(qwen_gemm_argmax_test, X86KernelVectorArgmaxKeepsBf16OrderingAcrossPackedBlock) {
    constexpr int64_t k = 2;
    constexpr int64_t n = 64;
    auto lhs = MakeBf16({1.0f, 1.0f}, {1, k});
    std::vector<float> rhs_values(static_cast<size_t>(n * k), -4.0f);
    for (int64_t col = 0; col < n; ++col) {
        rhs_values[static_cast<size_t>(col * k)] = -1.0f + static_cast<float>(col % 11) * 0.125f;
        rhs_values[static_cast<size_t>(col * k + 1)] = -0.5f + static_cast<float>((col * 3) % 7) * 0.0625f;
    }
    rhs_values[17 * k] = 3.0f;
    rhs_values[17 * k + 1] = 2.0f;
    rhs_values[49 * k] = 3.0f;
    rhs_values[49 * k + 1] = 2.0f;
    rhs_values[7 * k] = std::numeric_limits<float>::quiet_NaN();
    rhs_values[7 * k + 1] = std::numeric_limits<float>::quiet_NaN();
    auto rhs = MakeBf16(rhs_values, {n, k});
    auto output = std::make_shared<feather::Tensor>(sizeof(int64_t));
    output->Resize({1});
    output->set_data_type(feather::DataType::INT64);

    feather::operators::QwenGemmArgmaxParam param;
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                 feather::DataType::BF16,
                                                                 "QwenGemmArgmax");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->Prepare(), 0);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int64_t>()[0], 17);
}

TEST(qwen_gemm_argmax_test, X86SingleThreadArgmaxUsesWideTilePath) {
    constexpr int64_t k = 4;
    constexpr int64_t n = 96;
    auto lhs = MakeBf16({1.0f, 1.0f, 1.0f, 1.0f}, {1, k});
    std::vector<float> rhs_values(static_cast<size_t>(n * k), -1.0f);
    for (int64_t row = 0; row < k; ++row) {
        rhs_values[static_cast<size_t>(65 * k + row)] = row == 0 ? 4.0f : 0.0f;
    }
    auto rhs = MakeBf16(rhs_values, {n, k});
    feather::kernel::x86::PackedBf16TransposedRhs packed_rhs;
    ASSERT_TRUE(packed_rhs.Pack(rhs->data<uint16_t>(), k, n));

    int64_t token = -1;
    ASSERT_EQ(feather::kernel::x86::ComputeLinearRowMajorX86Bf16PackedTransposedRhsArgmaxSingleThread(
                  lhs->data<uint16_t>(), rhs->data<uint16_t>(), packed_rhs, k, n, &token),
              0);
    EXPECT_EQ(token, 65);
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_gemm_argmax_test, CudaKernelMatchesCommonForBf16GreedyAndNaNSemantics) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto lhs = MakeBf16({1.0f, 2.0f}, {1, 2});
    auto rhs = MakeBf16({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2});
    auto common_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    common_out->Resize({1});
    common_out->set_data_type(feather::DataType::INT64);
    auto cuda_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    cuda_out->Resize({1});
    cuda_out->set_data_type(feather::DataType::INT64);

    feather::operators::QwenGemmArgmaxParam common_param{};
    common_param.a = lhs;
    common_param.b = rhs;
    common_param.out = common_out;
    auto common = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::BF16,
                                                                 "QwenGemmArgmax");
    ASSERT_NE(common, nullptr);
    common->SetParam(&common_param);
    ASSERT_EQ(common->compute(), 0);

    auto cuda_param = common_param;
    cuda_param.out = cuda_out;
    auto cuda = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA,
                                                               feather::DataType::BF16,
                                                               "QwenGemmArgmax");
    ASSERT_NE(cuda, nullptr);
    cuda->SetParam(&cuda_param);
    feather::kernel::ResetLastCudaQwenGemmArgmaxBackend();
    ASSERT_EQ(cuda->compute(), 0);
    EXPECT_EQ(feather::kernel::LastCudaQwenGemmArgmaxBackend(),
              feather::kernel::CudaQwenGemmArgmaxBackend::kFusedGemv);
    EXPECT_EQ(cuda_out->data<int64_t>()[0], common_out->data<int64_t>()[0]);
    EXPECT_EQ(cuda_out->data<int64_t>()[0], 1);

    auto nan_rhs = MakeBf16({std::numeric_limits<float>::quiet_NaN(), 1.0f, -0.0f, 0.0f}, {2, 2});
    auto nan_common_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    nan_common_out->Resize({1});
    nan_common_out->set_data_type(feather::DataType::INT64);
    auto nan_cuda_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    nan_cuda_out->Resize({1});
    nan_cuda_out->set_data_type(feather::DataType::INT64);
    common_param.b = nan_rhs;
    common_param.out = nan_common_out;
    common->SetParam(&common_param);
    ASSERT_EQ(common->compute(), 0);
    cuda_param.b = nan_rhs;
    cuda_param.out = nan_cuda_out;
    cuda->SetParam(&cuda_param);
    ASSERT_EQ(cuda->compute(), 0);
    EXPECT_EQ(nan_cuda_out->data<int64_t>()[0], nan_common_out->data<int64_t>()[0]);
    EXPECT_EQ(nan_cuda_out->data<int64_t>()[0], 1);
}

TEST(qwen_gemm_argmax_test, CudaFusedGemvCoversMultipleCandidateBlocks) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int64_t k = 37;
    constexpr int64_t n = 2053;
    std::vector<float> lhs_values(static_cast<size_t>(k));
    for (int64_t index = 0; index < k; ++index) {
        lhs_values[static_cast<size_t>(index)] = static_cast<float>((index % 9) - 4) * 0.125f;
    }
    std::vector<float> rhs_values(static_cast<size_t>(n * k));
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t column = 0; column < k; ++column) {
            rhs_values[static_cast<size_t>(row * k + column)] =
                static_cast<float>(((row + 3 * column) % 17) - 8) * 0.0625f;
        }
    }
    for (int64_t column = 0; column < k; ++column) {
        rhs_values[static_cast<size_t>(2049 * k + column)] = lhs_values[static_cast<size_t>(column)] * 2.0f;
        rhs_values[static_cast<size_t>(n - 1) * k + static_cast<size_t>(column)] =
            lhs_values[static_cast<size_t>(column)] * 2.0f;
    }

    auto lhs = MakeBf16(lhs_values, {1, k});
    auto rhs = MakeBf16(rhs_values, {n, k});
    auto common_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    common_out->Resize({1});
    common_out->set_data_type(feather::DataType::INT64);
    auto cuda_out = std::make_shared<feather::Tensor>(sizeof(int64_t));
    cuda_out->Resize({1});
    cuda_out->set_data_type(feather::DataType::INT64);

    feather::operators::QwenGemmArgmaxParam common_param{};
    common_param.a = lhs;
    common_param.b = rhs;
    common_param.out = common_out;
    auto common = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                 feather::DataType::BF16,
                                                                 "QwenGemmArgmax");
    ASSERT_NE(common, nullptr);
    common->SetParam(&common_param);
    ASSERT_EQ(common->compute(), 0);

    auto cuda_param = common_param;
    cuda_param.out = cuda_out;
    auto cuda = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA,
                                                               feather::DataType::BF16,
                                                               "QwenGemmArgmax");
    ASSERT_NE(cuda, nullptr);
    cuda->SetParam(&cuda_param);
    feather::kernel::ResetLastCudaQwenGemmArgmaxBackend();
    ASSERT_EQ(cuda->compute(), 0);
    EXPECT_EQ(feather::kernel::LastCudaQwenGemmArgmaxBackend(),
              feather::kernel::CudaQwenGemmArgmaxBackend::kFusedGemv);
    EXPECT_EQ(cuda_out->data<int64_t>()[0], common_out->data<int64_t>()[0]);
    EXPECT_EQ(cuda_out->data<int64_t>()[0], 2049);
}
#endif

TEST(qwen_gemm_argmax_test, QwenPassReplacesOnlyFinalLogitsGemm) {
    feather::model::ModelDesc model;
    model.name = "qwen_argmax_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"logits"};
    model.graph.values = {
        {feather::model::TensorDesc{"hidden", {1, 2}, feather::DataType::BF16}, false},
        {feather::model::TensorDesc{"lm_head", {3, 2}, feather::DataType::BF16}, true},
        {feather::model::TensorDesc{"logits", {1, 3}, feather::DataType::BF16}, false},
    };
    feather::model::NodeDesc logits;
    logits.name = "lm_head_gemm";
    logits.op_type = "Gemm";
    logits.inputs = {"hidden", "lm_head"};
    logits.outputs = {"logits"};
    logits.attributes["transB"] = int64_t{1};
    model.graph.nodes = {logits};

    feather::StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    graph.SetKernelDevice(feather::DeviceType::X86);
    ASSERT_EQ(graph.SetTensor("hidden", MakeBf16({1.0f, 2.0f}, {1, 2})), 0);
    ASSERT_EQ(graph.SetTensor("lm_head", MakeBf16({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2})), 0);
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    const auto* node = graph.GetNode("lm_head_gemm");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, "QwenGemmArgmax");
    EXPECT_EQ(graph.GetTensor("logits")->data_type(), feather::DataType::INT64);
}

TEST(qwen_gemm_argmax_test, QwenPassLeavesBiasedLogitsGemmUnchanged) {
    feather::model::ModelDesc model;
    model.name = "qwen_argmax_bias_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"logits"};
    model.graph.values = {
        {feather::model::TensorDesc{"hidden", {1, 2}, feather::DataType::BF16}, false},
        {feather::model::TensorDesc{"lm_head", {3, 2}, feather::DataType::BF16}, true},
        {feather::model::TensorDesc{"bias", {3}, feather::DataType::BF16}, true},
        {feather::model::TensorDesc{"logits", {1, 3}, feather::DataType::BF16}, false},
    };
    feather::model::NodeDesc logits;
    logits.name = "lm_head_gemm";
    logits.op_type = "Gemm";
    logits.inputs = {"hidden", "lm_head", "bias"};
    logits.outputs = {"logits"};
    logits.attributes["transB"] = int64_t{1};
    model.graph.nodes = {logits};

    feather::StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    graph.SetKernelDevice(feather::DeviceType::X86);
    ASSERT_EQ(graph.SetTensor("hidden", MakeBf16({1.0f, 2.0f}, {1, 2})), 0);
    ASSERT_EQ(graph.SetTensor("lm_head", MakeBf16({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2})), 0);
    ASSERT_EQ(graph.SetTensor("bias", MakeBf16({0.0f, 0.0f, 0.0f}, {3})), 0);
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);
    const auto* node = graph.GetNode("lm_head_gemm");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, "Gemm");
    EXPECT_EQ(graph.GetTensor("logits")->data_type(), feather::DataType::BF16);
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_gemm_argmax_test, QwenPassLowersFinalLogitsGemmToCuda) {
    feather::model::ModelDesc model;
    model.name = "qwen_argmax_cuda_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"logits"};
    model.graph.values = {
        {feather::model::TensorDesc{"hidden", {1, 2}, feather::DataType::BF16}, false},
        {feather::model::TensorDesc{"lm_head", {3, 2}, feather::DataType::BF16}, true},
        {feather::model::TensorDesc{"logits", {1, 3}, feather::DataType::BF16}, false},
    };
    feather::model::NodeDesc logits;
    logits.name = "lm_head_gemm";
    logits.op_type = "Gemm";
    logits.inputs = {"hidden", "lm_head"};
    logits.outputs = {"logits"};
    logits.attributes["transB"] = int64_t{1};
    model.graph.nodes = {logits};

    feather::StaticGraph graph;
    graph.SetKernelDevice(feather::DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("hidden", MakeBf16({1.0f, 2.0f}, {1, 2})), 0);
    ASSERT_EQ(graph.SetTensor("lm_head", MakeBf16({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2})), 0);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<feather::PassManager>();
    passes->AddPass(std::make_unique<feather::QwenGemmArgmaxFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    const auto* node = graph.GetNode("lm_head_gemm");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, "QwenGemmArgmax");
    EXPECT_EQ(graph.GetTensor("logits")->data_type(), feather::DataType::INT64);

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0);
    const auto* runtime_node = runtime.GetNode("lm_head_gemm");
    ASSERT_NE(runtime_node, nullptr);
    EXPECT_EQ(runtime_node->kernel_device, feather::DeviceType::CUDA);
}
#endif
