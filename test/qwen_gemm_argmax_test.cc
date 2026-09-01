#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <limits>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#include "core/graph_lowering.h"
#include "src/kernel/cuda/runtime.h"
#endif

#include "core/static_graph.h"
#include "model/model_format.h"
#include "pass/qwen_gemm_argmax_fusion_pass.h"
#include "src/kernel/qwen_gemm_argmax.h"
#include "src/operator/qwen_gemm_argmax_op.h"
#include "util/bf16.h"
#include "util/fp8.h"

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

template <feather::DataType dtype>
std::shared_ptr<feather::Tensor> MakeFp8(const std::vector<float>& values,
                                         const std::vector<int64_t>& dims, float scale) {
    auto tensor = std::make_shared<feather::Tensor>();
    if constexpr (dtype == feather::DataType::FP8E4M3) {
        std::vector<feather::Fp8E4M3> encoded;
        encoded.reserve(values.size());
        for (const float value : values) {
            encoded.push_back(feather::Fp8E4M3{feather::FloatToFp8E4M3(value / scale)});
        }
        tensor->Assign<feather::Fp8E4M3>(encoded, dims);
    } else {
        std::vector<feather::Fp8E5M2> encoded;
        encoded.reserve(values.size());
        for (const float value : values) {
            encoded.push_back(feather::Fp8E5M2{feather::FloatToFp8E5M2(value / scale)});
        }
        tensor->Assign<feather::Fp8E5M2>(encoded, dims);
    }
    tensor->set_quantization({true, scale});
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

TEST(qwen_gemm_argmax_test, X86Fp8KernelReturnsScaledGreedyIndexForBothFormats) {
    for (const auto dtype : {feather::DataType::FP8E4M3, feather::DataType::FP8E5M2}) {
        std::shared_ptr<feather::Tensor> lhs;
        std::shared_ptr<feather::Tensor> rhs;
        if (dtype == feather::DataType::FP8E4M3) {
            lhs = MakeFp8<feather::DataType::FP8E4M3>({1.0f, 2.0f}, {1, 2}, 0.25f);
            rhs = MakeFp8<feather::DataType::FP8E4M3>({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2}, 0.5f);
        } else {
            lhs = MakeFp8<feather::DataType::FP8E5M2>({1.0f, 2.0f}, {1, 2}, 0.25f);
            rhs = MakeFp8<feather::DataType::FP8E5M2>({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, {3, 2}, 0.5f);
        }
        rhs->set_immutable(true);
        auto output = std::make_shared<feather::Tensor>(sizeof(int64_t));
        output->Resize({1});
        output->set_data_type(feather::DataType::INT64);

        feather::operators::QwenGemmArgmaxParam param;
        param.a = lhs;
        param.b = rhs;
        param.out = output;
        auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, dtype,
                                                                    "QwenGemmArgmax");
        ASSERT_NE(kernel, nullptr) << "missing FP8 QwenGemmArgmax dtype=" << static_cast<int>(dtype);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->Prepare(), 0);
        ASSERT_EQ(kernel->compute(), 0);
        EXPECT_EQ(output->data<int64_t>()[0], 1);
    }
}

TEST(qwen_gemm_argmax_test, X86Fp8ArgmaxMatchesQuantizedOrderingAtFormatBoundaries) {
    const std::vector<float> candidates = {
        -0.0f, 0.0f, 0.0624f, 0.0625f, 0.09375f, 0.125f, 0.1875f,
        0.2499f, 0.25f, 0.375f, 0.5f, 1.0f, 1.0625f, 1.1875f,
        2.0f, 7.0f, 15.0f, 31.0f, 63.0f, 127.0f, 255.0f, 447.9f,
    };
    for (const auto dtype : {feather::DataType::FP8E4M3, feather::DataType::FP8E5M2}) {
        std::shared_ptr<feather::Tensor> lhs;
        std::shared_ptr<feather::Tensor> rhs;
        if (dtype == feather::DataType::FP8E4M3) {
            lhs = MakeFp8<feather::DataType::FP8E4M3>({1.0f}, {1, 1}, 1.0f);
            rhs = MakeFp8<feather::DataType::FP8E4M3>(candidates, {static_cast<int64_t>(candidates.size()), 1}, 1.0f);
        } else {
            lhs = MakeFp8<feather::DataType::FP8E5M2>({1.0f}, {1, 1}, 1.0f);
            rhs = MakeFp8<feather::DataType::FP8E5M2>(candidates, {static_cast<int64_t>(candidates.size()), 1}, 1.0f);
        }
        rhs->set_immutable(true);
        auto output = std::make_shared<feather::Tensor>(sizeof(int64_t));
        output->Resize({1});
        output->set_data_type(feather::DataType::INT64);

        feather::operators::QwenGemmArgmaxParam param{};
        param.a = lhs;
        param.b = rhs;
        param.out = output;
        param.output_scale = 0.25f;
        auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, dtype,
                                                                     "QwenGemmArgmax");
        ASSERT_NE(kernel, nullptr);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->Prepare(), 0);
        ASSERT_EQ(kernel->compute(), 0);

        int64_t expected = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (size_t index = 0; index < candidates.size(); ++index) {
            const float projected = dtype == feather::DataType::FP8E4M3
                                        ? feather::Fp8E4M3ToFloat(feather::FloatToFp8E4M3(candidates[index]))
                                        : feather::Fp8E5M2ToFloat(feather::FloatToFp8E5M2(candidates[index]));
            const uint8_t output_code = dtype == feather::DataType::FP8E4M3
                                            ? feather::FloatToFp8E4M3(projected / 0.25f)
                                            : feather::FloatToFp8E5M2(projected / 0.25f);
            const float dequantized = dtype == feather::DataType::FP8E4M3
                                          ? feather::Fp8E4M3ToFloat(output_code) * 0.25f
                                          : feather::Fp8E5M2ToFloat(output_code) * 0.25f;
            const float rounded = feather::BFloat16ToFloat(feather::FloatToBFloat16(dequantized));
            if (rounded > best) {
                best = rounded;
                expected = static_cast<int64_t>(index);
            }
        }
        EXPECT_EQ(output->data<int64_t>()[0], expected) << "dtype=" << static_cast<int>(dtype);
    }
}

TEST(qwen_gemm_argmax_test, X86Fp8PackedArgmaxMatchesDirectAcrossWorkerTiles) {
    constexpr int64_t k = 2048;
    constexpr int64_t n = 128;
    constexpr float lhs_scale = 0.125f;
    constexpr float rhs_scale = 0.25f;
    constexpr float output_scale = 0.5f;

    for (const auto dtype : {feather::DataType::FP8E4M3, feather::DataType::FP8E5M2}) {
        std::vector<uint8_t> lhs(static_cast<size_t>(k));
        std::vector<uint8_t> rhs(static_cast<size_t>(n * k));
        for (int64_t row = 0; row < k; ++row) {
            const float value = static_cast<float>((row * 13) % 29 - 14) * lhs_scale;
            lhs[static_cast<size_t>(row)] =
                dtype == feather::DataType::FP8E4M3 ? feather::FloatToFp8E4M3(value / lhs_scale)
                                                     : feather::FloatToFp8E5M2(value / lhs_scale);
        }
        for (int64_t column = 0; column < n; ++column) {
            for (int64_t row = 0; row < k; ++row) {
                const float value = static_cast<float>((column * 7 + row * 3) % 31 - 15) * rhs_scale;
                rhs[static_cast<size_t>(column * k + row)] =
                    dtype == feather::DataType::FP8E4M3 ? feather::FloatToFp8E4M3(value / rhs_scale)
                                                         : feather::FloatToFp8E5M2(value / rhs_scale);
            }
        }

        int64_t direct_token = -1;
        ASSERT_EQ(feather::kernel::x86::ComputeLinearRowMajorX86Fp8TransposedRhsArgmax(
                      dtype, lhs.data(), lhs_scale, rhs.data(), rhs_scale, nullptr, k, n, output_scale,
                      &direct_token),
                  0);

        feather::kernel::x86::PackedFp8TransposedRhs packed;
        ASSERT_TRUE(packed.Pack(dtype, rhs.data(), k, n));
        int64_t packed_token = -1;
        ASSERT_EQ(feather::kernel::x86::ComputeLinearRowMajorX86Fp8TransposedRhsArgmax(
                      dtype, lhs.data(), lhs_scale, rhs.data(), rhs_scale, &packed, k, n, output_scale,
                      &packed_token),
                  0);
        EXPECT_EQ(packed_token, direct_token) << "dtype=" << static_cast<int>(dtype);
    }
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

TEST(qwen_gemm_argmax_test, QwenPassFusesFp8GemmCastIntoArgmax) {
    feather::model::ModelDesc model;
    model.name = "qwen_argmax_fp8_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"logits"};

    feather::model::ValueDesc hidden_value;
    hidden_value.tensor = {"hidden_fp8", {1, 2}, feather::DataType::FP8E4M3};
    hidden_value.tensor.quantization = {true, 0.25f};
    feather::model::ValueDesc weight_value;
    weight_value.tensor = {"lm_head_fp8", {3, 2}, feather::DataType::FP8E4M3};
    weight_value.tensor.quantization = {true, 0.5f};
    weight_value.constant = true;
    feather::model::ValueDesc fp8_logits_value;
    fp8_logits_value.tensor = {"logits_fp8", {1, 3}, feather::DataType::FP8E4M3};
    fp8_logits_value.tensor.quantization = {true, 0.125f};
    feather::model::ValueDesc logits_value;
    logits_value.tensor = {"logits", {1, 3}, feather::DataType::BF16};
    model.graph.values = {hidden_value, weight_value, fp8_logits_value, logits_value};

    feather::model::NodeDesc gemm;
    gemm.name = "lm_head_gemm";
    gemm.op_type = "Gemm";
    gemm.inputs = {"hidden_fp8", "lm_head_fp8"};
    gemm.outputs = {"logits_fp8"};
    gemm.attributes["transB"] = int64_t{1};
    feather::model::NodeDesc cast;
    cast.name = "logits_cast";
    cast.op_type = "Cast";
    cast.inputs = {"logits_fp8"};
    cast.outputs = {"logits"};
    cast.attributes["to"] = int64_t{16};
    model.graph.nodes = {gemm, cast};

    feather::StaticGraph graph;
    ASSERT_EQ(graph.SetModel(model), 0);
    graph.SetKernelDevice(feather::DeviceType::X86);
    ASSERT_EQ(graph.SetTensor("hidden_fp8", MakeFp8<feather::DataType::FP8E4M3>({1.0f, 2.0f}, {1, 2}, 0.25f)), 0);
    ASSERT_EQ(graph.SetTensor("lm_head_fp8",
                             MakeFp8<feather::DataType::FP8E4M3>({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f},
                                                                  {3, 2}, 0.5f)),
              0);
    ASSERT_EQ(graph.Build(), 0);
    auto passes = std::make_shared<feather::PassManager>();
    passes->AddPass(std::make_unique<feather::QwenGemmArgmaxFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    const auto* fused = graph.GetNode("logits_cast");
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->op_type, "QwenGemmArgmax");
    EXPECT_EQ(graph.GetNode("lm_head_gemm"), nullptr);
    EXPECT_EQ(graph.GetProducer("logits"), "logits_cast");
    ASSERT_NE(graph.GetTensor("logits"), nullptr);
    EXPECT_EQ(graph.GetTensor("logits")->data_type(), feather::DataType::INT64);

    const auto& graph_nodes = graph.model().graph.nodes;
    const auto fused_desc = std::find_if(graph_nodes.begin(), graph_nodes.end(),
                                         [](const feather::model::NodeDesc& node) {
                                              return node.name == "logits_cast";
                                         });
    ASSERT_NE(fused_desc, graph_nodes.end());
    const auto scale_it = fused_desc->attributes.find("output_scale");
    ASSERT_NE(scale_it, fused_desc->attributes.end());
    ASSERT_TRUE(std::holds_alternative<float>(scale_it->second));
    EXPECT_FLOAT_EQ(std::get<float>(scale_it->second), 0.125f);
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_gemm_argmax_test, QwenPassKeepsCudaFp8LinearAsGenericOps) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    feather::model::ModelDesc model;
    model.name = "qwen_cuda_cast_matmul_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"output_bf16"};

    feather::model::ValueDesc input_value;
    input_value.tensor = {"input_bf16", {1, 2}, feather::DataType::BF16};
    feather::model::ValueDesc fp8_input_value;
    fp8_input_value.tensor = {"input_fp8", {1, 2}, feather::DataType::FP8E4M3};
    fp8_input_value.tensor.quantization = {true, 0.25f};
    feather::model::ValueDesc weight_value;
    weight_value.tensor = {"weight_fp8", {2, 3}, feather::DataType::FP8E4M3};
    weight_value.tensor.quantization = {true, 0.5f};
    weight_value.constant = true;
    feather::model::ValueDesc fp8_output_value;
    fp8_output_value.tensor = {"output_fp8", {1, 3}, feather::DataType::FP8E4M3};
    fp8_output_value.tensor.quantization = {true, 0.125f};
    feather::model::ValueDesc output_value;
    output_value.tensor = {"output_bf16", {1, 3}, feather::DataType::BF16};
    model.graph.values = {input_value, fp8_input_value, weight_value, fp8_output_value, output_value};

    feather::model::NodeDesc input_cast;
    input_cast.name = "input_cast";
    input_cast.op_type = "Cast";
    input_cast.inputs = {"input_bf16"};
    input_cast.outputs = {"input_fp8"};
    input_cast.attributes["to"] = int64_t{17};
    feather::model::NodeDesc projection;
    projection.name = "projection";
    projection.op_type = "MatMul";
    projection.inputs = {"input_fp8", "weight_fp8"};
    projection.outputs = {"output_fp8"};
    feather::model::NodeDesc output_cast;
    output_cast.name = "output_cast";
    output_cast.op_type = "Cast";
    output_cast.inputs = {"output_fp8"};
    output_cast.outputs = {"output_bf16"};
    output_cast.attributes["to"] = int64_t{16};
    model.graph.nodes = {input_cast, projection, output_cast};

    feather::StaticGraph graph;
    graph.SetKernelDevice(feather::DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("input_bf16", MakeBf16({1.0f, -2.0f}, {1, 2})), 0);
    auto weight = MakeFp8<feather::DataType::FP8E4M3>({1.0f, 0.5f, -0.5f, 2.0f, -1.0f, 0.25f}, {2, 3}, 0.5f);
    weight->set_immutable(true);
    ASSERT_EQ(graph.SetTensor("weight_fp8", weight), 0);
    ASSERT_EQ(graph.Build(), 0);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    const auto* input_cast_node = graph.GetNode("input_cast");
    ASSERT_NE(input_cast_node, nullptr);
    EXPECT_EQ(input_cast_node->op_type, "Cast");
    const auto* projection_node = graph.GetNode("projection");
    ASSERT_NE(projection_node, nullptr);
    EXPECT_EQ(projection_node->op_type, "MatMul");
    const auto* output_cast_node = graph.GetNode("output_cast");
    ASSERT_NE(output_cast_node, nullptr);
    EXPECT_EQ(output_cast_node->op_type, "Cast");
    ASSERT_NE(graph.GetTensor("output_bf16"), nullptr);
    EXPECT_EQ(graph.GetTensor("output_bf16")->data_type(), feather::DataType::BF16);

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0);
    {
        feather::kernel::cuda_detail::DeferredHostSyncScope deferred_host_sync;
        ASSERT_EQ(runtime.Run(), 0);
    }
    auto output = runtime.GetTensor("output_bf16");
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(feather::kernel::cuda_detail::SyncTensorToHost(
                  output.get(), output->numel() * sizeof(feather::BFloat16), output->raw_data()), 0);
    ASSERT_EQ(output->numel(), 3);
    EXPECT_EQ(output->data<feather::BFloat16>()[0].bits, feather::FloatToBFloat16(-3.0f));
    EXPECT_EQ(output->data<feather::BFloat16>()[1].bits, feather::FloatToBFloat16(2.5f));
    EXPECT_EQ(output->data<feather::BFloat16>()[2].bits, feather::FloatToBFloat16(-1.0f));
}

TEST(qwen_gemm_argmax_test, QwenPassKeepsFp8LogitsGemmUnfusedOnCuda) {
    feather::model::ModelDesc model;
    model.name = "qwen_argmax_fp8_cuda_fixture";
    model.graph.name = "decode";
    model.graph.outputs = {"logits"};

    feather::model::ValueDesc hidden_value;
    hidden_value.tensor = {"hidden_fp8", {1, 2}, feather::DataType::FP8E4M3};
    hidden_value.tensor.quantization = {true, 0.25f};
    feather::model::ValueDesc weight_value;
    weight_value.tensor = {"lm_head_fp8", {3, 2}, feather::DataType::FP8E4M3};
    weight_value.tensor.quantization = {true, 0.5f};
    weight_value.constant = true;
    feather::model::ValueDesc fp8_logits_value;
    fp8_logits_value.tensor = {"logits_fp8", {1, 3}, feather::DataType::FP8E4M3};
    fp8_logits_value.tensor.quantization = {true, 0.125f};
    feather::model::ValueDesc logits_value;
    logits_value.tensor = {"logits", {1, 3}, feather::DataType::BF16};
    model.graph.values = {hidden_value, weight_value, fp8_logits_value, logits_value};

    feather::model::NodeDesc gemm;
    gemm.name = "lm_head_gemm";
    gemm.op_type = "Gemm";
    gemm.inputs = {"hidden_fp8", "lm_head_fp8"};
    gemm.outputs = {"logits_fp8"};
    gemm.attributes["transB"] = int64_t{1};
    feather::model::NodeDesc cast;
    cast.name = "logits_cast";
    cast.op_type = "Cast";
    cast.inputs = {"logits_fp8"};
    cast.outputs = {"logits"};
    cast.attributes["to"] = int64_t{16};
    model.graph.nodes = {gemm, cast};

    feather::StaticGraph graph;
    graph.SetKernelDevice(feather::DeviceType::CUDA);
    ASSERT_EQ(graph.SetModel(model), 0);
    ASSERT_EQ(graph.SetTensor("hidden_fp8", MakeFp8<feather::DataType::FP8E4M3>({1.0f, 2.0f}, {1, 2}, 0.25f)), 0);
    ASSERT_EQ(graph.SetTensor("lm_head_fp8",
                             MakeFp8<feather::DataType::FP8E4M3>({1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f},
                                                                  {3, 2}, 0.5f)),
              0);
    ASSERT_EQ(graph.Build(), 0);

    auto passes = std::make_shared<feather::PassManager>();
    passes->AddPass(std::make_unique<feather::QwenGemmArgmaxFusionPass>());
    graph.SetPassManager(passes);
    ASSERT_EQ(graph.ApplyPasses(), 0);

    const auto* gemm_node = graph.GetNode("lm_head_gemm");
    ASSERT_NE(gemm_node, nullptr);
    EXPECT_EQ(gemm_node->op_type, "Gemm");
    const auto* cast_node = graph.GetNode("logits_cast");
    ASSERT_NE(cast_node, nullptr);
    EXPECT_EQ(cast_node->op_type, "Cast");
    ASSERT_NE(graph.GetTensor("logits"), nullptr);
    EXPECT_EQ(graph.GetTensor("logits")->data_type(), feather::DataType::BF16);

    feather::RuntimeGraph runtime;
    feather::GraphLowering lowering;
    ASSERT_EQ(lowering.Lower(graph, &runtime), 0);
    const auto* runtime_gemm = runtime.GetNode("lm_head_gemm");
    ASSERT_NE(runtime_gemm, nullptr);
    EXPECT_EQ(runtime_gemm->kernel_device, feather::DeviceType::CUDA);
    const auto* runtime_cast = runtime.GetNode("logits_cast");
    ASSERT_NE(runtime_cast, nullptr);
    EXPECT_EQ(runtime_cast->kernel_device, feather::DeviceType::CUDA);
}

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
