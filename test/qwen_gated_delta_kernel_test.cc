#include <gtest/gtest.h>

#include <memory>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/qwen_gated_delta.h"
#include "src/operator/params.h"

namespace {

std::shared_ptr<feather::Tensor> MakeTensor(const std::vector<float>& values,
                                            const std::vector<int64_t>& dims) {
    auto tensor = std::make_shared<feather::Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

feather::operators::QwenGatedDeltaStateParam MakeStateParam(
    const std::shared_ptr<feather::Tensor>& state, const std::shared_ptr<feather::Tensor>& k,
    const std::shared_ptr<feather::Tensor>& v, const std::shared_ptr<feather::Tensor>& beta,
    const std::shared_ptr<feather::Tensor>& decay, const std::shared_ptr<feather::Tensor>& out) {
    feather::operators::QwenGatedDeltaStateParam param{};
    param.state = state;
    param.k = k;
    param.v = v;
    param.beta = beta;
    param.decay = decay;
    param.out = out;
    return param;
}

feather::operators::QwenGatedDeltaParam MakeCombinedParam(
    const std::shared_ptr<feather::Tensor>& state, const std::shared_ptr<feather::Tensor>& k,
    const std::shared_ptr<feather::Tensor>& v, const std::shared_ptr<feather::Tensor>& beta,
    const std::shared_ptr<feather::Tensor>& decay, const std::shared_ptr<feather::Tensor>& q,
    const std::shared_ptr<feather::Tensor>& next_state, const std::shared_ptr<feather::Tensor>& out) {
    feather::operators::QwenGatedDeltaParam param{};
    param.state = state;
    param.k = k;
    param.v = v;
    param.beta = beta;
    param.decay = decay;
    param.q = q;
    param.next_state = next_state;
    param.out = out;
    return param;
}

TEST(qwen_gated_delta_kernel_test, StateUpdateMatchesReferenceOnCommonAndX86) {
    const auto state = MakeTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 1, 2, 3});
    const auto k = MakeTensor({0.5f, -1.0f}, {1, 1, 2});
    const auto v = MakeTensor({1.0f, 2.0f, 3.0f}, {1, 1, 3});
    const auto beta = MakeTensor({0.25f}, {1, 1, 1});
    const auto decay = MakeTensor({0.8f}, {1, 1, 1, 1});
    const std::vector<float> expected = {1.275f, 2.25f, 3.225f, 2.25f, 2.7f, 3.15f};

    for (const auto device : {feather::DeviceType::COMMON, feather::DeviceType::X86}) {
        auto out = MakeTensor(std::vector<float>(expected.size(), 0.0f), {1, 1, 2, 3});
        auto param = MakeStateParam(state, k, v, beta, decay, out);
        auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::FP32,
                                                                     "QwenGatedDeltaState");
        ASSERT_NE(kernel, nullptr) << static_cast<int>(device);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->compute(), 0) << static_cast<int>(device);
        ASSERT_EQ(out->data_type(), feather::DataType::FP32);
        for (size_t index = 0; index < expected.size(); ++index) {
            EXPECT_NEAR(out->data<float>()[index], expected[index], 1e-5f)
                << "device=" << static_cast<int>(device) << " index=" << index;
        }
    }
}

TEST(qwen_gated_delta_kernel_test, X86MultiHeadStateUpdateMatchesCommon) {
    constexpr int64_t kHeads = 16;
    constexpr int64_t kKey = 64;
    constexpr int64_t kValue = 128;
    std::vector<float> state(static_cast<size_t>(kHeads * kKey * kValue));
    std::vector<float> k(static_cast<size_t>(kHeads * kKey));
    std::vector<float> v(static_cast<size_t>(kHeads * kValue));
    std::vector<float> beta(static_cast<size_t>(kHeads));
    std::vector<float> decay(static_cast<size_t>(kHeads));
    for (size_t index = 0; index < state.size(); ++index) {
        state[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.03125f;
    }
    for (size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.0625f;
    }
    for (size_t index = 0; index < v.size(); ++index) {
        v[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    }
    for (int64_t head = 0; head < kHeads; ++head) {
        beta[static_cast<size_t>(head)] = 0.05f * static_cast<float>(head + 1);
        decay[static_cast<size_t>(head)] = 0.75f + 0.01f * static_cast<float>(head);
    }

    const auto state_tensor = MakeTensor(state, {1, kHeads, kKey, kValue});
    const auto k_tensor = MakeTensor(k, {1, kHeads, kKey});
    const auto v_tensor = MakeTensor(v, {1, kHeads, kValue});
    const auto beta_tensor = MakeTensor(beta, {1, kHeads, 1});
    const auto decay_tensor = MakeTensor(decay, {1, kHeads, 1, 1});
    const auto common_out = MakeTensor(std::vector<float>(state.size()), {1, kHeads, kKey, kValue});
    const auto x86_out = MakeTensor(std::vector<float>(state.size()), {1, kHeads, kKey, kValue});

    auto common_param = MakeStateParam(state_tensor, k_tensor, v_tensor, beta_tensor, decay_tensor, common_out);
    auto x86_param = MakeStateParam(state_tensor, k_tensor, v_tensor, beta_tensor, decay_tensor, x86_out);
    auto common = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON, feather::DataType::FP32,
                                                                 "QwenGatedDeltaState");
    auto x86 = feather::KernelDispatcher::instance().create(feather::DeviceType::X86, feather::DataType::FP32,
                                                              "QwenGatedDeltaState");
    ASSERT_NE(common, nullptr);
    ASSERT_NE(x86, nullptr);
    common->SetParam(&common_param);
    x86->SetParam(&x86_param);
    ASSERT_EQ(common->compute(), 0);
    ASSERT_EQ(x86->compute(), 0);
    for (size_t index = 0; index < state.size(); ++index) {
        EXPECT_NEAR(x86_out->data<float>()[index], common_out->data<float>()[index], 1e-5f) << index;
    }
}

TEST(qwen_gated_delta_kernel_test, OutputReductionMatchesReferenceOnCommonAndX86) {
    const auto state = MakeTensor({1.275f, 2.25f, 3.225f, 2.25f, 2.7f, 3.15f}, {1, 1, 2, 3});
    const auto q = MakeTensor({0.2f, -0.4f}, {1, 1, 2});
    const auto out = MakeTensor({0.0f, 0.0f, 0.0f}, {1, 1, 3});
    const std::vector<float> expected = {-0.645f, -0.63f, -0.615f};

    for (const auto device : {feather::DeviceType::COMMON, feather::DeviceType::X86}) {
        auto result = MakeTensor(std::vector<float>(expected.size(), 0.0f), {1, 1, 3});
        feather::operators::QwenGatedDeltaOutputParam param{};
        param.state = state;
        param.q = q;
        param.out = result;
        auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::FP32,
                                                                     "QwenGatedDeltaOutput");
        ASSERT_NE(kernel, nullptr) << static_cast<int>(device);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->compute(), 0) << static_cast<int>(device);
        ASSERT_EQ(result->data_type(), feather::DataType::FP32);
        for (size_t index = 0; index < expected.size(); ++index) {
            EXPECT_NEAR(result->data<float>()[index], expected[index], 1e-5f)
                << "device=" << static_cast<int>(device) << " index=" << index;
        }
    }
}

TEST(qwen_gated_delta_kernel_test, CombinedStateAndOutputMatchesReferenceOnCommonAndX86) {
    const auto state = MakeTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 1, 2, 3});
    const auto k = MakeTensor({0.5f, -1.0f}, {1, 1, 2});
    const auto v = MakeTensor({1.0f, 2.0f, 3.0f}, {1, 1, 3});
    const auto beta = MakeTensor({0.25f}, {1, 1, 1});
    const auto decay = MakeTensor({0.8f}, {1, 1, 1, 1});
    const auto q = MakeTensor({0.2f, -0.4f}, {1, 1, 2});
    const std::vector<float> expected_state = {1.275f, 2.25f, 3.225f, 2.25f, 2.7f, 3.15f};
    const std::vector<float> expected_output = {-0.645f, -0.63f, -0.615f};

    for (const auto device : {feather::DeviceType::COMMON, feather::DeviceType::X86}) {
        auto next_state = MakeTensor(std::vector<float>(expected_state.size()), {1, 1, 2, 3});
        auto out = MakeTensor(std::vector<float>(expected_output.size()), {1, 1, 3});
        auto param = MakeCombinedParam(state, k, v, beta, decay, q, next_state, out);
        auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::FP32,
                                                                     "QwenGatedDelta");
        ASSERT_NE(kernel, nullptr) << static_cast<int>(device);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->compute(), 0) << static_cast<int>(device);
        ASSERT_EQ(next_state->data_type(), feather::DataType::FP32);
        ASSERT_EQ(out->data_type(), feather::DataType::FP32);
        for (size_t index = 0; index < expected_state.size(); ++index) {
            EXPECT_NEAR(next_state->data<float>()[index], expected_state[index], 1e-5f)
                << "device=" << static_cast<int>(device) << " state index=" << index;
        }
        for (size_t index = 0; index < expected_output.size(); ++index) {
            EXPECT_NEAR(out->data<float>()[index], expected_output[index], 1e-5f)
                << "device=" << static_cast<int>(device) << " output index=" << index;
        }
    }
}

#ifdef FEATHER_WITH_CUDA
TEST(qwen_gated_delta_kernel_test, CudaStateOutputAndCombinedMatchCommon) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    constexpr int64_t heads = 2;
    constexpr int64_t key = 3;
    constexpr int64_t value = 5;
    std::vector<float> state_values(static_cast<size_t>(heads * key * value));
    std::vector<float> k_values(static_cast<size_t>(heads * key));
    std::vector<float> v_values(static_cast<size_t>(heads * value));
    std::vector<float> beta_values(static_cast<size_t>(heads));
    std::vector<float> decay_values(static_cast<size_t>(heads));
    std::vector<float> q_values(static_cast<size_t>(heads * key));
    for (size_t index = 0; index < state_values.size(); ++index) {
        state_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125f;
    }
    for (size_t index = 0; index < k_values.size(); ++index) {
        k_values[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.0625f;
        q_values[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.05f;
    }
    for (size_t index = 0; index < v_values.size(); ++index) {
        v_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
    }
    beta_values = {0.2f, 0.35f};
    decay_values = {0.9f, 0.8f};

    const auto state = MakeTensor(state_values, {1, heads, key, value});
    const auto k = MakeTensor(k_values, {1, heads, key});
    const auto v = MakeTensor(v_values, {1, heads, value});
    const auto beta = MakeTensor(beta_values, {1, heads, 1});
    const auto decay = MakeTensor(decay_values, {1, heads, 1, 1});
    const auto q = MakeTensor(q_values, {1, heads, key});
    const auto common_state_out = MakeTensor(std::vector<float>(state_values.size()), {1, heads, key, value});
    const auto cuda_state_out = MakeTensor(std::vector<float>(state_values.size()), {1, heads, key, value});

    auto common_state_param = MakeStateParam(state, k, v, beta, decay, common_state_out);
    auto common_state = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                       feather::DataType::FP32,
                                                                       "QwenGatedDeltaState");
    ASSERT_NE(common_state, nullptr);
    common_state->SetParam(&common_state_param);
    ASSERT_EQ(common_state->compute(), 0);

    auto cuda_state_param = MakeStateParam(state, k, v, beta, decay, cuda_state_out);
    auto cuda_state = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA,
                                                                     feather::DataType::FP32,
                                                                     "QwenGatedDeltaState");
    ASSERT_NE(cuda_state, nullptr);
    cuda_state->SetParam(&cuda_state_param);
    ASSERT_EQ(cuda_state->compute(), 0);

    for (size_t index = 0; index < state_values.size(); ++index) {
        EXPECT_NEAR(cuda_state_out->data<float>()[index], common_state_out->data<float>()[index], 1e-5f)
            << "state index=" << index;
    }

    const auto common_output = MakeTensor(std::vector<float>(static_cast<size_t>(heads * value)), {1, heads, value});
    const auto cuda_output = MakeTensor(std::vector<float>(static_cast<size_t>(heads * value)), {1, heads, value});
    feather::operators::QwenGatedDeltaOutputParam common_output_param{};
    common_output_param.state = common_state_out;
    common_output_param.q = q;
    common_output_param.out = common_output;
    auto common_output_kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                                feather::DataType::FP32,
                                                                                "QwenGatedDeltaOutput");
    ASSERT_NE(common_output_kernel, nullptr);
    common_output_kernel->SetParam(&common_output_param);
    ASSERT_EQ(common_output_kernel->compute(), 0);

    auto cuda_output_param = common_output_param;
    cuda_output_param.out = cuda_output;
    auto cuda_output_kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA,
                                                                              feather::DataType::FP32,
                                                                              "QwenGatedDeltaOutput");
    ASSERT_NE(cuda_output_kernel, nullptr);
    cuda_output_kernel->SetParam(&cuda_output_param);
    ASSERT_EQ(cuda_output_kernel->compute(), 0);
    for (size_t index = 0; index < static_cast<size_t>(heads * value); ++index) {
        EXPECT_NEAR(cuda_output->data<float>()[index], common_output->data<float>()[index], 1e-5f)
            << "output index=" << index;
    }

    const auto common_next_state = MakeTensor(std::vector<float>(state_values.size()), {1, heads, key, value});
    const auto common_combined_output = MakeTensor(std::vector<float>(static_cast<size_t>(heads * value)),
                                                   {1, heads, value});
    const auto cuda_next_state = MakeTensor(std::vector<float>(state_values.size()), {1, heads, key, value});
    const auto cuda_combined_output = MakeTensor(std::vector<float>(static_cast<size_t>(heads * value)),
                                                 {1, heads, value});
    auto common_combined_param = MakeCombinedParam(state, k, v, beta, decay, q, common_next_state,
                                                   common_combined_output);
    auto common_combined = feather::KernelDispatcher::instance().create(feather::DeviceType::COMMON,
                                                                           feather::DataType::FP32,
                                                                           "QwenGatedDelta");
    ASSERT_NE(common_combined, nullptr);
    common_combined->SetParam(&common_combined_param);
    ASSERT_EQ(common_combined->compute(), 0);

    auto cuda_combined_param = MakeCombinedParam(state, k, v, beta, decay, q, cuda_next_state,
                                                 cuda_combined_output);
    auto cuda_combined = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA,
                                                                        feather::DataType::FP32,
                                                                        "QwenGatedDelta");
    ASSERT_NE(cuda_combined, nullptr);
    cuda_combined->SetParam(&cuda_combined_param);
    ASSERT_EQ(cuda_combined->compute(), 0);
    for (size_t index = 0; index < state_values.size(); ++index) {
        EXPECT_NEAR(cuda_next_state->data<float>()[index], common_next_state->data<float>()[index], 1e-5f)
            << "combined state index=" << index;
    }
    for (size_t index = 0; index < static_cast<size_t>(heads * value); ++index) {
        EXPECT_NEAR(cuda_combined_output->data<float>()[index], common_combined_output->data<float>()[index], 1e-5f)
            << "combined output index=" << index;
    }
}
#endif

}  // namespace
