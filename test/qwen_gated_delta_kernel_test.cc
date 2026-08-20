#include <gtest/gtest.h>

#include <memory>
#include <vector>

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

}  // namespace
