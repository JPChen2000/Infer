#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/batch_normalization.h"
#include "src/kernel/cast.h"
#include "src/kernel/div.h"
#include "src/kernel/erf.h"
#include "src/kernel/equal.h"
#include "src/kernel/expand.h"
#include "src/kernel/gather.h"
#include "src/kernel/global_average_pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/reduce_mean.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/squeeze.h"
#include "src/kernel/sub.h"
#include "src/kernel/tanh.h"
#include "src/kernel/unsqueeze.h"
#include "src/kernel/where.h"
#include "src/operator/params.h"
#include "util/fp16.h"

namespace {

#ifdef FEATHER_WITH_CUDA

bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <feather::DataType dtype>
std::shared_ptr<feather::Tensor> MakeFloatingTensor(const std::vector<float>& values,
                                                     const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<feather::Tensor>();
    if constexpr (dtype == feather::DataType::FP32) {
        tensor->Assign<float>(values, shape);
    } else {
        std::vector<uint16_t> storage;
        storage.reserve(values.size());
        for (const float value : values) {
            storage.push_back(feather::FloatToHalf(value));
        }
        tensor->Assign<uint16_t>(storage, shape);
    }
    return tensor;
}

template <feather::DataType dtype>
std::shared_ptr<feather::Tensor> MakeFloatingOutput(const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<feather::Tensor>(shape);
    if constexpr (dtype == feather::DataType::FP32) {
        tensor->mutable_data<float>();
    } else {
        tensor->mutable_data<uint16_t>();
    }
    return tensor;
}

float ReadFloatingValue(const feather::Tensor& tensor, int64_t index) {
    if (tensor.data_type() == feather::DataType::FP32) {
        return tensor.data<float>()[index];
    }
    if (tensor.data_type() == feather::DataType::FP16) {
        return feather::HalfToFloat(tensor.data<uint16_t>()[index]);
    }
    return 0.0f;
}

template <feather::DataType dtype>
std::unique_ptr<feather::KernelBase> CreateCudaKernel(const char* op_type) {
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, dtype, op_type);
    EXPECT_NE(kernel, nullptr) << "missing CUDA kernel for " << op_type;
    if (kernel != nullptr) {
        EXPECT_EQ(kernel->device(), feather::DeviceType::CUDA) << "CUDA request fell back for " << op_type;
    }
    return kernel;
}

template <feather::DataType dtype>
void RunClassificationNumericOperators() {
    auto bn_input = MakeFloatingTensor<dtype>({1.0f, 3.0f, 2.0f, 4.0f}, {1, 2, 1, 2});
    auto bn_scale = MakeFloatingTensor<dtype>({2.0f, 3.0f}, {2});
    auto bn_bias = MakeFloatingTensor<dtype>({0.5f, -1.0f}, {2});
    auto bn_mean = MakeFloatingTensor<dtype>({1.0f, 2.0f}, {2});
    auto bn_var = MakeFloatingTensor<dtype>({4.0f, 1.0f}, {2});
    auto bn_out = MakeFloatingOutput<dtype>({1, 2, 1, 2});
    feather::operators::BatchNormParam bn_param{};
    bn_param.input = bn_input;
    bn_param.scale = bn_scale;
    bn_param.bias = bn_bias;
    bn_param.mean = bn_mean;
    bn_param.var = bn_var;
    bn_param.out = bn_out;
    bn_param.epsilon = 0.0f;
    auto batch_norm = CreateCudaKernel<dtype>("BatchNormalization");
    if (batch_norm == nullptr) {
        return;
    }
    batch_norm->SetParam(&bn_param);
    EXPECT_EQ(batch_norm->compute(), 0);
    const std::vector<float> expected_bn = {0.5f, 2.5f, -1.0f, 5.0f};
    for (size_t i = 0; i < expected_bn.size(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*bn_out, static_cast<int64_t>(i)), expected_bn[i], 3e-3f);
    }

    auto gap_input = MakeFloatingTensor<dtype>({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f},
                                                {1, 2, 2, 2});
    gap_input->set_layout(feather::DataLayout::NHWC);
    auto gap_out = MakeFloatingOutput<dtype>({1, 1, 1, 2});
    gap_out->set_layout(feather::DataLayout::NHWC);
    feather::operators::GlobalAveragePoolParam gap_param{};
    gap_param.input = gap_input;
    gap_param.out = gap_out;
    auto global_average_pool = CreateCudaKernel<dtype>("GlobalAveragePool");
    if (global_average_pool == nullptr) {
        return;
    }
    global_average_pool->SetParam(&gap_param);
    EXPECT_EQ(global_average_pool->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*gap_out, 0), 2.5f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*gap_out, 1), 25.0f, 3e-3f);

    auto lhs = MakeFloatingTensor<dtype>({4.0f, 9.0f, 8.0f, 12.0f}, {2, 2});
    auto rhs = MakeFloatingTensor<dtype>({2.0f, 3.0f}, {1, 2});
    auto sub_out = MakeFloatingOutput<dtype>({2, 2});
    feather::operators::BinaryParam binary_param{};
    binary_param.lhs = lhs;
    binary_param.rhs = rhs;
    binary_param.out = sub_out;
    auto sub = CreateCudaKernel<dtype>("Sub");
    if (sub == nullptr) {
        return;
    }
    sub->SetParam(&binary_param);
    EXPECT_EQ(sub->compute(), 0);
    const std::vector<float> expected_sub = {2.0f, 6.0f, 6.0f, 9.0f};
    for (size_t i = 0; i < expected_sub.size(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*sub_out, static_cast<int64_t>(i)), expected_sub[i], 3e-3f);
    }

    auto div_out = MakeFloatingOutput<dtype>({2, 2});
    binary_param.out = div_out;
    auto div = CreateCudaKernel<dtype>("Div");
    if (div == nullptr) {
        return;
    }
    div->SetParam(&binary_param);
    EXPECT_EQ(div->compute(), 0);
    const std::vector<float> expected_div = {2.0f, 3.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < expected_div.size(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*div_out, static_cast<int64_t>(i)), expected_div[i], 3e-3f);
    }

    auto unary_input = MakeFloatingTensor<dtype>({0.0f, 1.0f, 4.0f}, {3});
    feather::operators::UnaryParam unary_param{};
    unary_param.input = unary_input;
    for (const auto& item : std::vector<std::pair<const char*, std::vector<float>>>{
             {"Sqrt", {0.0f, 1.0f, 2.0f}},
             {"Tanh", {0.0f, std::tanh(1.0f), std::tanh(4.0f)}},
             {"Erf", {0.0f, std::erf(1.0f), std::erf(4.0f)}},
         }) {
        unary_param.out = MakeFloatingOutput<dtype>({3});
        auto unary = CreateCudaKernel<dtype>(item.first);
        if (unary == nullptr) {
            return;
        }
        unary->SetParam(&unary_param);
        EXPECT_EQ(unary->compute(), 0);
        for (size_t i = 0; i < item.second.size(); ++i) {
            EXPECT_NEAR(ReadFloatingValue(*unary_param.out, static_cast<int64_t>(i)), item.second[i], 3e-3f);
        }
    }

    auto pow_input = MakeFloatingTensor<dtype>({2.0f, 3.0f}, {2});
    auto pow_output = MakeFloatingOutput<dtype>({2});
    feather::operators::PowParam pow_param{};
    pow_param.input = pow_input;
    pow_param.out = pow_output;
    pow_param.exponent = 2.0f;
    auto pow = CreateCudaKernel<dtype>("Pow");
    if (pow == nullptr) {
        return;
    }
    pow->SetParam(&pow_param);
    EXPECT_EQ(pow->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*pow_output, 0), 4.0f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*pow_output, 1), 9.0f, 3e-3f);
}

template <feather::DataType dtype>
void RunTransformerNumericOperators() {
    auto data = MakeFloatingTensor<dtype>({1.0f, 2.0f}, {2});
    auto unsqueezed = MakeFloatingOutput<dtype>({1, 2, 1});
    feather::operators::AxesParam axes_param{};
    axes_param.input = data;
    axes_param.out = unsqueezed;
    axes_param.axes = {0, 2};
    auto unsqueeze = CreateCudaKernel<dtype>("Unsqueeze");
    if (unsqueeze == nullptr) {
        return;
    }
    unsqueeze->SetParam(&axes_param);
    EXPECT_EQ(unsqueeze->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*unsqueezed, 0), 1.0f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*unsqueezed, 1), 2.0f, 3e-3f);

    auto squeezed = MakeFloatingOutput<dtype>({2});
    axes_param.input = unsqueezed;
    axes_param.out = squeezed;
    auto squeeze = CreateCudaKernel<dtype>("Squeeze");
    if (squeeze == nullptr) {
        return;
    }
    squeeze->SetParam(&axes_param);
    EXPECT_EQ(squeeze->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*squeezed, 0), 1.0f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*squeezed, 1), 2.0f, 3e-3f);

    auto cast_out = MakeFloatingOutput<dtype == feather::DataType::FP32 ? feather::DataType::FP16
                                                                         : feather::DataType::FP32>({2});
    feather::operators::CastParam cast_param{};
    cast_param.input = data;
    cast_param.out = cast_out;
    cast_param.to = dtype == feather::DataType::FP32 ? feather::DataType::FP16 : feather::DataType::FP32;
    auto cast = CreateCudaKernel<dtype>("Cast");
    if (cast == nullptr) {
        return;
    }
    cast->SetParam(&cast_param);
    EXPECT_EQ(cast->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*cast_out, 0), 1.0f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*cast_out, 1), 2.0f, 3e-3f);

    auto reduce_input = MakeFloatingTensor<dtype>({1.0f, 3.0f, 5.0f, 7.0f}, {2, 2});
    auto reduce_out = MakeFloatingOutput<dtype>({2, 1});
    feather::operators::ReduceMeanParam reduce_param{};
    reduce_param.input = reduce_input;
    reduce_param.out = reduce_out;
    reduce_param.axes = {1};
    reduce_param.keepdims = true;
    auto reduce_mean = CreateCudaKernel<dtype>("ReduceMean");
    if (reduce_mean == nullptr) {
        return;
    }
    reduce_mean->SetParam(&reduce_param);
    EXPECT_EQ(reduce_mean->compute(), 0);
    EXPECT_NEAR(ReadFloatingValue(*reduce_out, 0), 2.0f, 3e-3f);
    EXPECT_NEAR(ReadFloatingValue(*reduce_out, 1), 6.0f, 3e-3f);

    auto gather_data = MakeFloatingTensor<dtype>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    auto indices = std::make_shared<feather::Tensor>();
    indices->Assign<int64_t>({2, 0}, {2});
    auto gather_out = MakeFloatingOutput<dtype>({2, 2});
    feather::operators::GatherParam gather_param{};
    gather_param.data = gather_data;
    gather_param.indices = indices;
    gather_param.out = gather_out;
    gather_param.axis = 0;
    auto gather = CreateCudaKernel<dtype>("Gather");
    if (gather == nullptr) {
        return;
    }
    gather->SetParam(&gather_param);
    EXPECT_EQ(gather->compute(), 0);
    const std::vector<float> expected_gather = {5.0f, 6.0f, 1.0f, 2.0f};
    for (size_t i = 0; i < expected_gather.size(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*gather_out, static_cast<int64_t>(i)), expected_gather[i], 3e-3f);
    }

    auto equal_lhs = MakeFloatingTensor<dtype>({1.0f, 2.0f, 1.0f, 4.0f}, {2, 2});
    auto equal_rhs = MakeFloatingTensor<dtype>({1.0f}, {});
    auto equal_out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2});
    equal_out->mutable_data<uint8_t>();
    equal_out->set_data_type(feather::DataType::BOOL);
    feather::operators::EqualParam equal_param{};
    equal_param.lhs = equal_lhs;
    equal_param.rhs = equal_rhs;
    equal_param.out = equal_out;
    auto equal = CreateCudaKernel<dtype>("Equal");
    if (equal == nullptr) {
        return;
    }
    equal->SetParam(&equal_param);
    EXPECT_EQ(equal->compute(), 0);
    EXPECT_EQ(equal_out->data<uint8_t>()[0], 1);
    EXPECT_EQ(equal_out->data<uint8_t>()[1], 0);
    EXPECT_EQ(equal_out->data<uint8_t>()[2], 1);
    EXPECT_EQ(equal_out->data<uint8_t>()[3], 0);

    auto expand_input = MakeFloatingTensor<dtype>({1.0f, 2.0f}, {1, 2});
    auto expand_shape = std::make_shared<feather::Tensor>();
    expand_shape->Assign<int64_t>({3, 2}, {2});
    auto expand_out = MakeFloatingOutput<dtype>({3, 2});
    feather::operators::ExpandParam expand_param{};
    expand_param.input = expand_input;
    expand_param.shape = expand_shape;
    expand_param.out = expand_out;
    auto expand = CreateCudaKernel<dtype>("Expand");
    if (expand == nullptr) {
        return;
    }
    expand->SetParam(&expand_param);
    EXPECT_EQ(expand->compute(), 0);
    for (int64_t i = 0; i < expand_out->numel(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*expand_out, i), i % 2 == 0 ? 1.0f : 2.0f, 3e-3f);
    }

    auto condition = std::make_shared<feather::Tensor>();
    condition->Assign<uint8_t>({1, 0}, {1, 2});
    condition->set_data_type(feather::DataType::BOOL);
    auto where_x = MakeFloatingTensor<dtype>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    auto where_y = MakeFloatingTensor<dtype>({-1.0f}, {});
    auto where_out = MakeFloatingOutput<dtype>({3, 2});
    feather::operators::WhereParam where_param{};
    where_param.condition = condition;
    where_param.x = where_x;
    where_param.y = where_y;
    where_param.out = where_out;
    auto where = CreateCudaKernel<dtype>("Where");
    if (where == nullptr) {
        return;
    }
    where->SetParam(&where_param);
    EXPECT_EQ(where->compute(), 0);
    for (int64_t i = 0; i < where_out->numel(); ++i) {
        EXPECT_NEAR(ReadFloatingValue(*where_out, i), i % 2 == 0 ? static_cast<float>(i + 1) : -1.0f, 3e-3f);
    }
}

TEST(cuda_extended_ops_test, ClassificationNumericOperatorsRunOnCudaFp32AndFp16) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    RunClassificationNumericOperators<feather::DataType::FP32>();
    RunClassificationNumericOperators<feather::DataType::FP16>();
}

TEST(cuda_extended_ops_test, TransformerNumericOperatorsRunOnCudaFp32AndFp16) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    RunTransformerNumericOperators<feather::DataType::FP32>();
    RunTransformerNumericOperators<feather::DataType::FP16>();
}

#endif

}  // namespace
