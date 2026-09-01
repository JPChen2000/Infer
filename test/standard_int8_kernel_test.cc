#include <gtest/gtest.h>

#include <memory>
#include <vector>
#include <typeinfo>

#include "core/kernel.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/add.h"
#endif
#include "core/tensor.h"
#include "quant/quantization.h"
#include "src/operator/elementwise_utils.h"
#include "src/operator/params.h"

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::QuantizationParams;
using feather::Tensor;

QuantizationParams Q(float scale, int32_t zero_point = 0) {
    QuantizationParams params;
    params.enabled = true;
    params.scale = scale;
    params.zero_point = zero_point;
    params.scales = {scale};
    params.zero_points = {zero_point};
    return params;
}

std::shared_ptr<Tensor> Int8(std::vector<int8_t> values, std::vector<int64_t> dims,
                             const QuantizationParams& quantization = Q(1.0f)) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<int8_t>(values, dims);
    tensor->set_quantization(quantization);
    return tensor;
}

std::shared_ptr<Tensor> Output(std::vector<int64_t> dims, const QuantizationParams& quantization = Q(1.0f)) {
    auto tensor = std::make_shared<Tensor>();
    size_t numel = 1;
    for (const int64_t dim : dims) numel *= static_cast<size_t>(dim);
    tensor->Assign<int8_t>(std::vector<int8_t>(numel, 0), dims);
    tensor->set_quantization(quantization);
    return tensor;
}

std::shared_ptr<Tensor> Float(std::vector<float> values, std::vector<int64_t> dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<float>(values, dims);
    return tensor;
}

std::shared_ptr<Tensor> Bool(std::vector<uint8_t> values, std::vector<int64_t> dims) {
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<uint8_t>(values, dims);
    tensor->set_data_type(DataType::BOOL);
    return tensor;
}

std::unique_ptr<feather::KernelBase> Common(const char* op) {
    return KernelDispatcher::instance().create(DeviceType::COMMON, DataType::INT8, op);
}

std::unique_ptr<feather::KernelBase> X86(const char* op) {
    return KernelDispatcher::instance().create(DeviceType::X86, DataType::INT8, op);
}

const char* const kStandardInt8Operators[] = {
    "Add", "Sub", "Mul", "Div", "ReLU", "Neg", "Sigmoid", "SiLU", "Exp", "Sqrt",
    "Tanh", "Erf", "Sin", "Cos", "Softplus", "Pow", "BatchNormalization",
    "AvgPool", "MaxPool", "GlobalAveragePool", "ReduceSum", "ReduceMean", "Identity",
    "Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Split",
    "Concat", "Expand", "Gather", "Where", "Resize", "Softmax", "Equal", "Cast",
    "ConstantOfShape", "Shape", "ResizeConcat", "YoloDecode"};

TEST(StandardInt8KernelTest, CommonRegistersEveryStandardDataOperator) {
    for (const char* op : kStandardInt8Operators) {
        EXPECT_NE(Common(op), nullptr) << op;
    }
}

TEST(StandardInt8KernelTest, X86RegistersEveryStandardDataOperator) {
    for (const char* op : kStandardInt8Operators) {
        EXPECT_NE(X86(op), nullptr) << op;
    }
}

#if defined(FEATHER_WITH_CUDA)
TEST(StandardInt8KernelTest, CudaRegistersEveryStandardDataOperator) {
    for (const char* op : kStandardInt8Operators) {
        EXPECT_NE(KernelDispatcher::instance().create(DeviceType::CUDA, DataType::INT8, op), nullptr) << op;
    }
}
#endif

#ifdef FEATHER_WITH_CUDA
TEST(StandardInt8KernelTest, CudaInt8DoesNotUseCommonKernel) {
    auto cuda_kernel = KernelDispatcher::instance().create(DeviceType::CUDA, DataType::INT8, "Add");
    ASSERT_NE(cuda_kernel, nullptr);
    EXPECT_EQ(typeid(*cuda_kernel),
              typeid(feather::kernel::AddKernel<DeviceType::CUDA, DataType::INT8>));
    EXPECT_NE(typeid(*cuda_kernel),
              typeid(feather::kernel::AddKernel<DeviceType::COMMON, DataType::INT8>));
}
#endif

TEST(StandardInt8KernelTest, ElementwiseUsesRealValuesAndRequantizes) {
    auto lhs = Int8({5, -2}, {1, 2}, Q(0.5f, 3));
    auto rhs = Int8({2, 4}, {1, 2}, Q(0.25f, -1));
    auto output = Output({1, 2}, Q(0.25f, 10));
    feather::operators::BinaryParam param;
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = output;
    auto kernel = Common("Add");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 17);
    EXPECT_EQ(output->data<int8_t>()[1], 5);
}

TEST(StandardInt8KernelTest, NonlinearAndReductionStayInInt8Domain) {
    auto input = Int8({1, 3, -2, 4}, {1, 4}, Q(0.5f));
    auto output = Output({1, 4}, Q(0.5f));
    feather::operators::UnaryParam unary;
    unary.input = input;
    unary.out = output;
    auto relu = Common("ReLU");
    ASSERT_NE(relu, nullptr);
    relu->SetParam(&unary);
    ASSERT_EQ(relu->compute(), 0);
    EXPECT_EQ(output->data<int8_t>()[0], 1);
    EXPECT_EQ(output->data<int8_t>()[2], 0);

    auto reduce_output = Output({1, 2}, Q(0.01f));
    feather::operators::ReduceMeanParam reduce;
    reduce.input = input;
    reduce.out = reduce_output;
    reduce.axes = {1};
    reduce.keepdims = true;
    auto mean = Common("ReduceMean");
    ASSERT_NE(mean, nullptr);
    mean->SetParam(&reduce);
    EXPECT_EQ(mean->compute(), 0);
}

TEST(StandardInt8KernelTest, SoftmaxAndBatchNormalizationProduceInt8) {
    auto input = Int8({0, 0}, {1, 2}, Q(1.0f));
    auto output = Output({1, 2}, Q(0.01f));
    feather::operators::SoftmaxParam softmax;
    softmax.input = input;
    softmax.out = output;
    softmax.axis = 1;
    auto kernel = Common("Softmax");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&softmax);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(output->data<int8_t>()[0], 50, 1);
    EXPECT_NEAR(output->data<int8_t>()[1], 50, 1);

    auto bn_input = Int8({2}, {1, 1, 1, 1}, Q(0.5f));
    auto bn_output = Output({1, 1, 1, 1}, Q(0.25f));
    feather::operators::BatchNormParam bn;
    bn.input = bn_input;
    bn.scale = Float({2.0f}, {1});
    bn.bias = Float({1.0f}, {1});
    bn.mean = Float({0.0f}, {1});
    bn.var = Float({1.0f}, {1});
    bn.out = bn_output;
    auto bn_kernel = Common("BatchNormalization");
    ASSERT_NE(bn_kernel, nullptr);
    bn_kernel->SetParam(&bn);
    ASSERT_EQ(bn_kernel->compute(), 0);
    EXPECT_EQ(bn_output->data<int8_t>()[0], 12);
}

TEST(StandardInt8KernelTest, SliceUsesAllOutputCoordinates) {
    auto input = Int8({1, 2, 3, 4, 5, 6}, {2, 3}, Q(0.5f));
    auto output = Output({2, 2}, Q(0.25f));
    feather::operators::SliceParam slice;
    slice.input = input;
    slice.out = output;
    slice.axis = 1;
    slice.start = 1;
    slice.end = 3;
    auto kernel = Common("Slice");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&slice);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(output->data<int8_t>(), output->data<int8_t>() + 4),
              (std::vector<int8_t>{4, 6, 10, 12}));
}

TEST(StandardInt8KernelTest, WhereBroadcastsConditionAndValues) {
    auto condition = Bool({1, 0}, {2, 1});
    auto x = Int8({2, 4, 6}, {1, 3}, Q(1.0f));
    auto y = Int8({10, 11, 12, 13, 14, 15}, {2, 3}, Q(1.0f));
    auto output = Output({2, 3}, Q(1.0f));
    feather::operators::WhereParam where;
    where.condition = condition;
    where.x = x;
    where.y = y;
    where.out = output;
    auto kernel = Common("Where");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&where);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_EQ(std::vector<int8_t>(output->data<int8_t>(), output->data<int8_t>() + 6),
              (std::vector<int8_t>{2, 4, 6, 13, 14, 15}));
}

TEST(StandardInt8KernelTest, FrontendOutputInferencePreservesInt8Quantization) {
    auto input = Int8({1, 2}, {1, 2}, Q(0.5f, -3));
    auto output = std::make_shared<Tensor>();
    output->set_data_type(DataType::INT8);
    output->set_quantization(Q(0.125f, 7));
    feather::operators::UnaryParam unary;
    unary.input = input;
    unary.out = output;
    ASSERT_EQ(feather::operators::elementwise_detail::InferUnaryOutput(&unary), 0);
    ASSERT_NE(unary.out, nullptr);
    EXPECT_EQ(unary.out->data_type(), DataType::INT8);
    EXPECT_TRUE(unary.out->quantization().enabled);
    EXPECT_FLOAT_EQ(unary.out->quantization().scale, 0.125f);
    EXPECT_EQ(unary.out->quantization().zero_point, 7);
    EXPECT_EQ(unary.out->dims().data(), (std::vector<int64_t>{1, 2}));
}



TEST(StandardInt8KernelTest, Int8FusedOperatorsExecuteOnCommonAndX86) {
    feather::kernel::EnsureBuiltinKernelsRegistered();

    auto make_int8 = [](const std::vector<int8_t>& values, const std::vector<int64_t>& dims,
                        float scale = 1.0f, int32_t zero_point = 0) {
        auto tensor = std::make_shared<feather::Tensor>();
        tensor->Assign<int8_t>(values, dims);
        feather::QuantizationParams params;
        params.enabled = true;
        params.granularity = feather::QuantizationGranularity::kPerTensor;
        params.scales = {scale};
        params.zero_points = {zero_point};
        tensor->set_quantization(params);
        return tensor;
    };
    auto make_float = [](const std::vector<float>& values, const std::vector<int64_t>& dims) {
        auto tensor = std::make_shared<feather::Tensor>();
        tensor->Assign<float>(values, dims);
        return tensor;
    };

    for (const auto device : {feather::DeviceType::COMMON, feather::DeviceType::X86}) {
        SCOPED_TRACE(device == feather::DeviceType::COMMON ? "COMMON" : "X86");
        {
            auto input = make_int8(std::vector<int8_t>{1, 2, 3, 4, 5, 6}, std::vector<int64_t>{1, 2, 3});
            auto output = std::make_shared<feather::Tensor>();
            output->Assign<int64_t>({0, 0, 0}, {3});
            feather::operators::ShapeParam param;
            param.input = input;
            param.out = output;
            param.start = 0;
            param.end = 3;
            auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::INT8, "Shape");
            ASSERT_NE(kernel, nullptr);
            kernel->SetParam(&param);
            ASSERT_EQ(kernel->compute(), 0);
            ASSERT_EQ(output->data_type(), feather::DataType::INT64);
            EXPECT_EQ(output->data<int64_t>()[0], 1);
            EXPECT_EQ(output->data<int64_t>()[1], 2);
            EXPECT_EQ(output->data<int64_t>()[2], 3);
        }

        {
            auto resize_input = make_int8(std::vector<int8_t>{2}, std::vector<int64_t>{1, 1, 1, 1});
            auto concat_input = make_int8(std::vector<int8_t>{3, 3, 3, 3}, std::vector<int64_t>{1, 1, 2, 2});
            auto output = make_int8(std::vector<int8_t>(8, 0), std::vector<int64_t>{1, 2, 2, 2});
            feather::operators::ResizeConcatParam param;
            param.resize_input = resize_input;
            param.concat_input = concat_input;
            param.out = output;
            param.scales = {1.0f, 1.0f, 2.0f, 2.0f};
            param.axis = 1;
            param.resize_input_index = 0;
            auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::INT8, "ResizeConcat");
            ASSERT_NE(kernel, nullptr);
            kernel->SetParam(&param);
            ASSERT_EQ(kernel->compute(), 0);
            for (int i = 0; i < 4; ++i) EXPECT_EQ(output->data<int8_t>()[i], 2);
            for (int i = 4; i < 8; ++i) EXPECT_EQ(output->data<int8_t>()[i], 3);
        }

        {
            auto input = make_int8(std::vector<int8_t>(12, 0), std::vector<int64_t>{1, 12, 1, 1});
            auto output = make_int8(std::vector<int8_t>(12, 0), std::vector<int64_t>{1, 2, 6});
            auto xy_scale = make_float({2.0f, 2.0f}, {2});
            auto grid = make_float({1.0f, 1.0f}, {2});
            auto stride = make_float({4.0f}, {1});
            auto wh_scale = make_float({2.0f, 2.0f}, {2});
            auto anchor_grid = make_float({4.0f, 4.0f}, {2});
            feather::operators::YoloDecodeParam param;
            param.input = input;
            param.xy_scale = xy_scale;
            param.grid = grid;
            param.stride = stride;
            param.wh_scale = wh_scale;
            param.anchor_grid = anchor_grid;
            param.out = output;
            auto kernel = feather::KernelDispatcher::instance().create(device, feather::DataType::INT8, "YoloDecode");
            ASSERT_NE(kernel, nullptr);
            kernel->SetParam(&param);
            ASSERT_EQ(kernel->compute(), 0);
            EXPECT_EQ(output->data<int8_t>()[0], 8);
            EXPECT_EQ(output->data<int8_t>()[1], 8);
            EXPECT_EQ(output->data<int8_t>()[2], 4);
            EXPECT_EQ(output->data<int8_t>()[3], 4);
            EXPECT_EQ(output->data<int8_t>()[4], 1);
            EXPECT_EQ(output->data<int8_t>()[5], 1);
            EXPECT_EQ(output->data<int8_t>()[6], 8);
            EXPECT_EQ(output->data<int8_t>()[7], 8);
        }
    }
}

}  // namespace
