#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/operator/params.h"
#include "util/bf16.h"
#include "util/fp8.h"
#include "util/types.h"

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#include "src/kernel/cuda/runtime.h"
#endif

namespace {

using feather::DataType;
using feather::DeviceType;
using feather::Fp8E4M3;
using feather::Fp8E5M2;
using feather::Tensor;

template <typename T>
std::shared_ptr<Tensor> MakeFp8Tensor(const std::vector<float>& values, const std::vector<int64_t>& dims,
                                      float scale) {
    std::vector<T> encoded;
    encoded.reserve(values.size());
    for (const float value : values) {
        if constexpr (std::is_same_v<T, Fp8E4M3>) {
            encoded.push_back({feather::FloatToFp8E4M3(value / scale)});
        } else {
            encoded.push_back({feather::FloatToFp8E5M2(value / scale)});
        }
    }
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign<T>(encoded, dims);
    tensor->set_quantization({true, scale});
    return tensor;
}

template <typename T>
float ReadFp8Value(const Tensor& tensor, int64_t index) {
    if constexpr (std::is_same_v<T, Fp8E4M3>) {
        return feather::Fp8E4M3ToFloat(tensor.data<Fp8E4M3>()[index].bits) * tensor.quantization_scale();
    }
    return feather::Fp8E5M2ToFloat(tensor.data<Fp8E5M2>()[index].bits) * tensor.quantization_scale();
}

template <typename T>
void VerifyFp8IndexBroadcastAndReduceOps(DeviceType device, float tolerance) {
    const DataType dtype = feather::DataTypeTrait<T>::type();
    auto input = MakeFp8Tensor<T>({1.0f, 2.0f, 4.0f, 8.0f}, {2, 2}, 0.5f);

    auto reduce_output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    reduce_output->set_data_type(dtype);
    reduce_output->set_quantization({true, 0.5f});
    auto reduce = feather::KernelDispatcher::instance().create(device, dtype, "ReduceSum");
    ASSERT_NE(reduce, nullptr);
    feather::operators::ReduceSumParam reduce_param{};
    reduce_param.input = input;
    reduce_param.out = reduce_output;
    reduce_param.axes = {1};
    reduce_param.keepdims = false;
    reduce->SetParam(&reduce_param);
    ASSERT_EQ(reduce->compute(), 0) << "device=" << static_cast<int>(device);
    EXPECT_NEAR(ReadFp8Value<T>(*reduce_output, 0), 3.0f, tolerance);
    EXPECT_NEAR(ReadFp8Value<T>(*reduce_output, 1), 12.0f, tolerance);

    auto indices = std::make_shared<Tensor>();
    indices->Assign<int32_t>({1, 0}, {2});
    auto gather_output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    gather_output->set_data_type(dtype);
    gather_output->set_quantization({true, 0.5f});
    auto gather = feather::KernelDispatcher::instance().create(device, dtype, "Gather");
    ASSERT_NE(gather, nullptr);
    feather::operators::GatherParam gather_param{};
    gather_param.data = input;
    gather_param.indices = indices;
    gather_param.out = gather_output;
    gather_param.axis = 0;
    gather->SetParam(&gather_param);
    ASSERT_EQ(gather->compute(), 0) << "device=" << static_cast<int>(device);
    for (const auto& [index, expected] : std::vector<std::pair<int64_t, float>>{{0, 4.0f}, {1, 8.0f},
                                                                                   {2, 1.0f}, {3, 2.0f}}) {
        EXPECT_NEAR(ReadFp8Value<T>(*gather_output, index), expected, tolerance);
    }

    auto expand_input = MakeFp8Tensor<T>({1.0f, 2.0f}, {2, 1}, 0.5f);
    auto shape = std::make_shared<Tensor>();
    shape->Assign<int64_t>({2, 3}, {2});
    auto expand_output = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});
    expand_output->set_data_type(dtype);
    expand_output->set_quantization({true, 0.5f});
    auto expand = feather::KernelDispatcher::instance().create(device, dtype, "Expand");
    ASSERT_NE(expand, nullptr);
    feather::operators::ExpandParam expand_param{};
    expand_param.input = expand_input;
    expand_param.shape = shape;
    expand_param.out = expand_output;
    expand->SetParam(&expand_param);
    ASSERT_EQ(expand->compute(), 0) << "device=" << static_cast<int>(device);
    for (const auto& [index, expected] : std::vector<std::pair<int64_t, float>>{{0, 1.0f}, {1, 1.0f}, {2, 1.0f},
                                                                                   {3, 2.0f}, {4, 2.0f}, {5, 2.0f}}) {
        EXPECT_NEAR(ReadFp8Value<T>(*expand_output, index), expected, tolerance);
    }

    auto condition = std::make_shared<Tensor>(std::vector<int64_t>{2, 1});
    auto* condition_data = condition->mutable_data<uint8_t>();
    condition_data[0] = 1;
    condition_data[1] = 0;
    condition->set_data_type(DataType::BOOL);
    auto x = MakeFp8Tensor<T>({1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f}, {2, 3}, 0.5f);
    auto y = MakeFp8Tensor<T>({4.0f, 8.0f, 16.0f}, {1, 3}, 0.5f);
    auto where_output = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});
    where_output->set_data_type(dtype);
    where_output->set_quantization({true, 0.5f});
    auto where = feather::KernelDispatcher::instance().create(device, dtype, "Where");
    ASSERT_NE(where, nullptr);
    feather::operators::WhereParam where_param{};
    where_param.condition = condition;
    where_param.x = x;
    where_param.y = y;
    where_param.out = where_output;
    where->SetParam(&where_param);
    ASSERT_EQ(where->compute(), 0) << "device=" << static_cast<int>(device);
    for (const auto& [index, expected] : std::vector<std::pair<int64_t, float>>{{0, 1.0f}, {1, 2.0f}, {2, 4.0f},
                                                                                   {3, 4.0f}, {4, 8.0f}, {5, 16.0f}}) {
        EXPECT_NEAR(ReadFp8Value<T>(*where_output, index), expected, tolerance);
    }
}

template <DataType dtype, typename Scalar>
void VerifyX86Fp8PreparedLinearKernels() {
    constexpr int64_t k = 4096;
    constexpr int64_t n = 64;
    std::vector<float> lhs_values(static_cast<size_t>(k), 1.0f);
    std::vector<float> rhs_values(static_cast<size_t>(k * n));
    for (int64_t index = 0; index < k * n; ++index) {
        rhs_values[static_cast<size_t>(index)] = static_cast<float>((index % 9) - 4) * 0.125f;
    }
    auto lhs = MakeFp8Tensor<Scalar>(lhs_values, {1, k}, 0.25f);
    auto rhs = MakeFp8Tensor<Scalar>(rhs_values, {k, n}, 0.25f);
    rhs->set_immutable(true);

    auto make_output = [&]() {
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, n});
        output->set_data_type(dtype);
        output->set_quantization({true, 0.25f});
        return output;
    };

    auto prepared_output = make_output();
    feather::operators::MatMulParam matmul_param{};
    matmul_param.a = lhs;
    matmul_param.b = rhs;
    matmul_param.out = prepared_output;
    auto prepared_matmul = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "MatMul");
    ASSERT_NE(prepared_matmul, nullptr);
    prepared_matmul->SetParam(&matmul_param);
    ASSERT_EQ(prepared_matmul->Prepare(), 0);
    ASSERT_EQ(prepared_matmul->compute(), 0);

    auto direct_output = make_output();
    feather::operators::MatMulParam direct_param = matmul_param;
    direct_param.out = direct_output;
    auto direct_matmul = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "MatMul");
    ASSERT_NE(direct_matmul, nullptr);
    direct_matmul->SetParam(&direct_param);
    ASSERT_EQ(direct_matmul->compute(), 0);
    for (int64_t index = 0; index < n; ++index) {
        EXPECT_NEAR(ReadFp8Value<Scalar>(*prepared_output, index), ReadFp8Value<Scalar>(*direct_output, index),
                    0.5f);
    }

    auto gemm_output = make_output();
    feather::operators::GemmParam gemm_param{};
    gemm_param.a = lhs;
    gemm_param.b = rhs;
    gemm_param.out = gemm_output;
    auto gemm = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "Gemm");
    ASSERT_NE(gemm, nullptr);
    gemm->SetParam(&gemm_param);
    ASSERT_EQ(gemm->Prepare(), 0);
    ASSERT_EQ(gemm->compute(), 0);

    auto fc_output = make_output();
    feather::operators::FcParam fc_param{};
    fc_param.input = lhs;
    fc_param.w = rhs;
    fc_param.out = fc_output;
    auto fc = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "FC");
    ASSERT_NE(fc, nullptr);
    fc->SetParam(&fc_param);
    ASSERT_EQ(fc->Prepare(), 0);
    ASSERT_EQ(fc->compute(), 0);
    for (int64_t index = 0; index < n; ++index) {
        EXPECT_NEAR(ReadFp8Value<Scalar>(*gemm_output, index), ReadFp8Value<Scalar>(*fc_output, index), 0.5f);
    }
}

TEST(fp8_test, X86Fp8PreparedLinearKernelsMatchDirectPath) {
    VerifyX86Fp8PreparedLinearKernels<DataType::FP8E4M3, Fp8E4M3>();
    VerifyX86Fp8PreparedLinearKernels<DataType::FP8E5M2, Fp8E5M2>();
}

TEST(fp8_test, E4M3RoundTripsFiniteValues) {
    for (const float value : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 3.25f, 448.0f}) {
        const auto encoded = feather::FloatToFp8E4M3(value);
        const float decoded = feather::Fp8E4M3ToFloat(encoded);
        EXPECT_TRUE(std::signbit(value) == std::signbit(decoded) || value == 0.0f);
        EXPECT_NEAR(decoded, value, std::max(0.01f, std::fabs(value) * 0.2f));
    }
}

TEST(fp8_test, E5M2RoundTripsFiniteValuesAndSpecials) {
    for (const float value : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 3.25f, 57344.0f}) {
        const auto encoded = feather::FloatToFp8E5M2(value);
        const float decoded = feather::Fp8E5M2ToFloat(encoded);
        EXPECT_TRUE(std::signbit(value) == std::signbit(decoded) || value == 0.0f);
        EXPECT_NEAR(decoded, value, std::max(0.01f, std::fabs(value) * 0.2f));
    }
    EXPECT_TRUE(std::isinf(feather::Fp8E5M2ToFloat(feather::FloatToFp8E5M2(
        std::numeric_limits<float>::infinity()))));
    EXPECT_TRUE(std::isnan(feather::Fp8E5M2ToFloat(feather::FloatToFp8E5M2(
        std::numeric_limits<float>::quiet_NaN()))));
    EXPECT_TRUE(std::isnan(feather::Fp8E4M3ToFloat(feather::FloatToFp8E4M3(
        std::numeric_limits<float>::quiet_NaN()))));
}

TEST(fp8_test, Fp8EncodingsMatchStandardBitPatterns) {
    EXPECT_EQ(feather::FloatToFp8E4M3(1.0f), 0x38U);
    EXPECT_EQ(feather::FloatToFp8E4M3(0.001953125f), 0x01U);
    EXPECT_EQ(feather::FloatToFp8E4M3(0.015625f), 0x08U);
    EXPECT_EQ(feather::FloatToFp8E4M3(448.0f), 0x7eU);
    EXPECT_EQ(feather::FloatToFp8E4M3(std::numeric_limits<float>::infinity()), 0x7eU);
    EXPECT_EQ(feather::FloatToFp8E4M3(std::numeric_limits<float>::quiet_NaN()), 0x7fU);
    EXPECT_FLOAT_EQ(feather::Fp8E4M3ToFloat(0x01U), 0.001953125f);
    EXPECT_FLOAT_EQ(feather::Fp8E4M3ToFloat(0x08U), 0.015625f);
    EXPECT_FLOAT_EQ(feather::Fp8E4M3ToFloat(0x7eU), 448.0f);

    EXPECT_EQ(feather::FloatToFp8E5M2(1.0f), 0x3cU);
    EXPECT_EQ(feather::FloatToFp8E5M2(0.0000152587890625f), 0x01U);
    EXPECT_EQ(feather::FloatToFp8E5M2(0.00006103515625f), 0x04U);
    EXPECT_EQ(feather::FloatToFp8E5M2(57344.0f), 0x7bU);
    EXPECT_EQ(feather::FloatToFp8E5M2(std::numeric_limits<float>::infinity()), 0x7cU);
    EXPECT_EQ(feather::FloatToFp8E5M2(std::numeric_limits<float>::quiet_NaN()), 0x7dU);
    EXPECT_FLOAT_EQ(feather::Fp8E5M2ToFloat(0x01U), 0.0000152587890625f);
    EXPECT_FLOAT_EQ(feather::Fp8E5M2ToFloat(0x04U), 0.00006103515625f);
    EXPECT_FLOAT_EQ(feather::Fp8E5M2ToFloat(0x7bU), 57344.0f);

    EXPECT_EQ(feather::FloatToFp8E4M3(1.0625f), 0x38U);
    EXPECT_EQ(feather::FloatToFp8E4M3(1.1875f), 0x3aU);
    EXPECT_EQ(feather::FloatToFp8E5M2(1.125f), 0x3cU);
    EXPECT_EQ(feather::FloatToFp8E5M2(1.375f), 0x3eU);
}

TEST(fp8_test, UsesOneByteStorageAndDistinctTypes) {
    EXPECT_EQ(sizeof(Fp8E4M3), 1U);
    EXPECT_EQ(sizeof(Fp8E5M2), 1U);
    EXPECT_EQ(feather::DataTypeBytes(DataType::FP8E4M3), 1U);
    EXPECT_EQ(feather::DataTypeBytes(DataType::FP8E5M2), 1U);
    EXPECT_NE(DataType::FP8E4M3, DataType::FP8E5M2);
    EXPECT_EQ(feather::DataTypeTrait<Fp8E4M3>::type(), DataType::FP8E4M3);
    EXPECT_EQ(feather::DataTypeTrait<Fp8E5M2>::type(), DataType::FP8E5M2);
}

TEST(fp8_test, TensorCarriesScaleThroughSharing) {
    auto source = MakeFp8Tensor<Fp8E4M3>({1.0f, -2.0f}, {2}, 0.25f);
    auto shared = std::make_shared<Tensor>();
    shared->ShareDataWith(*source);
    EXPECT_TRUE(shared->quantization().enabled);
    EXPECT_FLOAT_EQ(shared->quantization_scale(), 0.25f);
    EXPECT_EQ(shared->data_type(), DataType::FP8E4M3);
}

TEST(fp8_test, CommonFp8AddUsesInputAndOutputScales) {
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 0.25f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({3.0f, 4.0f}, {2}, 0.5f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP8E4M3);
    out->set_quantization({true, 0.25f});

    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP8E4M3, "Add");
    ASSERT_NE(kernel, nullptr);
    feather::operators::BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[0].bits) * 0.25f, 4.0f, 0.5f);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[1].bits) * 0.25f, 6.0f, 0.5f);
}

TEST(fp8_test, CommonFp8BinaryRejectsMismatchedOutputShape) {
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2, 1}, 1.0f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({3.0f, 4.0f}, {2, 1}, 1.0f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
    out->set_data_type(DataType::FP8E4M3);

    for (const char* op : {"Add", "Mul"}) {
        auto kernel = feather::KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP8E4M3, op);
        ASSERT_NE(kernel, nullptr);
        feather::operators::BinaryParam param{};
        param.lhs = lhs;
        param.rhs = rhs;
        param.out = out;
        kernel->SetParam(&param);
        EXPECT_EQ(kernel->compute(), -1) << op;
    }
}

TEST(fp8_test, CommonFp8LinearKernelsAreRegistered) {
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        for (const auto& op : {"Cast", "Add", "Mul", "Sub", "Div", "MatMul", "Gemm", "FC"}) {
            auto kernel = feather::KernelDispatcher::instance().create(DeviceType::COMMON, dtype, op);
            EXPECT_NE(kernel, nullptr) << "missing Common FP8 kernel for " << op;
        }
    }
}

TEST(fp8_test, X86Fp8LinearKernelsAreRegistered) {
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        for (const auto& op : {"Cast", "Add", "Mul", "Sub", "Div", "MatMul", "Gemm", "FC"}) {
            auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, op);
            EXPECT_NE(kernel, nullptr) << "missing X86 FP8 kernel for " << op;
            if (kernel != nullptr) {
                EXPECT_EQ(kernel->device(), DeviceType::X86) << "X86 FP8 request fell back for " << op;
            }
        }
    }
}

TEST(fp8_test, X86Fp8AddUsesInputAndOutputScales) {
    auto lhs = MakeFp8Tensor<Fp8E5M2>({1.0f, 2.0f}, {2}, 0.25f);
    auto rhs = MakeFp8Tensor<Fp8E5M2>({3.0f, 4.0f}, {2}, 0.5f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP8E5M2);
    out->set_quantization({true, 0.25f});

    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E5M2, "Add");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::X86);
    feather::operators::BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(feather::Fp8E5M2ToFloat(out->data<Fp8E5M2>()[0].bits) * 0.25f, 4.0f, 0.75f);
    EXPECT_NEAR(feather::Fp8E5M2ToFloat(out->data<Fp8E5M2>()[1].bits) * 0.25f, 6.0f, 0.75f);
}

TEST(fp8_test, X86Fp8BinaryRejectsWrongOutputType) {
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 1.0f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({3.0f, 4.0f}, {2}, 1.0f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP32);

    for (const char* op : {"Add", "Mul"}) {
        auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3, op);
        ASSERT_NE(kernel, nullptr);
        feather::operators::BinaryParam param{};
        param.lhs = lhs;
        param.rhs = rhs;
        param.out = out;
        kernel->SetParam(&param);
        EXPECT_EQ(kernel->compute(), -1) << op;
    }
}

TEST(fp8_test, HostFp8IndexBroadcastAndReduceOpsPreserveScaledValues) {
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        VerifyFp8IndexBroadcastAndReduceOps<Fp8E4M3>(device, 0.25f);
    }
}

TEST(fp8_test, HostFp8ImageKernelsRejectInvalidContracts) {
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, 1.0f);

        auto gap_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 1});
        gap_out->set_data_type(DataType::FP8E4M3);
        auto gap = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "GlobalAveragePool");
        ASSERT_NE(gap, nullptr);
        feather::operators::GlobalAveragePoolParam gap_param{};
        gap_param.input = input;
        gap_param.out = gap_out;
        gap->SetParam(&gap_param);
        EXPECT_EQ(gap->compute(), -1) << "device=" << static_cast<int>(device);

        auto bn_scale = MakeFp8Tensor<Fp8E4M3>({1.0f}, {1}, 1.0f);
        auto bn_bias = MakeFp8Tensor<Fp8E4M3>({0.0f}, {1}, 1.0f);
        auto bn_mean = MakeFp8Tensor<Fp8E4M3>({0.0f}, {1}, 1.0f);
        auto bn_var = MakeFp8Tensor<Fp8E4M3>({1.0f}, {1}, 1.0f);
        auto bn_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
        bn_out->set_data_type(DataType::FP8E4M3);
        bn_out->set_layout(feather::DataLayout::NHWC);
        auto bn = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "BatchNormalization");
        ASSERT_NE(bn, nullptr);
        feather::operators::BatchNormParam bn_param{};
        bn_param.input = input;
        bn_param.scale = bn_scale;
        bn_param.bias = bn_bias;
        bn_param.mean = bn_mean;
        bn_param.var = bn_var;
        bn_param.out = bn_out;
        bn->SetParam(&bn_param);
        EXPECT_EQ(bn->compute(), -1) << "device=" << static_cast<int>(device);

        auto weight = MakeFp8Tensor<Fp8E4M3>({1.0f}, {1, 1, 1, 1}, 1.0f);
        auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3});
        conv_out->set_data_type(DataType::FP8E4M3);
        auto conv = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "Conv2D");
        ASSERT_NE(conv, nullptr);
        feather::operators::Conv2dParam conv_param{};
        conv_param.input = input;
        conv_param.w = weight;
        conv_param.out = conv_out;
        conv_param.stride_h = 1;
        conv_param.stride_w = 1;
        conv_param.dilation_h = 1;
        conv_param.dilation_w = 1;
        conv_param.group = 0;
        conv->SetParam(&conv_param);
        EXPECT_EQ(conv->compute(), -1) << "device=" << static_cast<int>(device);

        auto resize_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
        resize_out->set_data_type(DataType::FP8E4M3);
        auto resize = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "Resize");
        ASSERT_NE(resize, nullptr);
        feather::operators::ResizeParam resize_param{};
        resize_param.input = input;
        resize_param.out = resize_out;
        resize_param.scales = {1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
        resize->SetParam(&resize_param);
        EXPECT_EQ(resize->compute(), -1) << "device=" << static_cast<int>(device);
    }
}

TEST(fp8_test, X86Fp8QwenDepthwiseConvMatchesCommonForBothFormats) {
    constexpr int64_t channels = 19;
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        std::vector<float> input_values(static_cast<size_t>(channels * 4));
        std::vector<float> weight_values(static_cast<size_t>(channels * 4));
        for (int64_t index = 0; index < channels * 4; ++index) {
            input_values[static_cast<size_t>(index)] =
                static_cast<float>((index * 7) % 23 - 11) * 0.125f;
            weight_values[static_cast<size_t>(index)] =
                static_cast<float>((index * 5) % 19 - 9) * 0.0625f;
        }

        std::shared_ptr<Tensor> input;
        std::shared_ptr<Tensor> weight;
        if (dtype == DataType::FP8E4M3) {
            input = MakeFp8Tensor<Fp8E4M3>(input_values, {1, channels, 1, 4}, 0.25f);
            weight = MakeFp8Tensor<Fp8E4M3>(weight_values, {channels, 1, 1, 4}, 0.125f);
        } else {
            input = MakeFp8Tensor<Fp8E5M2>(input_values, {1, channels, 1, 4}, 0.25f);
            weight = MakeFp8Tensor<Fp8E5M2>(weight_values, {channels, 1, 1, 4}, 0.125f);
        }

        auto make_output = [dtype]() {
            auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, channels, 1, 1});
            output->set_data_type(dtype);
            output->set_quantization({true, 0.25f});
            output->set_layout(feather::DataLayout::NCHW);
            return output;
        };

        auto common_output = make_output();
        auto x86_output = make_output();
        feather::operators::Conv2dParam common_param{};
        common_param.input = input;
        common_param.w = weight;
        common_param.out = common_output;
        common_param.stride_h = 1;
        common_param.stride_w = 1;
        common_param.pad_h = 0;
        common_param.pad_w = 0;
        common_param.dilation_h = 1;
        common_param.dilation_w = 1;
        common_param.group = channels;
        feather::operators::Conv2dParam x86_param = common_param;
        x86_param.out = x86_output;

        auto common = feather::KernelDispatcher::instance().create(DeviceType::COMMON, dtype, "Conv2D");
        auto x86 = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "Conv2D");
        ASSERT_NE(common, nullptr);
        ASSERT_NE(x86, nullptr);
        common->SetParam(&common_param);
        x86->SetParam(&x86_param);
        ASSERT_EQ(common->compute(), 0) << "dtype=" << static_cast<int>(dtype);
        ASSERT_EQ(x86->compute(), 0) << "dtype=" << static_cast<int>(dtype);
        ASSERT_EQ(common_output->data_type(), dtype);
        ASSERT_EQ(x86_output->data_type(), dtype);
        for (int64_t index = 0; index < channels; ++index) {
            EXPECT_EQ(static_cast<const uint8_t*>(common_output->raw_data())[index],
                      static_cast<const uint8_t*>(x86_output->raw_data())[index])
                << "dtype=" << static_cast<int>(dtype) << " channel=" << index;
        }
    }
}

TEST(fp8_test, HostFp8ResizeConcatRejectsBatchAndChannelScaling) {
    auto resize_input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, 1.0f);
    auto concat_input = MakeFp8Tensor<Fp8E4M3>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
        {1, 1, 4, 4}, 1.0f);
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        for (const auto& scales : {std::vector<float>{1.01f, 1.0f, 2.0f, 2.0f},
                                   std::vector<float>{1.0f, 1.01f, 2.0f, 2.0f}}) {
            auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 4, 4});
            output->set_data_type(DataType::FP8E4M3);
            output->set_quantization({true, 1.0f});
            auto kernel = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "ResizeConcat");
            ASSERT_NE(kernel, nullptr);
            feather::operators::ResizeConcatParam param{};
            param.resize_input = resize_input;
            param.concat_input = concat_input;
            param.out = output;
            param.scales = scales;
            param.axis = 1;
            kernel->SetParam(&param);
            EXPECT_EQ(kernel->compute(), -1) << "device=" << static_cast<int>(device);
        }
    }
}

TEST(fp8_test, X86Fp8CastToFp32UsesScale) {
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, -2.0f}, {2}, 0.25f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP32);

    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3, "Cast");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::X86);
    feather::operators::CastParam param{};
    param.input = input;
    param.out = out;
    param.to = DataType::FP32;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(out->data<float>()[0], 1.0f, 0.25f);
    EXPECT_NEAR(out->data<float>()[1], -2.0f, 0.25f);
}

TEST(fp8_test, X86Fp8CastHandlesVectorPayloadsForBothFormats) {
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        auto input = std::make_shared<Tensor>(std::vector<int64_t>{32});
        input->set_data_type(dtype);
        input->set_quantization({true, 0.25f});
        auto* raw = static_cast<uint8_t*>(input->raw_data());
        for (int i = 0; i < 32; ++i) {
            const float value = static_cast<float>((i % 9) - 4) * 0.5f;
            raw[i] = dtype == DataType::FP8E4M3 ? feather::FloatToFp8E4M3(value / 0.25f)
                                                : feather::FloatToFp8E5M2(value / 0.25f);
        }
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{32});
        output->set_data_type(DataType::FP32);
        feather::operators::CastParam param{};
        param.input = input;
        param.out = output;
        param.to = DataType::FP32;
        auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "Cast");
        ASSERT_NE(kernel, nullptr);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->compute(), 0);
        for (int i = 0; i < 32; ++i) {
            const float expected = static_cast<float>((i % 9) - 4) * 0.5f;
            EXPECT_NEAR(output->data<float>()[i], expected, 0.25f);
        }
    }
}

TEST(fp8_test, X86Fp8CastToBf16PreservesEveryEncoding) {
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        auto input = std::make_shared<Tensor>(std::vector<int64_t>{256});
        input->set_data_type(dtype);
        input->set_quantization({true, 0.25f});
        auto* raw = static_cast<uint8_t*>(input->raw_data());
        for (int index = 0; index < 256; ++index) {
            raw[index] = static_cast<uint8_t>(index);
        }

        auto output = std::make_shared<Tensor>(std::vector<int64_t>{256});
        output->set_data_type(DataType::BF16);
        feather::operators::CastParam param{};
        param.input = input;
        param.out = output;
        param.to = DataType::BF16;
        auto kernel = feather::KernelDispatcher::instance().create(DeviceType::X86, dtype, "Cast");
        ASSERT_NE(kernel, nullptr);
        kernel->SetParam(&param);
        ASSERT_EQ(kernel->compute(), 0);
        for (int index = 0; index < 256; ++index) {
            const float decoded = dtype == DataType::FP8E4M3
                                      ? feather::Fp8E4M3ToFloat(static_cast<uint8_t>(index))
                                      : feather::Fp8E5M2ToFloat(static_cast<uint8_t>(index));
            EXPECT_EQ(output->data<feather::BFloat16>()[index].bits,
                      feather::FloatToBFloat16(decoded * 0.25f))
                << "dtype=" << static_cast<int>(dtype) << " code=" << index;
        }
    }
}

TEST(fp8_test, CommonFp8CastRejectsMismatchedInputAndOutputContracts) {
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, -2.0f}, {2}, 0.25f);
    auto wrong_shape = std::make_shared<Tensor>(std::vector<int64_t>{3});
    wrong_shape->set_data_type(DataType::FP32);
    feather::operators::CastParam shape_param{};
    shape_param.input = input;
    shape_param.out = wrong_shape;
    shape_param.to = DataType::FP32;
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP8E4M3, "Cast");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&shape_param);
    EXPECT_EQ(kernel->compute(), -1);

    auto wrong_input = std::make_shared<Tensor>();
    wrong_input->Assign<float>({1.0f, 2.0f}, {2});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    output->set_data_type(DataType::FP32);
    feather::operators::CastParam dtype_param{};
    dtype_param.input = wrong_input;
    dtype_param.out = output;
    dtype_param.to = DataType::FP32;
    kernel->SetParam(&dtype_param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, HostFp8CastRequiresMatchingShapeAndPerTensorScale) {
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, -2.0f}, {2, 1}, 1.0f);
        auto wrong_shape = std::make_shared<Tensor>(std::vector<int64_t>{1, 2});
        wrong_shape->set_data_type(DataType::FP32);
        auto fp8_kernel = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "Cast");
        ASSERT_NE(fp8_kernel, nullptr);
        feather::operators::CastParam shape_param{};
        shape_param.input = input;
        shape_param.out = wrong_shape;
        shape_param.to = DataType::FP32;
        fp8_kernel->SetParam(&shape_param);
        EXPECT_EQ(fp8_kernel->compute(), -1) << "device=" << static_cast<int>(device);

        input->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
        auto matching_output = std::make_shared<Tensor>(std::vector<int64_t>{2, 1});
        matching_output->set_data_type(DataType::FP32);
        feather::operators::CastParam input_scale_param{};
        input_scale_param.input = input;
        input_scale_param.out = matching_output;
        input_scale_param.to = DataType::FP32;
        fp8_kernel->SetParam(&input_scale_param);
        EXPECT_EQ(fp8_kernel->compute(), -1) << "device=" << static_cast<int>(device);

        auto fp32_input = std::make_shared<Tensor>();
        fp32_input->Assign<float>({1.0f, -2.0f}, {2});
        auto fp8_output = std::make_shared<Tensor>(std::vector<int64_t>{2});
        fp8_output->set_data_type(DataType::FP8E4M3);
        fp8_output->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
        auto fp32_kernel = feather::KernelDispatcher::instance().create(device, DataType::FP32, "Cast");
        ASSERT_NE(fp32_kernel, nullptr);
        feather::operators::CastParam output_scale_param{};
        output_scale_param.input = fp32_input;
        output_scale_param.out = fp8_output;
        output_scale_param.to = DataType::FP8E4M3;
        fp32_kernel->SetParam(&output_scale_param);
        EXPECT_EQ(fp32_kernel->compute(), -1) << "device=" << static_cast<int>(device);
    }
}

TEST(fp8_test, HostFp8ArithmeticRejectsNonPerTensorScale) {
    auto lhs = MakeFp8Tensor<Fp8E5M2>({1.0f, 2.0f}, {2}, 1.0f);
    lhs->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
    auto rhs = MakeFp8Tensor<Fp8E5M2>({3.0f, 4.0f}, {2}, 1.0f);
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{2});
        output->set_data_type(DataType::FP8E5M2);
        output->set_quantization({true, 1.0f});
        auto kernel = feather::KernelDispatcher::instance().create(device, DataType::FP8E5M2, "Add");
        ASSERT_NE(kernel, nullptr);
        feather::operators::BinaryParam param{};
        param.lhs = lhs;
        param.rhs = rhs;
        param.out = output;
        kernel->SetParam(&param);
        EXPECT_EQ(kernel->compute(), -1) << "device=" << static_cast<int>(device);
    }
}

TEST(fp8_test, HostFp8PowRejectsNonPerTensorExponentScale) {
    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        auto input = MakeFp8Tensor<Fp8E4M3>({2.0f}, {1}, 1.0f);
        auto exponent = MakeFp8Tensor<Fp8E4M3>({2.0f}, {1}, 1.0f);
        exponent->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{1});
        output->set_data_type(DataType::FP8E4M3);
        output->set_quantization({true, 1.0f});
        auto kernel = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "Pow");
        ASSERT_NE(kernel, nullptr);

        feather::operators::PowParam param{};
        param.input = input;
        param.exponent_tensor = exponent;
        param.out = output;
        kernel->SetParam(&param);
        EXPECT_EQ(kernel->compute(), -1) << "device=" << static_cast<int>(device);
    }
}

TEST(fp8_test, HostFp8FcRejectsMalformedOutputAndBiasContracts) {
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
    auto weight = MakeFp8Tensor<Fp8E4M3>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, 1.0f);
    auto wrong_output = std::make_shared<Tensor>(std::vector<int64_t>{1});
    wrong_output->set_data_type(DataType::FP8E4M3);
    wrong_output->set_quantization({true, 1.0f});
    auto wrong_bias = std::make_shared<Tensor>();
    wrong_bias->Assign<float>({1.0f, 2.0f}, {2});

    for (const auto device : {DeviceType::COMMON, DeviceType::X86}) {
        auto kernel = feather::KernelDispatcher::instance().create(device, DataType::FP8E4M3, "FC");
        ASSERT_NE(kernel, nullptr);
        feather::operators::FcParam shape_param{};
        shape_param.input = input;
        shape_param.w = weight;
        shape_param.out = wrong_output;
        kernel->SetParam(&shape_param);
        EXPECT_EQ(kernel->compute(), -1) << "device=" << static_cast<int>(device);

        auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
        output->set_data_type(DataType::FP8E4M3);
        output->set_quantization({true, 1.0f});
        feather::operators::FcParam bias_param{};
        bias_param.input = input;
        bias_param.w = weight;
        bias_param.bias = wrong_bias;
        bias_param.out = output;
        kernel->SetParam(&bias_param);
        EXPECT_EQ(kernel->compute(), -1) << "device=" << static_cast<int>(device);
    }
}

TEST(fp8_test, X86Fp8MatMulGemmAndFcUseFp32Accumulation) {
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 0.25f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, 0.5f);
    auto expected = std::vector<float>{19.0f, 22.0f, 43.0f, 50.0f};

    auto matmul_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    matmul_out->set_data_type(DataType::FP8E4M3);
    matmul_out->set_quantization({true, 0.25f});
    feather::operators::MatMulParam matmul_param{};
    matmul_param.a = lhs;
    matmul_param.b = rhs;
    matmul_param.out = matmul_out;
    auto matmul = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3, "MatMul");
    ASSERT_NE(matmul, nullptr);
    ASSERT_EQ(matmul->device(), DeviceType::X86);
    matmul->SetParam(&matmul_param);
    ASSERT_EQ(matmul->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E4M3ToFloat(matmul_out->data<Fp8E4M3>()[i].bits) * 0.25f, expected[i], 2.0f);
    }

    auto gemm_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    gemm_out->set_data_type(DataType::FP8E4M3);
    gemm_out->set_quantization({true, 0.25f});
    feather::operators::GemmParam gemm_param{};
    gemm_param.a = lhs;
    gemm_param.b = rhs;
    gemm_param.out = gemm_out;
    auto gemm = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3, "Gemm");
    ASSERT_NE(gemm, nullptr);
    ASSERT_EQ(gemm->device(), DeviceType::X86);
    gemm->SetParam(&gemm_param);
    ASSERT_EQ(gemm->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E4M3ToFloat(gemm_out->data<Fp8E4M3>()[i].bits) * 0.25f, expected[i], 2.0f);
    }

    auto fc_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    fc_out->set_data_type(DataType::FP8E4M3);
    fc_out->set_quantization({true, 0.25f});
    feather::operators::FcParam fc_param{};
    fc_param.input = lhs;
    fc_param.w = rhs;
    fc_param.out = fc_out;
    auto fc = feather::KernelDispatcher::instance().create(DeviceType::X86, DataType::FP8E4M3, "FC");
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(fc->device(), DeviceType::X86);
    fc->SetParam(&fc_param);
    ASSERT_EQ(fc->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E4M3ToFloat(fc_out->data<Fp8E4M3>()[i].bits) * 0.25f, expected[i], 2.0f);
    }
}

#ifdef FEATHER_WITH_CUDA
namespace {
bool HasCudaFp8TestDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}  // namespace

TEST(fp8_test, CudaFp8KernelsAreRegisteredWithoutFallback) {
    for (const auto dtype : {DataType::FP8E4M3, DataType::FP8E5M2}) {
        for (const auto& op : {"Cast", "Add", "Mul", "Sub", "Div", "MatMul", "Gemm", "FC"}) {
            auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, dtype, op);
            ASSERT_NE(kernel, nullptr) << "missing CUDA FP8 kernel for " << op;
            EXPECT_EQ(kernel->device(), DeviceType::CUDA) << "CUDA FP8 request fell back for " << op;
        }
    }
}

TEST(fp8_test, CudaFp8CastRejectsOutputContractMismatch) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, -2.0f}, {2}, 1.0f);
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Cast");
    ASSERT_NE(kernel, nullptr);

    auto wrong_shape = std::make_shared<Tensor>(std::vector<int64_t>{3});
    wrong_shape->set_data_type(DataType::FP32);
    feather::operators::CastParam shape_param{};
    shape_param.input = input;
    shape_param.out = wrong_shape;
    shape_param.to = DataType::FP32;
    kernel->SetParam(&shape_param);
    EXPECT_EQ(kernel->compute(), -1);

    auto wrong_type = std::make_shared<Tensor>(std::vector<int64_t>{2});
    wrong_type->set_data_type(DataType::FP32);
    feather::operators::CastParam dtype_param{};
    dtype_param.input = input;
    dtype_param.out = wrong_type;
    dtype_param.to = DataType::FP8E4M3;
    kernel->SetParam(&dtype_param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8RejectsNonPerTensorScale) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 1.0f);
    lhs->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
    auto rhs = MakeFp8Tensor<Fp8E4M3>({3.0f, 4.0f}, {2}, 1.0f);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    output->set_data_type(DataType::FP8E4M3);
    output->set_quantization({true, 1.0f});
    auto add = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Add");
    ASSERT_NE(add, nullptr);
    feather::operators::BinaryParam add_param{};
    add_param.lhs = lhs;
    add_param.rhs = rhs;
    add_param.out = output;
    add->SetParam(&add_param);
    EXPECT_EQ(add->compute(), -1);

    auto fp32_input = std::make_shared<Tensor>();
    fp32_input->Assign<float>({1.0f, -2.0f}, {2});
    auto fp8_output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    fp8_output->set_data_type(DataType::FP8E4M3);
    fp8_output->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
    auto cast = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP32, "Cast");
    ASSERT_NE(cast, nullptr);
    feather::operators::CastParam cast_param{};
    cast_param.input = fp32_input;
    cast_param.out = fp8_output;
    cast_param.to = DataType::FP8E4M3;
    cast->SetParam(&cast_param);
    EXPECT_EQ(cast->compute(), -1);
}

TEST(fp8_test, CudaFp8PowRejectsNonPerTensorExponentScale) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E5M2>({2.0f}, {1}, 1.0f);
    auto exponent = MakeFp8Tensor<Fp8E5M2>({2.0f}, {1}, 1.0f);
    exponent->set_quantization({true, 1.0f, feather::QuantizationGranularity::kPerChannel, 0, 0});
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1});
    output->set_data_type(DataType::FP8E5M2);
    output->set_quantization({true, 1.0f});
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "Pow");
    ASSERT_NE(kernel, nullptr);

    feather::operators::PowParam param{};
    param.input = input;
    param.exponent_tensor = exponent;
    param.out = output;
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8GemmRejectsNonFiniteCoefficients) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, 1.0f);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    output->set_data_type(DataType::FP8E4M3);
    output->set_quantization({true, 1.0f});
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Gemm");
    ASSERT_NE(kernel, nullptr);
    feather::operators::GemmParam param{};
    param.a = lhs;
    param.b = rhs;
    param.out = output;
    param.alpha = std::numeric_limits<float>::quiet_NaN();
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8ReduceRejectsAxisValuesThatOverflowKernelRank) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    output->set_data_type(DataType::FP8E4M3);
    output->set_quantization({true, 1.0f});
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "ReduceSum");
    ASSERT_NE(kernel, nullptr);

    feather::operators::ReduceSumParam param{};
    param.input = input;
    param.out = output;
    param.axes = {static_cast<int64_t>(1ULL << 32)};
    param.keepdims = false;
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaLaunchDivUpHandlesMaximumInt64WithoutOverflow) {
    EXPECT_EQ(feather::kernel::cuda_detail::DivUp(0, 256), 0);
    EXPECT_EQ(feather::kernel::cuda_detail::DivUp(std::numeric_limits<int64_t>::max(), 256),
              std::numeric_limits<int64_t>::max() / 256 + 1);
}

TEST(fp8_test, CudaFp8AddUsesInputAndOutputScales) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 0.25f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({3.0f, 4.0f}, {2}, 0.5f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP8E4M3);
    out->set_quantization({true, 0.25f});

    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Add");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::CUDA);
    feather::operators::BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[0].bits) * 0.25f, 4.0f, 0.75f);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[1].bits) * 0.25f, 6.0f, 0.75f);
}

TEST(fp8_test, CudaFp8IndexBroadcastAndReduceOpsPreserveScaledValues) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    VerifyFp8IndexBroadcastAndReduceOps<Fp8E5M2>(DeviceType::CUDA, 0.5f);
}

TEST(fp8_test, CudaFp8AvgPoolPreservesNhwcLayout) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
        {1, 2, 4, 2}, 1.0f);
    input->set_layout(feather::DataLayout::NHWC);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2});
    out->set_data_type(DataType::FP8E4M3);
    out->set_quantization({true, 1.0f});
    out->set_layout(feather::DataLayout::NHWC);

    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "AvgPool");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->device(), DeviceType::CUDA);
    feather::operators::PoolParam param{};
    param.input = input;
    param.out = out;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_h = 0;
    param.pad_w = 0;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected{6.0f, 7.0f, 10.0f, 11.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[i].bits), expected[i], 0.25f);
    }
}

TEST(fp8_test, CudaFp8ImageKernelsRejectInvalidContracts) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, 1.0f);

    auto conv_weight = MakeFp8Tensor<Fp8E4M3>({1.0f}, {1, 1, 1, 1}, 1.0f);
    auto conv_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3});
    conv_out->set_data_type(DataType::FP8E4M3);
    auto conv = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Conv2D");
    ASSERT_NE(conv, nullptr);
    feather::operators::Conv2dParam conv_param{};
    conv_param.input = input;
    conv_param.w = conv_weight;
    conv_param.out = conv_out;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.group = 0;
    conv->SetParam(&conv_param);
    EXPECT_EQ(conv->compute(), -1);

    auto resize_out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 2});
    resize_out->set_data_type(DataType::FP8E4M3);
    auto resize = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Resize");
    ASSERT_NE(resize, nullptr);
    feather::operators::ResizeParam resize_param{};
    resize_param.input = input;
    resize_param.out = resize_out;
    resize_param.scales = {1.0f, 1.0f, 1.0f, 1.0f};
    resize->SetParam(&resize_param);
    EXPECT_EQ(resize->compute(), -1);
}

TEST(fp8_test, CudaFp8UnaryRejectsInvalidOutputContract) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 1.0f);
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "ReLU");
    ASSERT_NE(kernel, nullptr);

    auto wrong_type = std::make_shared<Tensor>(std::vector<int64_t>{2});
    wrong_type->set_data_type(DataType::FP32);
    feather::operators::UnaryParam type_param{};
    type_param.input = input;
    type_param.out = wrong_type;
    kernel->SetParam(&type_param);
    EXPECT_EQ(kernel->compute(), -1);

    auto wrong_shape = std::make_shared<Tensor>(std::vector<int64_t>{3});
    wrong_shape->set_data_type(DataType::FP8E4M3);
    feather::operators::UnaryParam shape_param{};
    shape_param.input = input;
    shape_param.out = wrong_shape;
    kernel->SetParam(&shape_param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8ElementwiseRejectsInvalidInputAndOutputContracts) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 1.0f);
    auto rhs_wrong_type = MakeFp8Tensor<Fp8E5M2>({3.0f, 4.0f}, {2}, 1.0f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    out->set_data_type(DataType::FP8E4M3);
    auto add = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Add");
    ASSERT_NE(add, nullptr);
    feather::operators::BinaryParam type_param{};
    type_param.lhs = lhs;
    type_param.rhs = rhs_wrong_type;
    type_param.out = out;
    add->SetParam(&type_param);
    EXPECT_EQ(add->compute(), -1);

    auto wrong_output = std::make_shared<Tensor>(std::vector<int64_t>{2});
    wrong_output->set_data_type(DataType::FP32);
    auto relu = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "ReLU");
    ASSERT_NE(relu, nullptr);
    feather::operators::UnaryParam output_param{};
    output_param.input = lhs;
    output_param.out = wrong_output;
    relu->SetParam(&output_param);
    EXPECT_EQ(relu->compute(), -1);
}

TEST(fp8_test, CudaFp8SliceRejectsNonAxisShapeMismatch) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, 1.0f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1});
    out->set_data_type(DataType::FP8E4M3);
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Slice");
    ASSERT_NE(kernel, nullptr);
    feather::operators::SliceParam param{};
    param.input = input;
    param.out = out;
    param.axis = 0;
    param.start = 0;
    param.end = 1;
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8GatherRejectsOutOfRangeIndex) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto data = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
    auto indices = std::make_shared<Tensor>();
    indices->Assign<int32_t>({0, 2}, {2});
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->set_data_type(DataType::FP8E4M3);
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Gather");
    ASSERT_NE(kernel, nullptr);
    feather::operators::GatherParam param{};
    param.data = data;
    param.indices = indices;
    param.out = out;
    param.axis = 0;
    kernel->SetParam(&param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8ReduceAndSoftmaxRejectWrongOutputType) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E5M2>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);

    auto reduce_out = std::make_shared<Tensor>(std::vector<int64_t>{2});
    reduce_out->set_data_type(DataType::FP32);
    auto reduce = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "ReduceSum");
    ASSERT_NE(reduce, nullptr);
    feather::operators::ReduceSumParam reduce_param{};
    reduce_param.input = input;
    reduce_param.out = reduce_out;
    reduce_param.axes = {1};
    reduce_param.keepdims = false;
    reduce->SetParam(&reduce_param);
    EXPECT_EQ(reduce->compute(), -1);

    auto softmax_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    softmax_out->set_data_type(DataType::FP32);
    auto softmax = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "Softmax");
    ASSERT_NE(softmax, nullptr);
    feather::operators::SoftmaxParam softmax_param{};
    softmax_param.input = input;
    softmax_param.out = softmax_out;
    softmax_param.axis = 1;
    softmax->SetParam(&softmax_param);
    EXPECT_EQ(softmax->compute(), -1);
}

TEST(fp8_test, CudaFp8LinearAcceptsUnspecifiedOutputType) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
    auto rhs = MakeFp8Tensor<Fp8E4M3>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, 1.0f);

    auto matmul_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    feather::operators::MatMulParam matmul_param{};
    matmul_param.a = lhs;
    matmul_param.b = rhs;
    matmul_param.out = matmul_out;
    auto matmul = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "MatMul");
    ASSERT_NE(matmul, nullptr);
    matmul->SetParam(&matmul_param);
    EXPECT_EQ(matmul->compute(), 0);

    auto gemm_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    feather::operators::GemmParam gemm_param{};
    gemm_param.a = lhs;
    gemm_param.b = rhs;
    gemm_param.out = gemm_out;
    auto gemm = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "Gemm");
    ASSERT_NE(gemm, nullptr);
    gemm->SetParam(&gemm_param);
    EXPECT_EQ(gemm->compute(), 0);

    auto fc_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    feather::operators::FcParam fc_param{};
    fc_param.input = lhs;
    fc_param.w = rhs;
    fc_param.out = fc_out;
    auto fc = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3, "FC");
    ASSERT_NE(fc, nullptr);
    fc->SetParam(&fc_param);
    EXPECT_EQ(fc->compute(), 0);
}

TEST(fp8_test, CudaFp8BatchNormUsesPerTensorParameterScales) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({2.0f, 4.0f}, {1, 2, 1, 1}, 0.5f);
    auto scale = MakeFp8Tensor<Fp8E4M3>({2.0f, 3.0f}, {2}, 0.5f);
    auto bias = MakeFp8Tensor<Fp8E4M3>({1.0f, 2.0f}, {2}, 0.25f);
    auto mean = MakeFp8Tensor<Fp8E4M3>({0.0f, 1.0f}, {2}, 0.125f);
    auto variance = MakeFp8Tensor<Fp8E4M3>({1.0f, 4.0f}, {2}, 0.25f);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 1, 1});
    out->set_data_type(DataType::FP8E4M3);
    out->set_quantization({true, 1.0f});
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3,
                                                                 "BatchNormalization");
    ASSERT_NE(kernel, nullptr);
    feather::operators::BatchNormParam param{};
    param.input = input;
    param.scale = scale;
    param.bias = bias;
    param.mean = mean;
    param.var = variance;
    param.out = out;
    param.epsilon = 0.0f;
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[0].bits), 5.0f, 0.5f);
    EXPECT_NEAR(feather::Fp8E4M3ToFloat(out->data<Fp8E4M3>()[1].bits), 6.5f, 0.75f);
}

TEST(fp8_test, CudaFp8BatchNormRejectsInvalidDecodedParameters) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto input = MakeFp8Tensor<Fp8E4M3>({2.0f, 4.0f}, {1, 2, 1, 1}, 1.0f);
    auto bias = MakeFp8Tensor<Fp8E4M3>({0.0f, 0.0f}, {2}, 1.0f);
    auto mean = MakeFp8Tensor<Fp8E4M3>({0.0f, 0.0f}, {2}, 1.0f);
    auto variance = MakeFp8Tensor<Fp8E4M3>({1.0f, 1.0f}, {2}, 1.0f);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 1, 1});
    output->set_data_type(DataType::FP8E4M3);
    output->set_quantization({true, 1.0f});
    auto kernel = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E4M3,
                                                                 "BatchNormalization");
    ASSERT_NE(kernel, nullptr);

    auto non_finite_scale = MakeFp8Tensor<Fp8E4M3>({std::numeric_limits<float>::quiet_NaN(), 1.0f}, {2}, 1.0f);
    feather::operators::BatchNormParam non_finite_param{};
    non_finite_param.input = input;
    non_finite_param.scale = non_finite_scale;
    non_finite_param.bias = bias;
    non_finite_param.mean = mean;
    non_finite_param.var = variance;
    non_finite_param.out = output;
    non_finite_param.epsilon = 0.0f;
    kernel->SetParam(&non_finite_param);
    EXPECT_EQ(kernel->compute(), -1);

    auto invalid_variance = MakeFp8Tensor<Fp8E4M3>({-1.0f, 1.0f}, {2}, 1.0f);
    auto valid_scale = MakeFp8Tensor<Fp8E4M3>({1.0f, 1.0f}, {2}, 1.0f);
    feather::operators::BatchNormParam variance_param = non_finite_param;
    variance_param.scale = valid_scale;
    variance_param.var = invalid_variance;
    kernel->SetParam(&variance_param);
    EXPECT_EQ(kernel->compute(), -1);
}

TEST(fp8_test, CudaFp8MatMulGemmAndFcUseFp32Accumulation) {
    if (!HasCudaFp8TestDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    auto lhs = MakeFp8Tensor<Fp8E5M2>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 0.25f);
    auto rhs = MakeFp8Tensor<Fp8E5M2>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, 0.5f);
    const auto expected = std::vector<float>{19.0f, 22.0f, 43.0f, 50.0f};

    auto matmul_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    matmul_out->set_data_type(DataType::FP8E5M2);
    matmul_out->set_quantization({true, 0.25f});
    feather::operators::MatMulParam matmul_param{};
    matmul_param.a = lhs;
    matmul_param.b = rhs;
    matmul_param.out = matmul_out;
    auto matmul = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "MatMul");
    ASSERT_NE(matmul, nullptr);
    ASSERT_EQ(matmul->device(), DeviceType::CUDA);
    matmul->SetParam(&matmul_param);
    ASSERT_EQ(matmul->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E5M2ToFloat(matmul_out->data<Fp8E5M2>()[i].bits) * 0.25f, expected[i], 3.0f);
    }

    auto gemm_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    gemm_out->set_data_type(DataType::FP8E5M2);
    gemm_out->set_quantization({true, 0.25f});
    feather::operators::GemmParam gemm_param{};
    gemm_param.a = lhs;
    gemm_param.b = rhs;
    gemm_param.out = gemm_out;
    auto gemm = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "Gemm");
    ASSERT_NE(gemm, nullptr);
    ASSERT_EQ(gemm->device(), DeviceType::CUDA);
    gemm->SetParam(&gemm_param);
    ASSERT_EQ(gemm->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E5M2ToFloat(gemm_out->data<Fp8E5M2>()[i].bits) * 0.25f, expected[i], 3.0f);
    }

    auto fc_out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    fc_out->set_data_type(DataType::FP8E5M2);
    fc_out->set_quantization({true, 0.25f});
    feather::operators::FcParam fc_param{};
    fc_param.input = lhs;
    fc_param.w = rhs;
    fc_param.out = fc_out;
    auto fc = feather::KernelDispatcher::instance().create(DeviceType::CUDA, DataType::FP8E5M2, "FC");
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(fc->device(), DeviceType::CUDA);
    fc->SetParam(&fc_param);
    ASSERT_EQ(fc->compute(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::Fp8E5M2ToFloat(fc_out->data<Fp8E5M2>()[i].bits) * 0.25f, expected[i], 3.0f);
    }
}
#endif

}  // namespace
