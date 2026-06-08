#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#ifdef FEATHER_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/kernel.h"
#include "src/kernel/add.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/gemm.h"
#include "src/kernel/matmul.h"
#include "src/kernel/resize.h"
#include "src/kernel/silu.h"
#include "src/kernel/slice.h"
#include "src/kernel/split.h"
#include "src/kernel/transpose.h"
#include "src/kernel/yolo_decode.h"
#include "src/operator/params.h"
#include "util/fp16.h"

namespace {

#ifdef FEATHER_WITH_CUDA
bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

}  // namespace

TEST(cuda_kernel_test, AddRunsOnCudaFP32WithBroadcast) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<float>({10.0f, 20.0f, 30.0f}, {1, 3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 3});

    feather::operators::BinaryParam param;
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;

    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Add");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::AddKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, Conv2DRunsOnCudaFP16) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
         feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f)},
        {1, 1, 3, 3});
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(0.0f),
         feather::FloatToHalf(-1.0f)},
        {1, 1, 2, 2});
    auto bias = std::make_shared<feather::Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f)}, {1});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 1, 2, 2});
    out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = out;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::Conv2DKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {-3.5f, -3.5f, -3.5f, -3.5f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, Conv2DPointwiseRunsOnCudaFP32) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 2, 2});
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<float>({1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f}, {3, 2, 1, 1});
    auto bias = std::make_shared<feather::Tensor>();
    bias->Assign<float>({0.5f, -1.0f, 0.0f}, {3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 3, 2, 2});

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.bias = bias;
    param.out = out;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.5f, 2.5f, 3.5f, 4.5f, 9.0f, 19.0f,
                                         29.0f, 39.0f, 32.0f, 64.0f, 96.0f, 128.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, Conv2DDepthwiseRunsOnCudaFP16) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f),  feather::FloatToHalf(2.0f),  feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f),  feather::FloatToHalf(5.0f),  feather::FloatToHalf(6.0f),
         feather::FloatToHalf(7.0f),  feather::FloatToHalf(8.0f),  feather::FloatToHalf(9.0f),
         feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f), feather::FloatToHalf(30.0f),
         feather::FloatToHalf(40.0f), feather::FloatToHalf(50.0f), feather::FloatToHalf(60.0f),
         feather::FloatToHalf(70.0f), feather::FloatToHalf(80.0f), feather::FloatToHalf(90.0f)},
        {1, 2, 3, 3});
    auto weight = std::make_shared<feather::Tensor>();
    weight->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(0.0f),
         feather::FloatToHalf(-1.0f), feather::FloatToHalf(0.0f), feather::FloatToHalf(1.0f),
         feather::FloatToHalf(1.0f), feather::FloatToHalf(0.0f)},
        {2, 1, 2, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 2, 2});
    out->mutable_data<uint16_t>();

    feather::operators::Conv2dParam param{};
    param.input = input;
    param.w = weight;
    param.out = out;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.group = 2;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Conv2D");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {-4.0f, -4.0f, -4.0f, -4.0f, 60.0f, 80.0f, 120.0f, 140.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, MatMulRunsOnCudaFP32) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto a = std::make_shared<feather::Tensor>();
    a->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = std::make_shared<feather::Tensor>();
    b->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                     {3, 4});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 4});

    feather::operators::MatMulParam param{};
    param.a = a;
    param.b = b;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "MatMul");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {38.0f, 44.0f, 50.0f, 56.0f, 83.0f, 98.0f, 113.0f, 128.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, GemmRunsOnCudaFP16WithVectorBias) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto a = std::make_shared<feather::Tensor>();
    a->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
        {2, 3});
    auto b = std::make_shared<feather::Tensor>();
    b->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
         feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f),
         feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f)},
        {3, 4});
    auto bias = std::make_shared<feather::Tensor>();
    bias->Assign<uint16_t>({feather::FloatToHalf(0.5f), feather::FloatToHalf(-1.0f),
                            feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f)},
                           {4});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 4});
    out->mutable_data<uint16_t>();

    feather::operators::GemmParam param{};
    param.a = a;
    param.b = b;
    param.bias = bias;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Gemm");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {38.5f, 43.0f, 52.0f, 59.0f, 83.5f, 97.0f, 115.0f, 131.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, ConcatRunsOnCudaFP32AlongMiddleAxis) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                       {2, 2, 3});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<float>({101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f}, {2, 1, 3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 3, 3});

    feather::operators::ConcatParam param;
    param.inputs = {lhs, rhs};
    param.out = out;
    param.axis = 1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Concat");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::ConcatKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,
                                         101.0f, 102.0f, 103.0f, 7.0f,   8.0f,   9.0f,
                                         10.0f,  11.0f,  12.0f,  104.0f, 105.0f, 106.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, ConcatRunsOnCudaFP16AlongNegativeAxis) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f)},
        {1, 2, 2});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<uint16_t>(
        {feather::FloatToHalf(10.0f), feather::FloatToHalf(20.0f), feather::FloatToHalf(30.0f),
         feather::FloatToHalf(40.0f), feather::FloatToHalf(50.0f), feather::FloatToHalf(60.0f)},
        {1, 2, 3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 5});
    out->mutable_data<uint16_t>();

    feather::operators::ConcatParam param;
    param.inputs = {lhs, rhs};
    param.out = out;
    param.axis = -1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Concat");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::ConcatKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.0f, 2.0f, 10.0f, 20.0f, 30.0f,
                                         3.0f, 4.0f, 40.0f, 50.0f, 60.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, SplitRunsOnCudaFP32AlongMiddleAxis) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>(
        {1.0f,   2.0f,   3.0f,   101.0f, 102.0f, 103.0f, 201.0f, 202.0f, 203.0f,
         4.0f,   5.0f,   6.0f,   104.0f, 105.0f, 106.0f, 204.0f, 205.0f, 206.0f},
        {2, 3, 3});
    auto out0 = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 1, 3});
    auto out1 = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 2, 3});

    feather::operators::SplitParam param;
    param.input = input;
    param.outputs = {out0, out1};
    param.axis = 1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Split");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::SplitKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> expected1 = {101.0f, 102.0f, 103.0f, 201.0f, 202.0f, 203.0f,
                                          104.0f, 105.0f, 106.0f, 204.0f, 205.0f, 206.0f};
    for (size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_FLOAT_EQ(out0->data<float>()[i], expected0[i]);
    }
    for (size_t i = 0; i < expected1.size(); ++i) {
        EXPECT_FLOAT_EQ(out1->data<float>()[i], expected1[i]);
    }
#endif
}

TEST(cuda_kernel_test, SplitRunsOnCudaFP16AlongNegativeAxis) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(10.0f),
         feather::FloatToHalf(20.0f), feather::FloatToHalf(30.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f), feather::FloatToHalf(40.0f), feather::FloatToHalf(50.0f),
         feather::FloatToHalf(60.0f)},
        {1, 2, 5});
    auto out0 = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 2});
    auto out1 = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 3});
    out0->mutable_data<uint16_t>();
    out1->mutable_data<uint16_t>();

    feather::operators::SplitParam param;
    param.input = input;
    param.outputs = {out0, out1};
    param.axis = -1;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Split");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::SplitKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected0 = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> expected1 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    for (size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out0->data<uint16_t>()[i]), expected0[i], 1e-3f);
    }
    for (size_t i = 0; i < expected1.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out1->data<uint16_t>()[i]), expected1[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, TransposeRunsOnCudaFP32WithThreeDimPerm) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 2, 3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{3, 1, 2});

    feather::operators::TransposeParam param;
    param.input = input;
    param.out = out;
    param.perm = {2, 0, 1};

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Transpose");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::TransposeKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, SliceRunsOnCudaFP16AlongMiddleAxis) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>(
        {feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
         feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f),
         feather::FloatToHalf(7.0f), feather::FloatToHalf(8.0f), feather::FloatToHalf(9.0f),
         feather::FloatToHalf(10.0f), feather::FloatToHalf(11.0f), feather::FloatToHalf(12.0f)},
        {2, 2, 3});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 1, 3});
    out->mutable_data<uint16_t>();

    feather::operators::SliceParam param;
    param.input = input;
    param.out = out;
    param.axis = 1;
    param.start = 1;
    param.end = 2;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "Slice");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::SliceKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, ResizeRunsOnCudaFP32NearestNeighborNchw) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 1, 4, 4});

    feather::operators::ResizeParam param;
    param.input = input;
    param.out = out;
    param.scales = {1.0f, 1.0f, 2.0f, 2.0f};

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "Resize");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::ResizeKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                         3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
#endif
}

TEST(cuda_kernel_test, SiluRunsOnCudaFP32) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({-2.0f, -1.0f, 0.0f, 2.0f}, {4});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{4});

    feather::operators::UnaryParam param;
    param.input = input;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "SiLU");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::SiluKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t i = 0; i < input->numel(); ++i) {
        const float expected = input->data<float>()[i] / (1.0f + std::exp(-input->data<float>()[i]));
        EXPECT_NEAR(out->data<float>()[i], expected, 1e-6f);
    }
#endif
}

TEST(cuda_kernel_test, SiluRunsOnCudaFP16) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto input = std::make_shared<feather::Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(-2.0f), feather::FloatToHalf(-1.0f),
                             feather::FloatToHalf(0.0f), feather::FloatToHalf(2.0f)},
                            {4});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{4});
    out->mutable_data<uint16_t>();

    feather::operators::UnaryParam param;
    param.input = input;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "SiLU");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::SiluKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    for (int64_t i = 0; i < input->numel(); ++i) {
        const float value = feather::HalfToFloat(input->data<uint16_t>()[i]);
        const float expected = value / (1.0f + std::exp(-value));
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected, 1e-3f);
    }
#endif
}

TEST(cuda_kernel_test, YoloDecodeRunsOnCudaFP32) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto raw = std::make_shared<feather::Tensor>();
    raw->Assign<float>({0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
                        2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f},
                       {1, 12, 1, 1});
    auto xy_scale = std::make_shared<feather::Tensor>();
    xy_scale->Assign<float>({2.0f}, {1});
    auto grid = std::make_shared<feather::Tensor>();
    grid->Assign<float>({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2});
    auto stride = std::make_shared<feather::Tensor>();
    stride->Assign<float>({8.0f}, {1});
    auto wh_scale = std::make_shared<feather::Tensor>();
    wh_scale->Assign<float>({2.0f}, {1});
    auto anchor = std::make_shared<feather::Tensor>();
    anchor->Assign<float>({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 6});

    feather::operators::YoloDecodeParam param{};
    param.input = raw;
    param.xy_scale = xy_scale;
    param.grid = grid;
    param.stride = stride;
    param.wh_scale = wh_scale;
    param.anchor_grid = anchor;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP32, "YoloDecode");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::YoloDecodeKernel<feather::DeviceType::CUDA, feather::DataType::FP32>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {
        (1.0f + 10.0f) * 8.0f,
        (1.0f + 20.0f) * 8.0f,
        4.0f,
        6.0f,
        1.0f / (1.0f + std::exp(-1.0f)),
        1.0f / (1.0f + std::exp(1.0f)),
        ((1.0f / (1.0f + std::exp(-2.0f))) * 2.0f + 30.0f) * 8.0f,
        ((1.0f / (1.0f + std::exp(2.0f))) * 2.0f + 40.0f) * 8.0f,
        std::pow((1.0f / (1.0f + std::exp(-0.5f))) * 2.0f, 2.0f) * 8.0f,
        std::pow((1.0f / (1.0f + std::exp(0.5f))) * 2.0f, 2.0f) * 10.0f,
        0.5f,
        1.0f / (1.0f + std::exp(-4.0f)),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out->data<float>()[i], expected[i], 1e-5f);
    }
#endif
}

TEST(cuda_kernel_test, YoloDecodeRunsOnCudaFP16) {
#ifndef FEATHER_WITH_CUDA
    GTEST_SKIP() << "CUDA kernels are not built";
#else
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto make_half = [](const std::vector<float>& values, const std::vector<int64_t>& shape) {
        std::vector<uint16_t> half(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            half[i] = feather::FloatToHalf(values[i]);
        }
        auto tensor = std::make_shared<feather::Tensor>();
        tensor->Assign<uint16_t>(half, shape);
        return tensor;
    };

    auto raw = make_half({0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
                          2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 4.0f},
                         {1, 12, 1, 1});
    auto xy_scale = make_half({2.0f}, {1});
    auto grid = make_half({10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 1, 1, 2});
    auto stride = make_half({8.0f}, {1});
    auto wh_scale = make_half({2.0f}, {1});
    auto anchor = make_half({4.0f, 6.0f, 8.0f, 10.0f}, {1, 2, 1, 1, 2});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{1, 2, 6});
    out->mutable_data<uint16_t>();

    feather::operators::YoloDecodeParam param{};
    param.input = raw;
    param.xy_scale = xy_scale;
    param.grid = grid;
    param.stride = stride;
    param.wh_scale = wh_scale;
    param.anchor_grid = anchor;
    param.out = out;

    auto kernel =
        feather::KernelDispatcher::instance().create(feather::DeviceType::CUDA, feather::DataType::FP16, "YoloDecode");
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE((dynamic_cast<feather::kernel::YoloDecodeKernel<feather::DeviceType::CUDA, feather::DataType::FP16>*>(
                  kernel.get())),
              nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {
        (1.0f + 10.0f) * 8.0f,
        (1.0f + 20.0f) * 8.0f,
        4.0f,
        6.0f,
        1.0f / (1.0f + std::exp(-1.0f)),
        1.0f / (1.0f + std::exp(1.0f)),
        ((1.0f / (1.0f + std::exp(-2.0f))) * 2.0f + 30.0f) * 8.0f,
        ((1.0f / (1.0f + std::exp(2.0f))) * 2.0f + 40.0f) * 8.0f,
        std::pow((1.0f / (1.0f + std::exp(-0.5f))) * 2.0f, 2.0f) * 8.0f,
        std::pow((1.0f / (1.0f + std::exp(0.5f))) * 2.0f, 2.0f) * 10.0f,
        0.5f,
        1.0f / (1.0f + std::exp(-4.0f)),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 0.2f);
    }
#endif
}
