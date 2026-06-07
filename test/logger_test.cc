#include <gtest/gtest.h>

#include <memory>
#include <typeinfo>

#include "core/operator_registry.h"
#include "src/kernel/add.h"
#include "src/kernel/concat.h"
#include "src/kernel/conv2d.h"
#include "src/kernel/pool.h"
#include "src/kernel/pow.h"
#include "src/kernel/resize.h"
#include "src/kernel/reshape.h"
#include "src/kernel/sigmoid.h"
#include "src/kernel/split.h"
#include "src/kernel/transpose.h"
#include "src/operator/add_op.h"
#include "util/logger.h"

TEST(logger_test, DebugLoggingFlagMatchesBuildMacro) {
#ifdef FEATHER_DEBUG
    EXPECT_TRUE(feather::IsDebugLoggingEnabled());
#else
    EXPECT_FALSE(feather::IsDebugLoggingEnabled());
#endif
}

TEST(logger_test, KernelSelectionFallsBackToCommonWhenX86KernelMissing) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Softmax", {}, feather::DataType::FP32);
    ASSERT_NE(kernel, nullptr);
}

TEST(logger_test, KernelSelectionUsesExactX86KernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Add", {}, feather::DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel), typeid(feather::kernel::AddKernel<feather::DeviceType::X86, feather::DataType::FP32>));
}

TEST(logger_test, KernelSelectionUsesExactX86ConvKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Conv2D", {}, feather::DataType::FP32);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::Conv2DKernel<feather::DeviceType::X86, feather::DataType::FP32>));
}

TEST(logger_test, KernelSelectionUsesExactX86SigmoidKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Sigmoid", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::SigmoidKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86ConcatKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Concat", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::ConcatKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86SplitKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Split", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::SplitKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86TransposeKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Transpose", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::TransposeKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86ResizeKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Resize", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::ResizeKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86MaxPoolKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "MaxPool", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::MaxPoolKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86PowKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Pow", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::PowKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionUsesExactX86ReshapeKernelWhenRegistered) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::X86, "Reshape", {}, feather::DataType::FP16);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(typeid(*kernel),
              typeid(feather::kernel::ReshapeKernel<feather::DeviceType::X86, feather::DataType::FP16>));
}

TEST(logger_test, KernelSelectionSupportsDirectCommonLookup) {
    auto kernel = feather::CreateKernelForTensor(feather::DeviceType::COMMON, "Softmax", {}, feather::DataType::FP32);
    ASSERT_NE(kernel, nullptr);
}

TEST(logger_test, HostDeviceSelectionIsNeverUnknown) {
    EXPECT_NE(feather::GetHostRuntimeDevice(), feather::DeviceType::UNKNOWN);
}
