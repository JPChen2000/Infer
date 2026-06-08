#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/kernel.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "src/kernel/identity.h"
#include "src/kernel/slice.h"
#include "util/fp16.h"
#include "src/operator/identity_op.h"
#include "src/operator/params.h"
#include "src/operator/slice_op.h"

using feather::DataType;
using feather::DeviceType;
using feather::KernelDispatcher;
using feather::OpBase;
using feather::Tensor;
using feather::operators::SliceParam;
using feather::operators::UnaryParam;

TEST(identity_slice_op_test, IdentityRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4}, {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    UnaryParam param{};
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::IdentityOp>("identity0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Identity");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 2}));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], input->data<float>()[i]);
    }
}

TEST(identity_slice_op_test, SliceRunsOnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    SliceParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;
    param.start = 1;
    param.end = 3;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SliceOp>("slice0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Slice");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 2}));
    const std::vector<float> expected = {2, 3, 5, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(identity_slice_op_test, SliceSupportsAxis0OnX86) {
    auto input = std::make_shared<Tensor>();
    input->Assign<float>({1, 2, 3, 4, 5, 6}, {3, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});

    SliceParam param{};
    param.input = input;
    param.out = out;
    param.axis = 0;
    param.start = 1;
    param.end = 3;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SliceOp>("slice_axis0", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Slice");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    EXPECT_EQ(out->dims().data(), std::vector<int64_t>({2, 2}));
    const std::vector<float> expected = {3, 4, 5, 6};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out->data<float>()[i], expected[i]);
    }
}

TEST(identity_slice_op_test, IdentityRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f)},
                            {2, 2});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    UnaryParam param{};
    param.input = input;
    param.out = out;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::IdentityOp>("identity_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Identity");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]),
                    feather::HalfToFloat(input->data<uint16_t>()[i]), 1e-3f);
    }
}

TEST(identity_slice_op_test, X86Fp32IdentityUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Identity");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::IdentityKernel<DeviceType::X86, DataType::FP32>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(identity_slice_op_test, X86Fp16IdentityUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Identity");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::IdentityKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(identity_slice_op_test, CommonFp16SliceKernelRunsCorrectly) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    SliceParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;
    param.start = 1;
    param.end = 3;

    auto kernel = KernelDispatcher::instance().create(DeviceType::COMMON, DataType::FP16, "Slice");
    ASSERT_NE(kernel, nullptr);
    auto* common_kernel = dynamic_cast<feather::kernel::SliceKernel<DeviceType::COMMON, DataType::FP16>*>(kernel.get());
    ASSERT_NE(common_kernel, nullptr);

    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);

    const std::vector<float> expected = {2.0f, 3.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}

TEST(identity_slice_op_test, X86Fp32SliceUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP32, "Slice");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::SliceKernel<DeviceType::X86, DataType::FP32>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(identity_slice_op_test, X86Fp16SliceUsesRegisteredKernel) {
    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Slice");
    ASSERT_NE(kernel, nullptr);
    auto* x86_kernel = dynamic_cast<feather::kernel::SliceKernel<DeviceType::X86, DataType::FP16>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}

TEST(identity_slice_op_test, SliceRunsOnX86FP16) {
    auto input = std::make_shared<Tensor>();
    input->Assign<uint16_t>({feather::FloatToHalf(1.0f), feather::FloatToHalf(2.0f), feather::FloatToHalf(3.0f),
                             feather::FloatToHalf(4.0f), feather::FloatToHalf(5.0f), feather::FloatToHalf(6.0f)},
                            {2, 3});

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{2, 2});
    out->mutable_data<uint16_t>();

    SliceParam param{};
    param.input = input;
    param.out = out;
    param.axis = 1;
    param.start = 1;
    param.end = 3;

    std::shared_ptr<OpBase> op = std::make_shared<feather::operators::SliceOp>("slice_fp16", param);
    ASSERT_EQ(op->CheckShape(), 0);
    ASSERT_EQ(op->InferOutputShapes(), 0);

    auto kernel = KernelDispatcher::instance().create(DeviceType::X86, DataType::FP16, "Slice");
    ASSERT_NE(kernel, nullptr);
    op->AttachKernel(std::move(kernel));
    ASSERT_EQ(op->Run(), 0);

    const std::vector<float> expected = {2.0f, 3.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(feather::HalfToFloat(out->data<uint16_t>()[i]), expected[i], 1e-3f);
    }
}
