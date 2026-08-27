#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <typeinfo>
#include <vector>

#include "core/kernel.h"
#include "core/tensor.h"
#include "src/kernel/exp.h"
#include "src/operator/params.h"
#include "src/kernel/sqrt.h"
#include "src/kernel/sub.h"
#include "src/kernel/x86/elementwise.h"
#include "util/threading.h"

#ifdef FEATHER_WITH_OPENMP
#include <omp.h>
#endif

TEST(x86_math_kernel_test, RegistersFp32SubExpAndSqrtKernels) {
    auto sub = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                             feather::DataType::FP32, "Sub");
    auto exp = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                             feather::DataType::FP32, "Exp");
    auto sqrt = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                              feather::DataType::FP32, "Sqrt");

    ASSERT_NE(sub, nullptr);
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);
    EXPECT_EQ(typeid(*sub),
              typeid(feather::kernel::SubKernel<feather::DeviceType::X86, feather::DataType::FP32>));
    EXPECT_EQ(typeid(*exp),
              typeid(feather::kernel::ExpKernel<feather::DeviceType::X86, feather::DataType::FP32>));
    EXPECT_EQ(typeid(*sqrt),
              typeid(feather::kernel::SqrtKernel<feather::DeviceType::X86, feather::DataType::FP32>));
}

TEST(x86_math_kernel_test, SubUsesVectorLastDimensionBroadcast) {
    auto lhs = std::make_shared<feather::Tensor>();
    lhs->Assign<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3});
    auto rhs = std::make_shared<feather::Tensor>();
    rhs->Assign<float>({1.0f, 2.0f}, {2, 1});
    auto out = std::make_shared<feather::Tensor>(std::vector<int64_t>{2, 3});
    out->mutable_data<float>();

    feather::operators::BinaryParam param{};
    param.lhs = lhs;
    param.rhs = rhs;
    param.out = out;
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                 feather::DataType::FP32, "Sub");
    ASSERT_NE(kernel, nullptr);
    kernel->SetParam(&param);
    ASSERT_EQ(kernel->compute(), 0);
    const std::vector<float> expected = {9.0f, 19.0f, 29.0f, 38.0f, 48.0f, 58.0f};
    EXPECT_EQ(std::vector<float>(out->data<float>(), out->data<float>() + expected.size()), expected);
}

TEST(x86_math_kernel_test, ExpAndSqrtPreserveFp32Values) {
    auto input = std::make_shared<feather::Tensor>();
    input->Assign<float>({0.0f, 1.0f, 4.0f, 9.0f}, {4});
    auto exp_out = std::make_shared<feather::Tensor>(std::vector<int64_t>{4});
    exp_out->mutable_data<float>();
    auto sqrt_out = std::make_shared<feather::Tensor>(std::vector<int64_t>{4});
    sqrt_out->mutable_data<float>();

    feather::operators::UnaryParam exp_param{};
    exp_param.input = input;
    exp_param.out = exp_out;
    auto exp_kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                     feather::DataType::FP32, "Exp");
    ASSERT_NE(exp_kernel, nullptr);
    exp_kernel->SetParam(&exp_param);
    ASSERT_EQ(exp_kernel->compute(), 0);

    feather::operators::UnaryParam sqrt_param{};
    sqrt_param.input = input;
    sqrt_param.out = sqrt_out;
    auto sqrt_kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                      feather::DataType::FP32, "Sqrt");
    ASSERT_NE(sqrt_kernel, nullptr);
    sqrt_kernel->SetParam(&sqrt_param);
    ASSERT_EQ(sqrt_kernel->compute(), 0);

    for (size_t index = 0; index < 4; ++index) {
        EXPECT_FLOAT_EQ(exp_out->data<float>()[index], std::exp(input->data<float>()[index]));
        EXPECT_FLOAT_EQ(sqrt_out->data<float>()[index], std::sqrt(input->data<float>()[index]));
    }
}

TEST(x86_math_kernel_test, LargeFp32ElementwiseWorkUsesParallelRowPolicy) {
#ifdef FEATHER_WITH_OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(feather::DefaultThreadCount()));
    EXPECT_GT(feather::kernel::x86::elementwise_detail::Fp32ElementwiseWorkerCount(197, 3072), 1U);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#else
    GTEST_SKIP() << "OpenMP is disabled for this build";
#endif
}
