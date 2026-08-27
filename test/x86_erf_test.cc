#include <gtest/gtest.h>

#include <memory>

#include "core/kernel.h"
#include "src/kernel/erf.h"

TEST(x86_erf_test, Fp32KernelIsRegisteredOnX86) {
    auto kernel = feather::KernelDispatcher::instance().create(feather::DeviceType::X86,
                                                                feather::DataType::FP32, "Erf");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->device(), feather::DeviceType::X86);
    auto* x86_kernel =
        dynamic_cast<feather::kernel::ErfKernel<feather::DeviceType::X86, feather::DataType::FP32>*>(kernel.get());
    EXPECT_NE(x86_kernel, nullptr);
}
