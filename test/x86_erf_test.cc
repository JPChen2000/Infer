// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
