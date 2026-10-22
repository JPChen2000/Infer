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
#include <iostream>
#include <memory>
#include "util/types.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "util/logger.h"

using namespace feather;

TEST(tensor_test, TestX86) {
    auto buffer_pool = BufferPool::getInstance();
    auto buffer = buffer_pool->allocate(1024);
    EXPECT_EQ(buffer->size(), 1024);
    auto tensor = Tensor(buffer);
    std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int64_t> shape = {2, 3, 2};
    DDim dim(shape);
    tensor.Assign<float, DDim>(data.data(), dim);
    EXPECT_EQ(tensor.data_size(), 12);
    std::cout << tensor;
    std::vector<float> data1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15};
    std::vector<int64_t> shape1 = {3, 5};
    tensor.Assign<float>(data1, shape1);
    EXPECT_EQ(tensor.data_size(), 15);
    std::cout << tensor;
}

TEST(tensor_test, SwapsStorageForCompatibleTensors) {
    Tensor lhs(sizeof(int32_t) * 2);
    Tensor rhs(sizeof(int32_t) * 2);
    lhs.Resize({2});
    rhs.Resize({2});
    lhs.mutable_data<int32_t>()[0] = 11;
    lhs.mutable_data<int32_t>()[1] = 12;
    rhs.mutable_data<int32_t>()[0] = 21;
    rhs.mutable_data<int32_t>()[1] = 22;

    const void* lhs_storage = lhs.raw_data();
    const void* rhs_storage = rhs.raw_data();
    lhs.SwapStorage(rhs);

    EXPECT_EQ(lhs.raw_data(), rhs_storage);
    EXPECT_EQ(rhs.raw_data(), lhs_storage);
    EXPECT_EQ(lhs.dims().data(), std::vector<int64_t>({2}));
    EXPECT_EQ(rhs.dims().data(), std::vector<int64_t>({2}));
    EXPECT_EQ(lhs.data<int32_t>()[0], 21);
    EXPECT_EQ(lhs.data<int32_t>()[1], 22);
    EXPECT_EQ(rhs.data<int32_t>()[0], 11);
    EXPECT_EQ(rhs.data<int32_t>()[1], 12);
}
