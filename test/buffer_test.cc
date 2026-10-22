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

#include "core/dim.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "util/logger.h"
#include "util/types.h"

using namespace feather;

TEST(buffer_pool_test, TestX86) {
    auto buffer = BufferPool::getInstance()->allocate(1024);
    EXPECT_EQ(buffer->size(), 1024);
}

TEST(buffer_pool_test, ReuseReleasedSizeClassBlock) {
    auto first = BufferPool::getInstance()->allocate(1000);
    auto* first_data = first->data();
    EXPECT_GE(first->size(), 1000);
    first.reset();

    auto second = BufferPool::getInstance()->allocate(1001);
    EXPECT_EQ(second->data(), first_data);
    EXPECT_GE(second->size(), 1001);
}

TEST(buffer_test, TestX86) {
    auto buffer = std::make_shared<Buffer>(1024);
    EXPECT_EQ(buffer->size(), 1024);
}
