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
