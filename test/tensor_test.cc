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
