#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>
#include "util/types.h"
#include "util/fp16.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "util/logger.h"

using namespace feather;

TEST(datatype_test, TestX86) {
    auto buffer_pool = BufferPool::getInstance();
    auto buffer = buffer_pool->allocate(1024);
    EXPECT_EQ(buffer->size(), 1024);
    auto tensor = Tensor(buffer);
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {2, 3};
    DDim dim(shape);
    tensor.Assign<float, DDim>(data.data(), dim);
}

TEST(datatype_test, FP16RoundTripPreservesValuesNearTwo) {
    const float input = 1.99957275390625f;
    const uint16_t encoded = FloatToHalf(input);
    const float decoded = HalfToFloat(encoded);

    EXPECT_NEAR(decoded, input, 0.002f);
    EXPECT_GT(decoded, 1.5f);
}

TEST(datatype_test, FP16RoundTripPreservesSqrtTwoSquared) {
    const float input = 1.4140625f * 1.4140625f;
    const uint16_t encoded = FloatToHalf(input);
    const float decoded = HalfToFloat(encoded);

    EXPECT_NEAR(decoded, input, 0.002f);
    EXPECT_GT(decoded, 1.5f);
}
