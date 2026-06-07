#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include "util/types.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "util/logger.h"
#include "core/kernel.h"
#include "src/kernel/fc.h"
#include "src/operator/params.h"


using namespace feather;
using feather::operators::FcParam;

TEST(fccompute_test, TestX86) {
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {3, 2};
    auto tensor = std::make_shared<Tensor>();
    tensor->Assign(data, shape);
    EXPECT_EQ(tensor->data_size(), 6);

    auto input = std::make_shared<Tensor>();
    std::vector<float> data1 = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> shape1 = {2, 3};
    input->Assign(data1, shape1);
    EXPECT_EQ(input->data_size(), 6);

    auto bais = std::make_shared<Tensor>();
    bais->Assign<float>({0.123, 1.45, 2.13, 2.22}, {2, 2});
    
    std::vector<int64_t> shape2 = {2, 2};
    auto out = std::make_shared<Tensor>(shape2);
    std::cout << *out;
    auto kernel = feather::kernel::FcKernel<DeviceType::COMMON, DataType::FP32>();
    FcParam param;
    param.w = tensor;
    param.bias = bais;
    param.out = out;
    param.input = input;

    kernel.SetParam((void *)&param);
    kernel.compute();
    std::cout << *out;
    std::cout << *input;
    std::cout << *tensor;
}
