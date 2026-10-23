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
#include <type_traits>
#include <utility>
#include "util/types.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "core/execution_plan.h"
#include "util/logger.h"

using namespace feather;

TEST(tensor_test, RawAccessCannotWriteSharedSessionWeights) {
    // A mutable Tensor handle must not expose a writable pointer to frozen
    // weights; only explicit mutation may detach a session's storage.
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Tensor&>().raw_data()), const void*>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Tensor&>().raw_data()), const void*>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Tensor&>().storage()), std::shared_ptr<const Storage>>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Tensor&>().storage()), std::shared_ptr<const Storage>>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Storage&>().data()), const void*>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Storage&>().buffer()), std::shared_ptr<const Buffer>>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<Buffer&>().data()), const void*>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Buffer&>().data()), const void*>));
}

TEST(tensor_test, StorageIdentitySurvivesMaterializationMutationAndCopyOnWrite) {
    auto deferred = std::make_shared<Storage>(sizeof(float));
    const auto identity = deferred->id();
    const auto version = deferred->version();
    EXPECT_NE(identity, 0u);
    EXPECT_FALSE(deferred->materialized());
    EXPECT_NE(deferred->data(), nullptr);
    EXPECT_EQ(deferred->id(), identity);
    EXPECT_EQ(deferred->version(), version);

    Tensor owner; owner.Assign<float>({2}, {1});
    auto readonly = std::make_shared<Buffer>(owner.raw_data(), sizeof(float), owner.storage(), true);
    Tensor first; first.Resize({1}); first.ResetBuffer(readonly, sizeof(float));
    Tensor second; second.Resize({1}); second.ResetBuffer(readonly, sizeof(float));
    Tensor alias; alias.ShareDataWith(first);
    const auto first_id = first.storage()->id();
    const auto second_id = second.storage()->id();
    EXPECT_NE(first_id, second_id);
    EXPECT_EQ(alias.storage()->id(), first_id);
    const auto first_version = first.mutation_version();
    first.mutable_data<float>()[0] = 9;
    EXPECT_EQ(first.storage()->id(), first_id);
    EXPECT_EQ(alias.storage()->id(), first_id);
    EXPECT_EQ(second.storage()->id(), second_id);
    EXPECT_NE(first.mutation_version(), first_version);
    EXPECT_FLOAT_EQ(alias.data<float>()[0], 9);
    EXPECT_FLOAT_EQ(second.data<float>()[0], 2);
    EXPECT_FLOAT_EQ(owner.data<float>()[0], 2);
}

TEST(tensor_test, ReadOnlyBuffersRetainConstOwnersUntilCopyOnWrite) {
    auto owner = std::make_shared<const std::vector<float>>(std::vector<float>{2});
    std::weak_ptr<const std::vector<float>> lifetime = owner;
    Tensor tensor; tensor.Resize({1});
    tensor.ResetBuffer(std::make_shared<Buffer>(owner->data(), sizeof(float), owner), sizeof(float));
    owner.reset();
    EXPECT_FALSE(lifetime.expired());
    EXPECT_FLOAT_EQ(tensor.data<float>()[0], 2);
    tensor.mutable_data<float>()[0] = 9;
    EXPECT_TRUE(lifetime.expired());
    EXPECT_FLOAT_EQ(tensor.data<float>()[0], 9);
}

TEST(tensor_test, ExplicitRawWritesDetachSessionWeightsAndTrackAliasMutation) {
    model::ModelDesc model;
    model.graph.inputs = {"x"};
    model.graph.outputs = {"y"};
    for (const auto& name : {"x", "w", "y"}) {
        model::ValueDesc value;
        value.tensor.name = name;
        value.tensor.dims = {1};
        value.tensor.data_type = DataType::FP32;
        value.constant = std::string(name) == "w";
        model.graph.values.push_back(value);
    }
    model::NodeDesc add;
    add.name = "add"; add.op_type = "Add"; add.inputs = {"x", "w"}; add.outputs = {"y"};
    model.graph.nodes.push_back(add);
    StaticGraph graph;
    graph.SetKernelDevice(DeviceType::COMMON);
    ASSERT_EQ(graph.SetModel(model), 0);
    auto input = std::make_shared<Tensor>(); input->Assign<float>({1}, {1});
    auto weight = std::make_shared<Tensor>(); weight->Assign<float>({2}, {1});
    graph.SetTensor("x", input); graph.SetTensor("w", weight);
    ASSERT_TRUE(graph.BuildStatus().ok());
    std::shared_ptr<const CompiledGraph> compiled;
    ASSERT_TRUE(CompiledGraph::Compile(graph, &compiled).ok());
    std::unique_ptr<ExecutionSession> first, second, future;
    ASSERT_TRUE(compiled->CreateSession(&first).ok());
    ASSERT_TRUE(compiled->CreateSession(&second).ok());
    auto first_weight = first->runtime().GetTensor("w");
    auto second_weight = second->runtime().GetTensor("w");
    Tensor alias; alias.ShareDataWith(*first_weight);
    const auto first_version = first_weight->mutation_version();
    const auto second_version = second_weight->mutation_version();
    const void* shared = first_weight->raw_data();
    EXPECT_EQ(shared, second_weight->raw_data());
    EXPECT_EQ(first_weight->mutation_version(), first_version);
    static_cast<float*>(first_weight->mutable_data(sizeof(float)))[0] = 9;
    EXPECT_NE(first_weight->raw_data(), shared);
    EXPECT_EQ(alias.raw_data(), first_weight->raw_data());
    EXPECT_FLOAT_EQ(alias.data<float>()[0], 9);
    EXPECT_NE(alias.mutation_version(), first_version);
    EXPECT_EQ(alias.mutation_version(), first_weight->mutation_version());
    EXPECT_EQ(second_weight->mutation_version(), second_version);
    EXPECT_FLOAT_EQ(second_weight->data<float>()[0], 2);
    ASSERT_TRUE(compiled->CreateSession(&future).ok());
    ASSERT_TRUE(first->Run().ok());
    ASSERT_TRUE(second->Run().ok());
    ASSERT_TRUE(future->Run().ok());
    EXPECT_FLOAT_EQ(first->runtime().GetTensor("y")->data<float>()[0], 10);
    EXPECT_FLOAT_EQ(second->runtime().GetTensor("y")->data<float>()[0], 3);
    EXPECT_FLOAT_EQ(future->runtime().GetTensor("y")->data<float>()[0], 3);
}

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
