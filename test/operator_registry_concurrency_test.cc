// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "core/operator_registry.h"

namespace feather {
namespace {
struct BuilderGate { std::mutex mutex; std::condition_variable condition; bool entered{false}, release{false}; };
std::shared_ptr<OpBase> PauseBuilder(BuilderGate* gate) {
    std::unique_lock<std::mutex> lock(gate->mutex); gate->entered = true; gate->condition.notify_all();
    gate->condition.wait(lock, [&] { return gate->release; }); return nullptr;
}
TEST(operator_registry_concurrency_test, ReplacementRetainsActiveBuilderSnapshot) {
    auto& registry = OperatorRegistry::instance(); BuilderGate gate;
    auto owner = std::make_shared<int>(7); std::weak_ptr<int> lifetime = owner;
    registry.Register("ActiveBuilderLifetime", [owner, &gate](const auto&, auto&, const auto&) { return PauseBuilder(&gate); });
    owner.reset();
    std::thread reader([&] { model::NodeDesc n; n.op_type = "ActiveBuilderLifetime"; OperatorRegistry::TensorMap t; registry.Create(n, t); });
    { std::unique_lock<std::mutex> lock(gate.mutex); gate.condition.wait(lock, [&] { return gate.entered; }); }
    registry.Register("ActiveBuilderLifetime", [](const auto&, auto&, const auto&) { return std::shared_ptr<OpBase>{}; });
    EXPECT_FALSE(lifetime.expired());
    { std::lock_guard<std::mutex> lock(gate.mutex); gate.release = true; } gate.condition.notify_all(); reader.join();
    EXPECT_TRUE(lifetime.expired());
}
TEST(operator_registry_concurrency_test, BuildersCanRegisterAndRecursivelyCreateOperators) {
    auto& registry = OperatorRegistry::instance();
    registry.Register("RecursiveBuilder", [&](const auto&, auto& tensors, const auto& context) {
        registry.Register("InnerBuilder", [](const auto&, auto&, const auto&) { return std::shared_ptr<OpBase>{}; });
        model::NodeDesc inner; inner.op_type = "InnerBuilder"; return registry.Create(inner, tensors, context);
    });
    model::NodeDesc node; node.op_type = "RecursiveBuilder"; OperatorRegistry::TensorMap tensors;
    EXPECT_EQ(registry.Create(node, tensors), nullptr);
}
TEST(operator_registry_concurrency_test, RetiringBuilderCanRegisterFromCapturedOwnerDestructor) {
    EXPECT_EXIT({
        alarm(3);
        auto& registry = OperatorRegistry::instance();
        auto owner = std::shared_ptr<int>(new int(7), [](int* value) {
            OperatorRegistry::instance().Register("RetiredBuilderNested",
                [](const auto&, auto&, const auto&) { return std::shared_ptr<OpBase>{}; });
            delete value;
        });
        registry.Register("RetiredBuilder", [owner](const auto&, auto&, const auto&) { return std::shared_ptr<OpBase>{}; });
        owner.reset();
        registry.Register("RetiredBuilder", [](const auto&, auto&, const auto&) { return std::shared_ptr<OpBase>{}; });
        _exit(0);
    }, ::testing::ExitedWithCode(0), "");
}
}  // namespace
}  // namespace feather
