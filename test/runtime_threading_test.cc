#include <gtest/gtest.h>

#include <algorithm>
#include <thread>

#include "core/graph.h"

namespace feather {
namespace {

TEST(runtime_threading_test, DefaultThreadModeUsesParallelGraphExecution) {
    RuntimeGraph graph;
    EXPECT_EQ(graph.ThreadMode(), RuntimeThreadMode::kParallelGraph);
}

TEST(runtime_threading_test, DefaultThreadCountMatchesHardwareConcurrency) {
    RuntimeGraph graph;
    const size_t expected_thread_count = std::max<size_t>(1, std::thread::hardware_concurrency());
    EXPECT_EQ(graph.ThreadCount(), expected_thread_count);
}

TEST(runtime_threading_test, ThreadCountCanBeConfigured) {
    RuntimeGraph graph;
    graph.SetThreadCount(4);
    EXPECT_EQ(graph.ThreadCount(), 4u);
}

TEST(runtime_threading_test, SerialGraphModeDoesNotCreateGraphThreadPool) {
    RuntimeGraph graph;
    graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    graph.SetThreadCount(8);

    RuntimeNode node_a;
    node_a.name = "node_a";
    node_a.op_type = "Identity";
    graph.AddNode(std::move(node_a));

    RuntimeNode node_b;
    node_b.name = "node_b";
    node_b.op_type = "Identity";
    graph.AddNode(std::move(node_b));

    EXPECT_EQ(graph.Finalize(), 0);
    EXPECT_EQ(graph.WorkerCount(), 1u);
}

TEST(runtime_threading_test, DefaultParallelGraphModeCreatesThreadPoolWhenGraphHasMultipleNodes) {
    RuntimeGraph graph;

    RuntimeNode node_a;
    node_a.name = "node_a";
    node_a.op_type = "Identity";
    graph.AddNode(std::move(node_a));

    RuntimeNode node_b;
    node_b.name = "node_b";
    node_b.op_type = "Identity";
    graph.AddNode(std::move(node_b));

    EXPECT_EQ(graph.Finalize(), 0);
    EXPECT_EQ(graph.WorkerCount(), 2u);
}

}  // namespace
}  // namespace feather
