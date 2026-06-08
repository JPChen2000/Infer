#include <gtest/gtest.h>

#include "core/graph.h"
#include "util/threading.h"

namespace feather {
namespace {

TEST(runtime_threading_test, DefaultThreadModeUsesParallelGraphExecution) {
    RuntimeGraph graph;
    EXPECT_EQ(graph.ThreadMode(), RuntimeThreadMode::kParallelGraph);
}

TEST(runtime_threading_test, DefaultThreadCountUsesConfiguredLimit) {
    RuntimeGraph graph;
    EXPECT_EQ(feather::DefaultThreadCount(), 8u);
    EXPECT_EQ(graph.ThreadCount(), feather::DefaultThreadCount());
}

TEST(runtime_threading_test, WorkItemThreadCountIsCappedAtConfiguredLimit) {
    EXPECT_EQ(feather::ThreadCountForWorkItems(0), 1u);
    EXPECT_EQ(feather::ThreadCountForWorkItems(1), 1u);
    EXPECT_EQ(feather::ThreadCountForWorkItems(4), 4u);
    EXPECT_EQ(feather::ThreadCountForWorkItems(64), feather::DefaultThreadCount());
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
