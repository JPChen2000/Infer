#include <gtest/gtest.h>

#include <atomic>
#include <array>

#include "core/graph.h"
#include "src/kernel/gemm.h"
#include "util/thread_pool_nv.h"
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

TEST(runtime_threading_test, ThreadPoolBatchCompletesEveryTaskBeforeReturning) {
    ThreadPoolNv pool(3);
    std::array<std::atomic<int>, 11> executions{};

    pool.RunBatch(11, [&executions](int task_index) {
        executions[static_cast<size_t>(task_index)].fetch_add(1, std::memory_order_relaxed);
    });

    for (const auto& execution_count : executions) {
        EXPECT_EQ(execution_count.load(std::memory_order_relaxed), 1);
    }
}

TEST(runtime_threading_test, ThreadPoolBatchCanBeReusedAcrossParallelRounds) {
    ThreadPoolNv pool(8);
    for (int round = 0; round < 100; ++round) {
        std::array<std::atomic<int>, 17> executions{};
        pool.RunBatch(executions.size(), [&executions](int task_index) {
            executions[static_cast<size_t>(task_index)].fetch_add(1, std::memory_order_relaxed);
        });
        for (const auto& execution_count : executions) {
            EXPECT_EQ(execution_count.load(std::memory_order_relaxed), 1);
        }
    }
}

TEST(runtime_threading_test, ThreadPoolProcessesQueuedWorkAfterBatchCompletion) {
    ThreadPoolNv pool(4);
    std::atomic<int> batch_executions{0};
    pool.RunBatch(1, [&batch_executions](int) { batch_executions.fetch_add(1, std::memory_order_relaxed); });

    auto queued_result = pool.enqueue([](int worker_id) { return worker_id; });

    EXPECT_EQ(batch_executions.load(std::memory_order_relaxed), 1);
    EXPECT_GE(queued_result.get(), 0);
    pool.wait();
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

TEST(runtime_threading_test, FinalizeSafelyPreparesUnboundKernel) {
    RuntimeGraph graph;
    RuntimeNode node;
    node.name = "unbound_gemm";
    node.op_type = "Gemm";
    node.kernel = std::make_unique<feather::kernel::GemmKernel<feather::DeviceType::X86, feather::DataType::BF16>>();
    graph.AddNode(std::move(node));

    EXPECT_EQ(graph.Finalize(), 0);
    EXPECT_EQ(graph.Check(), -1);
}

}  // namespace
}  // namespace feather
