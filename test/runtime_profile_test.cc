#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "core/graph.h"

namespace feather {
namespace {

TEST(runtime_profile_test, EmptyProfileSummaryStartsEmpty) {
    RuntimeGraph graph;
    EXPECT_TRUE(graph.ProfileSummaries().empty());
}

TEST(runtime_profile_test, ProfilingIsDisabledByDefault) {
    RuntimeGraph graph;
    EXPECT_FALSE(graph.ProfilingEnabled());
}

TEST(runtime_profile_test, ProfileSummaryAccumulatesNodeTiming) {
    RuntimeGraph graph;
    graph.SetProfilingEnabled(true);

    graph.RecordNodeProfile("conv0", "Conv2D", 1.5);
    graph.RecordNodeProfile("conv0", "Conv2D", 2.5);

    const auto summaries = graph.ProfileSummaries();
    ASSERT_EQ(summaries.size(), 1u);
    EXPECT_EQ(summaries[0].node_name, "conv0");
    EXPECT_EQ(summaries[0].op_type, "Conv2D");
    EXPECT_EQ(summaries[0].call_count, 2);
    EXPECT_DOUBLE_EQ(summaries[0].total_ms, 4.0);
    EXPECT_DOUBLE_EQ(summaries[0].avg_ms, 2.0);
    EXPECT_DOUBLE_EQ(summaries[0].min_ms, 1.5);
    EXPECT_DOUBLE_EQ(summaries[0].max_ms, 2.5);
}

TEST(runtime_profile_test, ReenablingProfilingStartsAFreshNodeIndex) {
    RuntimeGraph graph;
    graph.SetProfilingEnabled(true);
    graph.RecordNodeProfile("old_node", "Add", 1.0);

    graph.SetProfilingEnabled(false);
    graph.SetProfilingEnabled(true);
    graph.RecordNodeProfile("new_node", "Mul", 2.0);

    const auto summaries = graph.ProfileSummaries();
    ASSERT_EQ(summaries.size(), 1u);
    EXPECT_EQ(summaries[0].node_name, "new_node");
    EXPECT_EQ(summaries[0].op_type, "Mul");
    EXPECT_EQ(summaries[0].call_count, 1);
    EXPECT_DOUBLE_EQ(summaries[0].total_ms, 2.0);
}

TEST(runtime_profile_test, ProfileSummaryIsSafeForParallelNodeRecording) {
    RuntimeGraph graph;
    graph.SetProfilingEnabled(true);

    constexpr int kThreadCount = 8;
    constexpr int kRecordsPerThread = 1000;
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int tid = 0; tid < kThreadCount; ++tid) {
        threads.emplace_back([&graph] {
            for (int i = 0; i < kRecordsPerThread; ++i) {
                graph.RecordNodeProfile("parallel_node", "Add", 1.0);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    const auto summaries = graph.ProfileSummaries();
    ASSERT_EQ(summaries.size(), 1u);
    EXPECT_EQ(summaries[0].call_count, kThreadCount * kRecordsPerThread);
    EXPECT_DOUBLE_EQ(summaries[0].total_ms, static_cast<double>(kThreadCount * kRecordsPerThread));
    EXPECT_DOUBLE_EQ(summaries[0].avg_ms, 1.0);
}

}  // namespace
}  // namespace feather
