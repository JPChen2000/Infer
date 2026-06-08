# X86 CPU Optimization Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repeatable x86 CPU benchmarking, per-op runtime profiling, and a unified threading policy so later kernel optimizations are guided by measurements and avoid nested thread oversubscription.

**Architecture:** Extend the existing `StaticGraph -> RuntimeGraph -> Kernel` execution path instead of introducing a separate benchmark runtime. Profiling data should be accumulated at runtime-node boundaries, surfaced through the demo runner, and used to compare thread modes where graph-level and kernel-level parallelism do not fight each other.

**Tech Stack:** C++17, CMake, gtest, existing `ThreadPoolNv`, existing YOLOv5 demo pipeline

---

### File Structure Map

**Files and responsibilities:**

- Modify: `include/core/graph.h`
  - Add runtime profiling data structures and thread-mode configuration APIs.
- Modify: `src/core/graph.cc`
  - Record per-node timing, expose profiling summaries, and make graph execution mode configurable.
- Modify: `include/demo/yolov5_runner.h`
  - Surface benchmark/profiling configuration and summarized results to callers.
- Modify: `src/demo/yolov5_runner.cc`
  - Add warmup + iteration benchmark flow and emit per-op summaries.
- Modify: `src/kernel/x86/conv2d.cc`
  - Make x86 convolution respect the new unified threading policy instead of always using its own internal pool.
- Modify: `CMakeLists.txt`
  - Build any new benchmark helper target or test file.
- Create: `test/runtime_profile_test.cc`
  - Verify profiling accumulation and reporting.
- Create: `test/runtime_threading_test.cc`
  - Verify the chosen thread mode keeps runtime results correct.
- Create: `demo/yolov5_benchmark_demo.cc`
  - Provide a dedicated benchmark entry point for repeatable CPU runs.
- Modify: `docs/project_overview.md`
  - Document the new CPU benchmark/profiling workflow if phase 1 lands.

---

### Task 1: Add Runtime Profiling Tests First

**Files:**
- Create: `test/runtime_profile_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>

#include "core/graph.h"

namespace feather {
namespace {

TEST(runtime_profile_test, EmptyProfileSummaryStartsEmpty) {
    RuntimeGraph graph;
    EXPECT_TRUE(graph.ProfileSummaries().empty());
}

TEST(runtime_profile_test, ProfilingCanBeDisabledByDefault) {
    RuntimeGraph graph;
    EXPECT_FALSE(graph.ProfilingEnabled());
}

}  // namespace
}  // namespace feather
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build -DBUILD_UNITTEST=ON && cmake --build build --target unit_tests && ./build/bin/unit_tests '--gtest_filter=runtime_profile_test.*'`
Expected: FAIL because `RuntimeGraph` does not expose `ProfileSummaries()` or `ProfilingEnabled()` yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct RuntimeProfileSummary {
    std::string node_name;
    std::string op_type;
    int64_t call_count{0};
    double total_ms{0.0};
    double avg_ms{0.0};
};

class RuntimeGraph {
   public:
    bool ProfilingEnabled() const { return profiling_enabled_; }
    const std::vector<RuntimeProfileSummary>& ProfileSummaries() const { return profile_summaries_; }

   private:
    bool profiling_enabled_{false};
    std::vector<RuntimeProfileSummary> profile_summaries_;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_profile_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/runtime_profile_test.cc CMakeLists.txt include/core/graph.h
git commit -m "test: add runtime profiling API coverage"
```

### Task 2: Implement Profiling Aggregation In RuntimeGraph

**Files:**
- Modify: `include/core/graph.h`
- Modify: `src/core/graph.cc`
- Modify: `test/runtime_profile_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
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
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_profile_test.ProfileSummaryAccumulatesNodeTiming'`
Expected: FAIL because `SetProfilingEnabled()` and `RecordNodeProfile()` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
void RuntimeGraph::SetProfilingEnabled(bool enabled) {
    profiling_enabled_ = enabled;
    if (!profiling_enabled_) {
        profile_summaries_.clear();
    }
}

void RuntimeGraph::RecordNodeProfile(const std::string& node_name, const std::string& op_type, double elapsed_ms) {
    if (!profiling_enabled_) {
        return;
    }
    auto it = std::find_if(profile_summaries_.begin(), profile_summaries_.end(),
                           [&](const RuntimeProfileSummary& summary) { return summary.node_name == node_name; });
    if (it == profile_summaries_.end()) {
        profile_summaries_.push_back(RuntimeProfileSummary{node_name, op_type, 1, elapsed_ms, elapsed_ms});
        return;
    }
    it->call_count += 1;
    it->total_ms += elapsed_ms;
    it->avg_ms = it->total_ms / static_cast<double>(it->call_count);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_profile_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph.h src/core/graph.cc test/runtime_profile_test.cc
git commit -m "feat: add runtime profile aggregation"
```

### Task 3: Time Runtime Nodes Through The Real Execution Path

**Files:**
- Modify: `include/core/graph.h`
- Modify: `src/core/graph.cc`
- Modify: `test/runtime_profile_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(runtime_profile_test, RuntimeNodeRunUpdatesProfileWhenEnabled) {
    RuntimeGraph graph;
    graph.SetProfilingEnabled(true);

    RuntimeNode node;
    node.name = "relu0";
    node.op_type = "ReLU";
    node.owner = std::make_shared<TestOpOwner>();
    node.kernel = std::make_unique<TestKernel>(&graph);

    ASSERT_EQ(node.Run(), 0);
    const auto summaries = graph.ProfileSummaries();
    ASSERT_EQ(summaries.size(), 1u);
    EXPECT_EQ(summaries[0].node_name, "relu0");
    EXPECT_EQ(summaries[0].call_count, 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_profile_test.RuntimeNodeRunUpdatesProfileWhenEnabled'`
Expected: FAIL because `RuntimeNode::Run()` does not report elapsed time back into the owning graph.

- [ ] **Step 3: Write minimal implementation**

```cpp
int32_t RuntimeNode::Run() {
    if (owner == nullptr || kernel == nullptr) {
        return -1;
    }
    const auto begin = std::chrono::steady_clock::now();
    const int32_t status = kernel->compute();
    const auto end = std::chrono::steady_clock::now();
    if (owner_graph != nullptr) {
        owner_graph->RecordNodeProfile(name, op_type,
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return status;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_profile_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph.h src/core/graph.cc test/runtime_profile_test.cc
git commit -m "feat: time runtime nodes during execution"
```

### Task 4: Add Thread-Mode Tests Before Changing Parallel Policy

**Files:**
- Create: `test/runtime_threading_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

```cpp
#include <gtest/gtest.h>

#include "core/graph.h"

namespace feather {
namespace {

TEST(runtime_threading_test, DefaultThreadModeIsSerialGraphExecution) {
    RuntimeGraph graph;
    EXPECT_EQ(graph.ThreadMode(), RuntimeThreadMode::kSerialGraph);
}

TEST(runtime_threading_test, ThreadCountCanBeConfigured) {
    RuntimeGraph graph;
    graph.SetThreadCount(4);
    EXPECT_EQ(graph.ThreadCount(), 4u);
}

}  // namespace
}  // namespace feather
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests '--gtest_filter=runtime_threading_test.*'`
Expected: FAIL because `RuntimeThreadMode`, `ThreadMode()`, `SetThreadCount()`, and `ThreadCount()` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
enum class RuntimeThreadMode {
    kSerialGraph,
    kParallelGraph,
};

class RuntimeGraph {
   public:
    RuntimeThreadMode ThreadMode() const { return thread_mode_; }
    void SetThreadMode(RuntimeThreadMode mode) { thread_mode_ = mode; }
    size_t ThreadCount() const { return worker_count_; }
    void SetThreadCount(size_t count) { worker_count_ = std::max<size_t>(1, count); }

   private:
    RuntimeThreadMode thread_mode_{RuntimeThreadMode::kSerialGraph};
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_threading_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/runtime_threading_test.cc CMakeLists.txt include/core/graph.h
git commit -m "test: add runtime threading API coverage"
```

### Task 5: Make RuntimeGraph Respect A Single Thread Policy

**Files:**
- Modify: `include/core/graph.h`
- Modify: `src/core/graph.cc`
- Modify: `test/runtime_threading_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(runtime_threading_test, SerialGraphModeDoesNotCreateGraphThreadPool) {
    RuntimeGraph graph;
    graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);
    graph.SetThreadCount(8);

    ASSERT_EQ(graph.Finalize(), 0);
    EXPECT_EQ(graph.WorkerCount(), 1u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_threading_test.SerialGraphModeDoesNotCreateGraphThreadPool'`
Expected: FAIL because `Finalize()` currently creates a graph pool whenever there is more than one node and hardware concurrency is greater than one.

- [ ] **Step 3: Write minimal implementation**

```cpp
int32_t RuntimeGraph::Finalize() {
    const auto status = BuildDependencies();
    if (status != 0) {
        thread_pool_.reset();
        worker_count_ = 1;
        return status;
    }

    if (thread_mode_ == RuntimeThreadMode::kSerialGraph) {
        thread_pool_.reset();
        worker_count_ = 1;
        return 0;
    }

    worker_count_ = std::max<size_t>(1, configured_thread_count_);
    worker_count_ = std::min(worker_count_, nodes_.size());
    if (worker_count_ <= 1) {
        thread_pool_.reset();
        worker_count_ = 1;
        return 0;
    }

    thread_pool_ = std::make_unique<ThreadPoolNv>(worker_count_);
    return 0;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_threading_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph.h src/core/graph.cc test/runtime_threading_test.cc
git commit -m "feat: make graph thread mode explicit"
```

### Task 6: Make X86 Conv2D Respect The Unified Thread Budget

**Files:**
- Modify: `include/core/graph.h`
- Modify: `src/kernel/x86/conv2d.cc`
- Modify: `test/runtime_threading_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(runtime_threading_test, KernelThreadBudgetCanBeReadFromRuntimeConfig) {
    RuntimeGraph graph;
    graph.SetThreadCount(6);
    graph.SetThreadMode(RuntimeThreadMode::kSerialGraph);

    EXPECT_EQ(graph.KernelThreadCount(), 6u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_threading_test.KernelThreadBudgetCanBeReadFromRuntimeConfig'`
Expected: FAIL because there is no kernel-thread budget API yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class RuntimeGraph {
   public:
    size_t KernelThreadCount() const { return configured_thread_count_; }
};

size_t GetConvThreadCount(int64_t total_work_items, size_t configured_threads) {
    if (total_work_items <= 1) {
        return 1;
    }
    return std::max<size_t>(1, std::min<size_t>(configured_threads, static_cast<size_t>(total_work_items)));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_threading_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph.h src/kernel/x86/conv2d.cc test/runtime_threading_test.cc
git commit -m "feat: share thread budget with x86 conv kernels"
```

### Task 7: Add A Dedicated YOLOv5 CPU Benchmark Entry Point

**Files:**
- Create: `demo/yolov5_benchmark_demo.cc`
- Modify: `include/demo/yolov5_runner.h`
- Modify: `src/demo/yolov5_runner.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing check**

```text
Search for a dedicated benchmark executable that supports warmup, iterations, and profiling output.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n 'warmup|iterations|profile|benchmark' demo include/demo src/demo`
Expected: Missing a dedicated benchmark entry point for repeated timed runs.

- [ ] **Step 3: Write minimal implementation**

```cpp
int main(int argc, char** argv) {
    feather::demo::BenchmarkOptions options;
    options.warmup_runs = 5;
    options.measure_runs = 20;
    options.enable_profiling = true;

    feather::demo::Yolov5Runner runner;
    // parse args, load model, run benchmark, print summary
}
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target yolov5_benchmark_demo`
Expected: PASS and produce `build/bin/yolov5_benchmark_demo`

- [ ] **Step 5: Commit**

```bash
git add demo/yolov5_benchmark_demo.cc include/demo/yolov5_runner.h src/demo/yolov5_runner.cc CMakeLists.txt
git commit -m "feat: add yolov5 cpu benchmark entry point"
```

### Task 8: Verify Phase 1 End-To-End And Document The Workflow

**Files:**
- Modify: `docs/project_overview.md`
- Modify: `docs/development_roadmap.md`

- [ ] **Step 1: Write the failing check**

```text
Search docs for benchmark and profiling instructions for x86 CPU optimization.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n 'benchmark|profiling|thread mode|x86 cpu' docs/project_overview.md docs/development_roadmap.md`
Expected: Missing or incomplete documentation for the new workflow.

- [ ] **Step 3: Write minimal implementation**

```md
Add a short section describing:
- how to build `yolov5_benchmark_demo`
- how to run warmup + measured iterations
- how to read per-op summaries
- why graph execution defaults to serial while heavy kernels use the configured thread budget
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target unit_tests yolov5_benchmark_demo && ./build/bin/unit_tests`
Expected: All tests PASS and benchmark target builds successfully.

- [ ] **Step 5: Commit**

```bash
git add docs/project_overview.md docs/development_roadmap.md
git commit -m "docs: describe x86 cpu benchmark workflow"
```
