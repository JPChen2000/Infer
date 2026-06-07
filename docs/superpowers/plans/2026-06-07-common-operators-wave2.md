# Common Operators Wave 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the next wave of common operators by implementing `AvgPool`, `MaxPool`, `Concat`, `Split`, and `Softmax` on the existing static-graph/runtime-graph execution pipeline.

**Architecture:** Reuse the current operator registration flow and keep implementations intentionally minimal: 2D tensors, x86 FP32 kernels, and graph-level coverage that exercises shape change, fan-in, fan-out, and row-wise post-processing.

**Tech Stack:** C++17, CMake, gtest

---

### Task 1: Add Pool Tests First

**Files:**
- Create: `test/pool_op_test.cc`
- Modify: `test/graph_pipeline_test.cc`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(pool_op_test, AvgPoolRunsOnX86) {
    // Assert 2D avg pool output.
}

TEST(pool_op_test, MaxPoolRunsOnX86) {
    // Assert 2D max pool output.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build -DBUILD_UNITTEST=ON && cmake --build build --target unit_tests && ./build/bin/unit_tests '--gtest_filter=pool_op_test.*'`
Expected: FAIL because pool operators do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class AvgPoolOp : public OpBase {};
class MaxPoolOp : public OpBase {};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=pool_op_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/pool_op_test.cc test/graph_pipeline_test.cc
git commit -m "test: add pool operator coverage"
```

### Task 2: Implement AvgPool And MaxPool

**Files:**
- Create: `src/operator/pool_op.h`
- Create: `src/operator/pool_op.cc`
- Create: `src/kernel/pool.h`
- Create: `src/kernel/x86/pool.cc`
- Modify: `src/operator/params.h`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(runtime_graph_pipeline_test, BuildAndRunAvgPoolReluGraph) {
    // Build AvgPool -> ReLU graph and assert output.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=runtime_graph_pipeline_test.BuildAndRunAvgPoolReluGraph'`
Expected: FAIL because `AvgPool` is not registered.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct PoolParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=pool_op_test.*:runtime_graph_pipeline_test.BuildAndRunAvgPoolReluGraph'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/operator/pool_op.h src/operator/pool_op.cc src/kernel/pool.h src/kernel/x86/pool.cc src/operator/params.h CMakeLists.txt test/graph_pipeline_test.cc
git commit -m "feat: add avgpool and maxpool operators"
```

### Task 3: Add Concat And Split

**Files:**
- Create: `test/concat_split_op_test.cc`
- Create: `src/operator/concat_op.h`
- Create: `src/operator/concat_op.cc`
- Create: `src/operator/split_op.h`
- Create: `src/operator/split_op.cc`
- Create: `src/kernel/concat.h`
- Create: `src/kernel/split.h`
- Create: `src/kernel/x86/concat.cc`
- Create: `src/kernel/x86/split.cc`
- Modify: `src/operator/params.h`
- Modify: `test/graph_pipeline_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(concat_split_op_test, ConcatRunsOnX86) {}
TEST(concat_split_op_test, SplitRunsOnX86) {}
TEST(runtime_graph_pipeline_test, BuildAndRunConcatSplitGraph) {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=concat_split_op_test.*:runtime_graph_pipeline_test.BuildAndRunConcatSplitGraph'`
Expected: FAIL because `Concat` and `Split` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct ConcatParam : ParamBase {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> out;
    int32_t axis;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=concat_split_op_test.*:runtime_graph_pipeline_test.BuildAndRunConcatSplitGraph'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/concat_split_op_test.cc src/operator/concat_op.h src/operator/concat_op.cc src/operator/split_op.h src/operator/split_op.cc src/kernel/concat.h src/kernel/split.h src/kernel/x86/concat.cc src/kernel/x86/split.cc src/operator/params.h test/graph_pipeline_test.cc CMakeLists.txt
git commit -m "feat: add concat and split operators"
```

### Task 4: Add Softmax

**Files:**
- Create: `test/softmax_op_test.cc`
- Create: `src/operator/softmax_op.h`
- Create: `src/operator/softmax_op.cc`
- Create: `src/kernel/softmax.h`
- Create: `src/kernel/x86/softmax.cc`
- Modify: `test/graph_pipeline_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(softmax_op_test, SoftmaxRunsOnX86) {}
TEST(runtime_graph_pipeline_test, BuildAndRunGemmSoftmaxGraph) {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=softmax_op_test.*:runtime_graph_pipeline_test.BuildAndRunGemmSoftmaxGraph'`
Expected: FAIL because `Softmax` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class SoftmaxOp : public OpBase {};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=softmax_op_test.*:runtime_graph_pipeline_test.BuildAndRunGemmSoftmaxGraph'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/softmax_op_test.cc src/operator/softmax_op.h src/operator/softmax_op.cc src/kernel/softmax.h src/kernel/x86/softmax.cc test/graph_pipeline_test.cc CMakeLists.txt
git commit -m "feat: add softmax operator"
```

### Task 5: Verify Whole Tree And Update Overview

**Files:**
- Modify: `docs/project_overview.md`
- Modify: `docs/development_roadmap.md`

- [ ] **Step 1: Write the failing check**

```text
Search docs for operator coverage summaries that do not include the new wave.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n 'Gemm / Conv2D / ReLU / Sigmoid / Reshape|Pooling|Concat|Split|Softmax' docs/project_overview.md docs/development_roadmap.md`
Expected: Missing or outdated wording.

- [ ] **Step 3: Write minimal implementation**

```md
Update docs to include:
- AvgPool / MaxPool
- Concat / Split
- Softmax
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add docs/project_overview.md docs/development_roadmap.md
git commit -m "docs: update operator coverage for wave 2"
```
