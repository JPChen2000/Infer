# Common Operators Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first wave of common operators by implementing `Gemm`, `Sigmoid`, and `Reshape` through the existing static-graph and runtime-graph pipeline.

**Architecture:** Reuse the current `OperatorRegistry`-driven operator construction flow. Each operator gets a semantic `Op`, an `x86 + FP32` kernel, shape inference, and graph-level tests through `StaticGraph -> GraphLowering -> RuntimeGraph`.

**Tech Stack:** C++17, CMake, gtest

---

### Task 1: Add Gemm Tests First

**Files:**
- Create: `test/gemm_op_test.cc`
- Modify: `test/graph_pipeline_test.cc`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(gemm_op_test, GemmRunsOnX86) {
    // Build a 2D gemm with optional bias and assert exact output.
}

TEST(runtime_graph_pipeline_test, BuildAndRunGemmSigmoidGraph) {
    // Build ModelDesc with Gemm -> Sigmoid and assert graph output.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build -DBUILD_UNITTEST=ON && cmake --build build --target unit_tests && ./build/bin/unit_tests '--gtest_filter=gemm_op_test.*:runtime_graph_pipeline_test.BuildAndRunGemmSigmoidGraph'`
Expected: FAIL because `Gemm` op/kernel do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class GemmOp : public OpBase {};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=gemm_op_test.*:runtime_graph_pipeline_test.BuildAndRunGemmSigmoidGraph'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/gemm_op_test.cc test/graph_pipeline_test.cc
git commit -m "test: add gemm operator coverage"
```

### Task 2: Implement Gemm Operator And Kernel

**Files:**
- Create: `src/operator/gemm_op.h`
- Create: `src/operator/gemm_op.cc`
- Create: `src/kernel/gemm.h`
- Create: `src/kernel/x86/gemm.cc`
- Modify: `src/operator/params.h`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(static_graph_test, BuildStaticGraphFromGemmModelDesc) {
    // Build a graph containing Gemm and assert output tensor shape is inferred.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_test.BuildStaticGraphFromGemmModelDesc'`
Expected: FAIL because `OperatorRegistry` cannot create `Gemm`.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct GemmParam : ParamBase {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> out;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=gemm_op_test.*:static_graph_test.BuildStaticGraphFromGemmModelDesc'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/operator/gemm_op.h src/operator/gemm_op.cc src/kernel/gemm.h src/kernel/x86/gemm.cc src/operator/params.h CMakeLists.txt test/static_graph_test.cc
git commit -m "feat: add gemm operator and x86 kernel"
```

### Task 3: Add Sigmoid Tests And Implementation

**Files:**
- Create: `test/sigmoid_op_test.cc`
- Create: `src/operator/sigmoid_op.h`
- Create: `src/operator/sigmoid_op.cc`
- Modify: `src/kernel/sigmoid.h`
- Create: `src/kernel/x86/sigmoid.cc`
- Modify: `test/graph_pipeline_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(sigmoid_op_test, SigmoidRunsOnX86) {
    // Assert values match sigmoid reference outputs.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=sigmoid_op_test.*'`
Expected: FAIL because `Sigmoid` op/kernel do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class SigmoidOp : public OpBase {};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=sigmoid_op_test.*:runtime_graph_pipeline_test.BuildAndRunGemmSigmoidGraph'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/sigmoid_op_test.cc src/operator/sigmoid_op.h src/operator/sigmoid_op.cc src/kernel/sigmoid.h src/kernel/x86/sigmoid.cc test/graph_pipeline_test.cc CMakeLists.txt
git commit -m "feat: add sigmoid operator and x86 kernel"
```

### Task 4: Add Reshape Tests And Implementation

**Files:**
- Create: `test/reshape_op_test.cc`
- Create: `src/operator/reshape_op.h`
- Create: `src/operator/reshape_op.cc`
- Create: `src/kernel/reshape.h`
- Create: `src/kernel/x86/reshape.cc`
- Modify: `src/operator/params.h`
- Modify: `test/graph_pipeline_test.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(reshape_op_test, ReshapePreservesDataAndChangesShape) {
    // Assert same data, new shape.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=reshape_op_test.*'`
Expected: FAIL because `Reshape` op/kernel do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct ReshapeParam : ParamBase {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> out;
    std::vector<int64_t> target_shape;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=reshape_op_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/reshape_op_test.cc src/operator/reshape_op.h src/operator/reshape_op.cc src/kernel/reshape.h src/kernel/x86/reshape.cc src/operator/params.h test/graph_pipeline_test.cc CMakeLists.txt
git commit -m "feat: add reshape operator and x86 kernel"
```

### Task 5: Verify Whole Tree And Update Overview

**Files:**
- Modify: `docs/project_overview.md`

- [ ] **Step 1: Write the failing check**

```text
Search the overview for the old operator set summary that only mentions FC, Conv2D, and ReLU.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n 'FC / Conv2D / ReLU|FC.*Conv2D.*ReLU' docs/project_overview.md`
Expected: Matches old wording.

- [ ] **Step 3: Write minimal implementation**

```md
Update the overview to mention `Gemm`, `Sigmoid`, and `Reshape`.
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add docs/project_overview.md
git commit -m "docs: update operator coverage overview"
```
