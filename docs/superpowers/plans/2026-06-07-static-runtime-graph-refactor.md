# Static Runtime Graph Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `RuntimeGraph` op-type branching with a registry-driven `StaticGraph -> GraphLowering -> RuntimeGraph` execution pipeline.

**Architecture:** Introduce `StaticGraph` as the semantic graph layer, move operator creation and kernel binding into operator-local registry builders, then lower semantic ops into runtime executable nodes. Keep runtime execution simple and kernel-centric so later graph optimization and fusion can happen before lowering.

**Tech Stack:** C++17, CMake, gtest

---

### Task 1: Lock The New Pipeline In Tests

**Files:**
- Create: `test/static_graph_test.cc`
- Modify: `test/graph_test.cc`
- Modify: `test/graph_pipeline_test.cc`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(static_graph_test, BuildStaticGraphFromModelDesc) {
    // Build model and tensors, then assert:
    // - static graph builds successfully
    // - operator count matches
    // - output tensor exists after shape inference
}

TEST(runtime_graph_test, BuildRuntimeGraphFromStaticGraph) {
    // Build static graph first, lower it, then assert runtime graph executes.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests --gtest_filter=static_graph_test.*:runtime_graph_test.BuildRuntimeGraphFromStaticGraph`
Expected: FAIL because `StaticGraph` and lowering APIs do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class StaticGraph;
class GraphLowering;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests --gtest_filter=static_graph_test.*:runtime_graph_test.BuildRuntimeGraphFromStaticGraph`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/static_graph_test.cc test/graph_test.cc test/graph_pipeline_test.cc
git commit -m "test: lock static and runtime graph pipeline"
```

### Task 2: Add StaticGraph And Operator Registry

**Files:**
- Create: `include/core/static_graph.h`
- Create: `include/core/operator_registry.h`
- Create: `src/core/static_graph.cc`
- Create: `src/core/operator_registry.cc`
- Modify: `src/operator/fc_op.cc`
- Modify: `src/operator/conv2d_op.cc`
- Modify: `src/operator/relu_op.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(static_graph_test, RejectsUnknownOperatorType) {
    // Unknown op type should fail in registry lookup.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests --gtest_filter=static_graph_test.RejectsUnknownOperatorType`
Expected: FAIL because registry lookup is missing.

- [ ] **Step 3: Write minimal implementation**

```cpp
class OperatorRegistry {
   public:
    using Builder = std::function<std::shared_ptr<OpBase>(
        const model::NodeDesc&,
        std::unordered_map<std::string, std::shared_ptr<Tensor>>&)>; 
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests --gtest_filter=static_graph_test.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/static_graph.h include/core/operator_registry.h src/core/static_graph.cc src/core/operator_registry.cc src/operator/fc_op.cc src/operator/conv2d_op.cc src/operator/relu_op.cc
git commit -m "feat: add static graph and operator registry"
```

### Task 3: Lower StaticGraph Into RuntimeGraph

**Files:**
- Create: `include/core/graph_lowering.h`
- Modify: `include/core/graph.h`
- Modify: `src/core/graph.cc`
- Create: `src/core/graph_lowering.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(runtime_graph_test, RuntimeGraphStoresExecutableNodesOnly) {
    // Build from static graph and assert runtime graph size matches lowered nodes.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests --gtest_filter=runtime_graph_test.RuntimeGraphStoresExecutableNodesOnly`
Expected: FAIL because lowering and runtime node APIs do not exist.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct RuntimeNode {
    std::string name;
    std::string op_type;
    std::shared_ptr<OpBase> binding_owner;
    std::unique_ptr<KernelBase> kernel;
    int32_t Run();
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests --gtest_filter=runtime_graph_test.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph_lowering.h include/core/graph.h src/core/graph.cc src/core/graph_lowering.cc
git commit -m "feat: lower static graph into runtime graph"
```

### Task 4: Update Docs And Verify Whole Tree

**Files:**
- Modify: `docs/project_overview.md`
- Modify: `docs/development_roadmap.md`

- [ ] **Step 1: Write the failing check**

```text
Search docs for claims that RuntimeGraph constructs ops directly or binds kernels by branching.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n "RuntimeGraph.*bind|build_graph\\(\\).*create|if/else" docs/project_overview.md docs/development_roadmap.md`
Expected: Matches old wording.

- [ ] **Step 3: Write minimal implementation**

```md
Update docs to describe:
- StaticGraph semantic stage
- operator registry creation
- runtime graph lowering
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add docs/project_overview.md docs/development_roadmap.md
git commit -m "docs: document static to runtime graph architecture"
```
