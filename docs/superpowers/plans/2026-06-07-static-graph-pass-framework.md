# Static Graph Pass Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `StaticGraph` optimization framework skeleton so graph passes, node rewrites, and later operator fusion have a stable place to live.

**Architecture:** Extend `StaticGraph` from a flat op list into a lightweight semantic graph IR that keeps node metadata, input/output names, and use-def relations. Add a pass abstraction plus a pass manager, then expose minimal graph query and rewrite APIs that keep graph state consistent after node removal or replacement.

**Tech Stack:** C++17, CMake, gtest

---

### Task 1: Lock Pass Execution And Graph Metadata In Tests

**Files:**
- Create: `test/static_graph_pass_test.cc`
- Modify: `test/static_graph_test.cc`

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(static_graph_pass_test, BuildCapturesNodeMetadataAndUseDef) {
    // Build Conv -> ReLU graph and assert:
    // - node count is 2
    // - producer of relu input is conv
    // - conv output has relu as a user
}

TEST(static_graph_pass_test, ApplyPassesRunsRegisteredPassesInOrder) {
    // Register two test passes and assert execution order.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.*'`
Expected: FAIL because pass and graph metadata APIs do not exist.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct StaticNode;
class GraphPass;
class PassManager;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/static_graph_pass_test.cc test/static_graph_test.cc
git commit -m "test: lock static graph pass framework behavior"
```

### Task 2: Add Pass Abstraction And Static Graph IR Metadata

**Files:**
- Create: `include/core/graph_pass.h`
- Create: `src/core/graph_pass.cc`
- Modify: `include/core/static_graph.h`
- Modify: `src/core/static_graph.cc`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(static_graph_pass_test, BuildCapturesProducerAndUsersForGraphValues) {
    // Validate value producer and consumer maps after build.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.BuildCapturesProducerAndUsersForGraphValues'`
Expected: FAIL because producer/user metadata is missing.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct StaticNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<OpBase> op;
    bool removed{false};
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.BuildCapturesProducerAndUsersForGraphValues'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/graph_pass.h src/core/graph_pass.cc include/core/static_graph.h src/core/static_graph.cc CMakeLists.txt
git commit -m "feat: add static graph pass and metadata skeleton"
```

### Task 3: Add Graph Rewrite Helpers

**Files:**
- Modify: `include/core/static_graph.h`
- Modify: `src/core/static_graph.cc`
- Modify: `test/static_graph_pass_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(static_graph_pass_test, RemoveNodeUpdatesUseDefAndOperatorView) {
    // Remove relu node and assert:
    // - node count shrinks
    // - output user list is updated
    // - operator list no longer exposes removed node
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.RemoveNodeUpdatesUseDefAndOperatorView'`
Expected: FAIL because graph rewrite helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```cpp
bool RemoveNode(const std::string& node_name);
bool ReplaceInputValue(const std::string& node_name,
                       const std::string& from,
                       const std::string& to);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.RemoveNodeUpdatesUseDefAndOperatorView'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/static_graph.h src/core/static_graph.cc test/static_graph_pass_test.cc
git commit -m "feat: add static graph rewrite helpers"
```

### Task 4: Wire Default Pass Manager Through StaticGraph::ApplyPasses

**Files:**
- Modify: `include/core/static_graph.h`
- Modify: `src/core/static_graph.cc`
- Modify: `test/static_graph_pass_test.cc`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(static_graph_pass_test, ApplyPassesUsesInstalledPassManager) {
    // Install pass manager with a test pass and assert ApplyPasses executes it.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.ApplyPassesUsesInstalledPassManager'`
Expected: FAIL because StaticGraph cannot accept pass managers yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
void SetPassManager(std::shared_ptr<PassManager> pass_manager);
int32_t ApplyPasses();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/bin/unit_tests '--gtest_filter=static_graph_pass_test.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/core/static_graph.h src/core/static_graph.cc test/static_graph_pass_test.cc
git commit -m "feat: wire static graph pass manager"
```

### Task 5: Verify Whole Tree And Update Roadmap Notes

**Files:**
- Modify: `docs/development_roadmap.md`

- [ ] **Step 1: Write the failing check**

```text
Search the roadmap for wording that says graph optimization is still entirely missing.
```

- [ ] **Step 2: Run check to verify it fails**

Run: `rg -n 'ApplyPasses\\(\\).*空|图优化.*空壳|pass framework' docs/development_roadmap.md`
Expected: Matches outdated wording.

- [ ] **Step 3: Write minimal implementation**

```md
Update roadmap to say:
- pass framework skeleton exists
- next work is to add real optimization passes
```

- [ ] **Step 4: Run verification to verify it passes**

Run: `cmake --build build --target unit_tests && ./build/bin/unit_tests`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add docs/development_roadmap.md
git commit -m "docs: record static graph pass framework progress"
```
