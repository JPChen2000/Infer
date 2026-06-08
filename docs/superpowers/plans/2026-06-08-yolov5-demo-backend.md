# Yolov5 Demo Backend Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--backend <common|x86>` to `yolov5_demo`.

**Architecture:** Store the selected kernel device on `StaticGraph`, apply it while operators attach kernels, and expose a small backend parser through `Yolov5Runner`. The demo CLI passes the parsed backend into `Yolov5Runner::Load()`.

**Tech Stack:** C++17, CMake, GoogleTest.

---

### Task 1: Failing Tests

**Files:**
- Modify: `test/graph_test.cc`
- Modify: `test/yolov5_demo_test.cc`

- [x] Add a `StaticGraph` test that sets `DeviceType::COMMON` and `DeviceType::X86`, lowers an `Add` node, and checks the concrete runtime kernel type.
- [x] Add parser tests for `common`, `x86`, and invalid backend strings.
- [x] Extend the existing yolov5 runner test to load with `Yolov5Backend::kX86` and require `backend=x86` in the build summary.
- [x] Run `cmake --build build --target unit_tests` and expect compile/test failure because the backend API is not implemented yet.

### Task 2: Core Backend Selection

**Files:**
- Modify: `include/core/operator_registry.h`
- Modify: `include/core/static_graph.h`
- Modify: `src/core/static_graph.cc`

- [x] Add an active kernel device scope used by `CreateHostKernelForTensor()`.
- [x] Add `StaticGraph::SetKernelDevice(DeviceType)` and `StaticGraph::KernelDevice()`.
- [x] Apply the selected device while building and rebuilding static graph nodes.

### Task 3: Yolov5 Runner and CLI

**Files:**
- Modify: `include/demo/yolov5_runner.h`
- Modify: `src/demo/yolov5_runner.cc`
- Modify: `demo/yolov5_infer_demo.cc`

- [x] Add `Yolov5Backend`, `ParseYolov5Backend()`, and `Yolov5BackendName()`.
- [x] Change `Yolov5Runner::Load()` to accept the backend with a host-default value.
- [x] Include the resolved backend name in build/run summaries.
- [x] Parse `--backend <common|x86>` in `yolov5_demo`.

### Task 4: Verification

**Files:**
- No production edits.

- [x] Run `cmake --build build --target unit_tests`.
- [x] Run `./build/bin/unit_tests --gtest_filter='static_graph_test.KernelDeviceSelects*Backend:yolov5_demo_test.ParseYolov5Backend*:yolov5_demo_test.LoadFthAndRunImageInference'`.
- [x] Run `cmake --build build --target yolov5_demo`.
- [x] Run `./build/bin/yolov5_demo --backend invalid` and confirm it exits nonzero with usage.
- [x] Run real `yolov5_demo` commands with `--backend common` and `--backend x86` and confirm both summaries show the selected backend.
