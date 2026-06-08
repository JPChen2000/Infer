# X86 CPU Optimization Design

## Background

The current inference stack already has the right high-level structure for CPU optimization:

- `StaticGraph` builds operators and tensors
- `GraphLowering` materializes a `RuntimeGraph`
- `RuntimeGraph` executes registered kernels on the host device
- the host device resolves to `DeviceType::X86` on x86 machines

This gives the project a clean place to optimize without changing model import or graph semantics. The remaining problem is that the hottest x86 kernels are still closer to "correct AVX2 implementations" than "fully optimized CPU kernels":

- `Conv2D` has AVX2 code paths, but still rebuilds per-output patches and maintains a second thread pool inside the kernel
- `Gemm` and `MatMul` transpose the RHS tensor inside every `compute()` call and do not use blocking, packing, or multi-threading
- `FC` currently falls back to a common scalar implementation
- `Softmax` currently falls back to a common scalar implementation
- graph-level parallel execution and kernel-level parallel execution can oversubscribe CPU threads

## Goals

1. Make x86 CPU performance measurable and repeatable before changing kernels.
2. Add enough profiling to identify per-op hot spots in real runs instead of guessing.
3. Remove avoidable thread oversubscription between `RuntimeGraph` and heavy x86 kernels.
4. Stage the remaining optimization work so we can land value incrementally:
   - phase 1: benchmark + profiling + thread model cleanup
   - phase 2: `Gemm` / `MatMul` / `FC`
   - phase 3: `Conv2D` FP32 fast path
   - phase 4: x86 kernel coverage for missing hotspots
   - phase 5: graph fusion and build-time tuning

## Non-Goals

- CUDA optimization
- ONNX importer redesign
- quantization in the first optimization wave
- changing graph semantics or operator behavior
- broad refactors unrelated to measured CPU bottlenecks

## Current Bottlenecks

### 1. Kernel math is only partially optimized

`src/kernel/x86/gemm.cc` and `src/kernel/x86/matmul.cc` use AVX2 FMA but still:

- allocate and transpose `rhs_t` on every invocation
- execute naive `m * n * k` loops
- do not block for L1/L2 cache
- do not pack weights into a reusable layout
- do not use kernel-level threading

`src/kernel/common/fc.cc` is still scalar and should be treated as a priority hotspot once phase 1 instrumentation is in place.

### 2. Conv2D spends too much time building temporary patches

The current x86 `Conv2D` implementation improves arithmetic throughput, but it still rebuilds temporary vectors inside inner loops for pointwise and direct convolutions. That pattern is simple and correct, but it burns memory bandwidth and cache capacity before the CPU can fully benefit from AVX2.

### 3. Threading is split across graph and kernel layers

`RuntimeGraph` creates a thread pool sized by hardware concurrency. `src/kernel/x86/conv2d.cc` also owns a static thread pool. On CNN-style workloads this can create nested parallelism and oversubscription, especially once more kernels become internally parallel.

### 4. Performance visibility is too coarse

`Yolov5Runner` exposes coarse timing buckets like `preprocess_ms`, `rungraph_ms`, and `postprocess_ms`, which is useful, but not enough to rank the hottest operators or compare threading strategies.

## Optimization Strategy

## Phase 1: Benchmark, Per-Op Profiling, And Unified Threading

### Benchmarking

Add a repeatable x86 benchmark path that can:

- load a known model
- run a warmup phase
- execute multiple timed iterations
- print aggregate latency statistics
- optionally print per-op timing totals

This benchmark should use the existing runtime path instead of introducing a second execution path. The benchmark is the acceptance gate for later optimization phases.

### Profiling

Add lightweight per-op timing collection around runtime node execution. The runtime should accumulate:

- op type
- node name
- call count
- total time
- average time

This should be available behind a runtime option or debug flag so normal use stays simple.

### Threading

Unify the thread model so one level of the runtime owns parallelism. For the first phase, the safest choice is:

- keep graph execution serial by default for CNN inference
- allow heavy x86 kernels to opt into internal parallelism
- make thread count configurable from one place

This avoids nested pools while preserving a clear path for future GEMM/Conv parallel kernels.

## Phase 2: Gemm / MatMul / FC

After phase 1 identifies exact hotspots, optimize dense linear algebra by:

- moving reusable RHS/weight packing out of the hottest loop
- adding blocked kernels
- adding x86 `FC` support instead of falling back to `COMMON`
- optionally using a backend such as oneDNN or OpenBLAS if integration cost is justified

This phase is expected to provide the fastest return after phase 1.

## Phase 3: Conv2D FP32 Fast Path

Refactor x86 `Conv2D` to reduce temporary patch construction and improve cache locality:

- specialize `1x1` convolution
- pack weights for output-channel tiles
- use micro-kernels such as `oc8` / `oc16`
- avoid per-output dynamic allocations

The current FP16 path should not be treated as the default acceleration strategy on AVX2-only CPUs because it mostly converts half values to float and back.

## Phase 4: Missing X86 Kernel Coverage

Fill common fallbacks that can still show up in real graphs:

- `FC`
- `Softmax`
- grouped or depthwise convolution variants if benchmarks show them

The goal is to reduce accidental fallback to scalar `COMMON` code for hot paths.

## Phase 5: Graph And Build Optimizations

Once kernel performance is stable, optimize above the kernel layer:

- add selected graph fusion such as `Conv + Bias + ReLU`
- add simple constant folding or transpose elimination when it helps CPU execution
- test compiler improvements such as `LTO`, `PGO`, and optional ISA-specific builds

## File Impact

Phase 1 is expected to touch:

- `src/core/graph.cc`
- `include/core/graph.h`
- `src/demo/yolov5_runner.cc`
- `include/demo/yolov5_runner.h`
- `include/util/timer.h` or a new profiling helper if needed
- `src/kernel/x86/conv2d.cc`
- `CMakeLists.txt`
- new benchmark and test files under `demo/` or `test/`

Later phases will additionally touch:

- `src/kernel/x86/gemm.cc`
- `src/kernel/x86/matmul.cc`
- `src/kernel/common/fc.cc`
- new x86 `FC` / `Softmax` kernels

## Testing Strategy

Phase 1 should add:

- unit coverage for profiling aggregation helpers
- runtime graph tests that verify the chosen thread mode still produces correct results
- a benchmark command that is easy to run locally on x86

All later optimization phases must keep correctness coverage green before using benchmark wins as evidence.

## Acceptance Criteria

The x86 optimization effort is on track when phase 1 delivers all of the following:

1. A repeatable benchmark command for CPU runs.
2. Per-op latency summaries that identify the dominant runtime nodes.
3. A single, documented threading policy that avoids nested graph/kernel oversubscription by default.
4. Existing unit tests still pass.
5. Later kernel work can be prioritized using measured data instead of intuition.
