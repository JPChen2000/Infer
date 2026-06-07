# Common Operators Wave 1 Design

## Background

The graph/runtime skeleton is now in place, but the executable operator set is still too small. Current execution only covers `FC`, `Conv2D`, and `ReLU`, which is not enough to support common CV subgraphs or the foundational dataflow needed before LLM-specific operators.

The next useful step is not more graph passes. It is to expand the common operator set while reusing the new architecture:

`ModelDesc -> StaticGraph(Op) -> GraphLowering -> RuntimeGraph(Kernel)`

## Scope

This wave implements three high-frequency operators first:

1. `MatMul/Gemm`
2. `Sigmoid`
3. `Reshape`

They are chosen because:

- `Gemm/MatMul` is foundational for both CV heads and LLM blocks.
- `Sigmoid` is a common elementwise activation.
- `Reshape` is essential for graph plumbing and tensor layout transitions.

## Goals

- Add operator definitions, kernel registration, and graph integration for `Gemm/MatMul`, `Sigmoid`, and `Reshape`.
- Keep all three on the existing `OperatorRegistry` path.
- Add both unit-level numerical tests and graph-level pipeline tests.

## Non-Goals

- Batched matmul
- Broadcast semantics
- Transposed GEMM variants
- Layout-aware reshape validation beyond element-count preservation
- CUDA implementations

## Design

### Gemm / MatMul

For the current codebase, `Gemm` should be implemented as the first-class graph/operator name because the model layer already uses `"Gemm"` in tests. The implementation can behave as a simple 2D matrix multiply with optional bias:

- inputs: `A`, `B`, optional `bias`
- output: `C`
- shape rule: `[M, K] x [K, N] -> [M, N]`

This should reuse the same semantic pattern as `FC`, but without treating the operator as a higher-level fully-connected layer. `FC` can remain as a separate op for now; no consolidation is needed in this wave.

### Sigmoid

`Sigmoid` is a unary elementwise op with:

- one input
- one output
- output shape identical to input shape

It should follow the same contract pattern as `ReLU`.

### Reshape

`Reshape` should initially support:

- one data input
- one output
- target shape provided by node attribute

Validation should only enforce:

- same total element count
- non-empty target shape

That is enough for current graph construction and testing.

## File Plan

- `src/operator/gemm_op.h/.cc`
- `src/operator/sigmoid_op.h/.cc`
- `src/operator/reshape_op.h/.cc`
- `src/kernel/gemm.h`
- `src/kernel/x86/gemm.cc`
- `src/kernel/x86/sigmoid.cc`
- `src/kernel/x86/reshape.cc`
- `src/kernel/reshape.h`
- tests for operator-level and graph-level execution

## Testing Strategy

- `Gemm` unit test for known matrix multiply results
- `Sigmoid` unit test against known values
- `Reshape` unit test for shape/data preservation
- Graph-level pipeline test that mixes at least `Gemm -> Sigmoid`
- Graph-level test that validates `Reshape` shape propagation through `StaticGraph` and runtime execution

## Acceptance Criteria

- `OperatorRegistry` can construct `Gemm`, `Sigmoid`, and `Reshape`
- `StaticGraph` can build graphs containing these operators
- `RuntimeGraph` can execute lowered kernels
- all tests remain green
