# Common Operators Wave 2 Design

## Background

The inference pipeline now supports:

- `FC`
- `Gemm`
- `Conv2D`
- `ReLU`
- `Sigmoid`
- `Reshape`

This is enough to prove the graph/runtime architecture, but it is still missing a full set of common graph-structure and post-processing operators. To make the engine more useful for real CV-style graphs, the next wave should focus on pooling, tensor concatenation/splitting, and a basic normalization/probability operator.

## Scope

This wave implements:

1. `AvgPool`
2. `MaxPool`
3. `Concat`
4. `Split`
5. `Softmax`

All operators stay on the existing path:

`OperatorRegistry -> StaticGraph -> GraphLowering -> RuntimeGraph`

## Goals

- Add semantic ops, x86 FP32 kernels, and graph integration for the five operators above.
- Keep implementations intentionally minimal and aligned with current tensor capability.
- Add both operator-level and graph-level tests for each operator family.

## Non-Goals

- CUDA kernels
- NCHW/NHWC layout-generalized pooling
- multi-axis concat/split
- broadcast-aware softmax
- batched tensor semantics

## Operator Design

### AvgPool / MaxPool

Pool operators should initially support only 2D single-tensor input:

- input shape: `[H, W]`
- output shape determined by:
  - `kernel_h`
  - `kernel_w`
  - `stride_h`
  - `stride_w`
  - `pad_h`
  - `pad_w`

`AvgPool` and `MaxPool` can share the same parameter structure. The first version only needs the standard valid sliding-window implementation over a 2D tensor.

### Concat

`Concat` should initially support:

- multiple input tensors
- one output tensor
- concat axis provided by node attribute

For the first version, constrain usage to tensors with identical rank and identical non-concat dimensions. Because current tests and operators mostly use 2D tensors, 2D concat is enough for now.

### Split

`Split` should initially support:

- one input tensor
- multiple outputs
- split axis provided by node attribute
- split sizes provided by attribute

For the first version, require the split sizes to be explicit so there is no inference ambiguity.

### Softmax

`Softmax` should initially support:

- one input tensor
- one output tensor
- 2D input
- axis = last dimension only

This is enough to cover common classifier-style outputs and keeps the numerical path straightforward.

## File Plan

- `src/operator/pool_op.h/.cc`
- `src/operator/concat_op.h/.cc`
- `src/operator/split_op.h/.cc`
- `src/operator/softmax_op.h/.cc`
- `src/kernel/pool.h`
- `src/kernel/concat.h`
- `src/kernel/split.h`
- `src/kernel/softmax.h`
- `src/kernel/x86/pool.cc`
- `src/kernel/x86/concat.cc`
- `src/kernel/x86/split.cc`
- `src/kernel/x86/softmax.cc`

## Testing Strategy

- operator-level numerical tests for each family
- graph-level tests:
  - `AvgPool -> ReLU`
  - `Concat -> Split`
  - `Gemm -> Softmax`

The graph-level tests matter because these operators stress different graph topologies than the first wave:

- pooling changes shape
- concat merges producer branches
- split fans one producer into multiple consumers
- softmax exercises row-wise post-processing

## Acceptance Criteria

- Each operator is constructible from `ModelDesc` via `OperatorRegistry`
- `StaticGraph` can build graphs containing these operators
- `RuntimeGraph` can execute lowered kernels
- all tests remain green
