# Pass Reorg And Fusion Design

## Goal

Reorganize graph passes into a dedicated `pass` area, remove pass headers from `include/core`, and add a small set of reusable fusion passes that operate on existing operators rather than introducing new framework abstractions.

## Scope

- Move pass declarations/definitions out of `core` paths into `pass` paths.
- Keep `StaticGraph` depending only on the pass interface.
- Split passes into:
  - general-purpose graph passes
  - model-specific passes
- Add reusable fusion passes based on currently supported operators.

## Directory Layout

- `include/pass/graph_pass.h`
- `include/pass/dead_node_elimination_pass.h`
- `include/pass/sigmoid_mul_fusion_pass.h`
- `include/pass/identity_elimination_pass.h`
- `include/pass/matmul_add_fusion_pass.h`
- `include/pass/yolo_decode_fusion_pass.h`
- `include/pass/register_builtin_passes.h`
- `src/pass/graph_pass.cc`
- `src/pass/dead_node_elimination_pass.cc`
- `src/pass/sigmoid_mul_fusion_pass.cc`
- `src/pass/identity_elimination_pass.cc`
- `src/pass/matmul_add_fusion_pass.cc`
- `src/pass/yolo_decode_fusion_pass.cc`
- `src/pass/register_builtin_passes.cc`

`StaticGraph` continues to own only `PassManager`.

## Pass Layers

### General-purpose

- `DeadNodeEliminationPass`
- `IdentityEliminationPass`
- `SigmoidMulFusionPass`
- `MatMulAddFusionPass`

### Model-specific

- `YoloDecodeFusionPass`

## Default Entry Points

Provide two simple factories:

- `CreateDefaultPassManager()`
- `CreateYoloPassManager()`

Default order:

1. `SigmoidMulFusionPass`
2. `MatMulAddFusionPass`
3. `IdentityEliminationPass`
4. `DeadNodeEliminationPass`

YOLO order:

1. Default passes
2. `YoloDecodeFusionPass`
3. `DeadNodeEliminationPass`

The final dead-node cleanup keeps graph state tidy after fusion.

## Fusion Rules

### IdentityEliminationPass

Pattern:

- `Identity(x) -> y`

Rewrite:

- redirect all users of `y` to `x`
- if `y` is graph output, do not rewrite
- remove the identity node once no users remain

### SigmoidMulFusionPass

Keep existing logic:

- `Sigmoid(x)` feeding `Mul(x, sigmoid(x))`
- rewrite to `SiLU(x)`

### MatMulAddFusionPass

Pattern:

- `MatMul(a, b) -> t`
- `Add(t, bias) -> out`

Constraints:

- `t` is not a graph output
- `t` has exactly one user, the `Add`
- the non-`t` input satisfies current `Gemm` bias rules:
  - 1D bias with length `n`
  - or 2D bias with shape `m x n`

Rewrite:

- replace the `Add` node with `Gemm(a, b, bias)`
- remove the `MatMul` node

This reuses the existing `Gemm` op and kernels instead of inventing a fused operator.

## Graph API Changes

Keep API growth minimal. Add one helper only if it materially simplifies multiple passes:

- `ReplaceAllUses(from, to, except_node = "")`

If current APIs are enough, avoid adding more helpers.

## Tests

Extend `test/static_graph_pass_test.cc` with:

- `IdentityEliminationPass` removes interior identity
- identity is preserved when its output is also a graph output
- `MatMulAddFusionPass` rewrites to `Gemm`
- fusion is skipped when matmul output has multiple users
- pass manager factory order behaves as expected
- YOLO factory still applies decode fusion

## Non-Goals

- No new dispatcher or pass registry framework
- No new fused operator definitions beyond existing ops
- No kernel-layer changes in this task
