# Static Graph To Runtime Graph Design

## Background

The current `RuntimeGraph::build_graph()` directly parses `ModelDesc`, constructs concrete ops, selects kernels, and stores executable units in one place. That couples model import, operator semantics, kernel binding, and runtime execution into a single class and scales poorly as operator count grows.

The agreed direction is:

1. Weight/model import builds a `StaticGraph`.
2. `StaticGraph` holds semantic `Op` nodes and graph-level tensor/value metadata.
3. Kernel binding happens during static graph construction through operator-owned registration logic, not in `RuntimeGraph`.
4. Static-graph optimization and operator fusion happen before runtime lowering.
5. Lowering emits a `RuntimeGraph` that is an executable DAG of runtime kernel nodes.

## Goals

- Remove op-type `if/else` branching from `RuntimeGraph`.
- Introduce an explicit `ModelDesc -> StaticGraph -> RuntimeGraph -> run()` pipeline.
- Keep current `FC / Conv2D / ReLU` behavior working.
- Preserve current tensor-based execution semantics while creating room for later graph optimization and fusion.

## Non-Goals

- Full graph-pass framework implementation.
- Multi-device scheduling.
- CUDA runtime lowering.
- Dynamic shape support.

## Architecture

### StaticGraph

`StaticGraph` owns:

- model metadata
- graph input/output names
- tensor/value map
- semantic operator list (`std::shared_ptr<OpBase>`)

Responsibilities:

- materialize tensors declared by `ModelDesc`
- validate required graph inputs/constants
- create concrete operators via registry
- let each operator registration path perform shape inference and kernel binding
- expose a future `ApplyPasses()` hook for graph optimization/fusion

### Operator Registry

Add an `OperatorRegistry` that maps `op_type -> builder`.

Each concrete operator source file registers one builder that knows how to:

- read `NodeDesc`
- resolve tensors from the tensor store
- construct the correct param object
- instantiate the concrete `Op`
- validate shape and infer outputs
- register/select/bind the matching kernel

This keeps operator-specific construction logic near the operator implementation and removes central branching.

### GraphLowering

`GraphLowering` converts `StaticGraph` operators into runtime executable nodes.

At this stage we keep the semantic `Op` alive only as binding/lifetime context for kernel parameters. Runtime nodes become the unit that `RuntimeGraph` executes, and `RuntimeGraph` no longer needs concrete operator headers or model parsing logic.

### RuntimeGraph

`RuntimeGraph` owns executable runtime nodes and the graph tensor map.

Responsibilities:

- store lowered runtime nodes in execution order
- validate node existence
- run each node
- expose graph tensors for tests and callers

It does not:

- parse `ModelDesc`
- know concrete operator subclasses
- bind kernels by op type

## Data Flow

1. Caller provides `ModelDesc` and tensors.
2. `StaticGraph::Build()` materializes values and creates bound semantic ops through `OperatorRegistry`.
3. `StaticGraph::ApplyPasses()` is the placeholder optimization boundary.
4. `GraphLowering::Lower()` creates runtime executable nodes from static ops.
5. `RuntimeGraph::Run()` executes runtime nodes in order.

## Testing Strategy

- Add a `StaticGraph` test that builds an FC graph from `ModelDesc`.
- Add a lowering test that proves runtime graph comes from `StaticGraph`.
- Update pipeline tests so runtime execution goes through `StaticGraph -> GraphLowering -> RuntimeGraph`.
- Keep the standalone op test that verifies unbound ops cannot run.

## Acceptance Criteria

- No operator-type branching remains in `RuntimeGraph`.
- `FC`, `Conv2D`, and `ReLU` are created via registry-driven operator builders.
- A model can be executed through the two-stage graph pipeline.
- Existing numerical tests remain green after the refactor.
