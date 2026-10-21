# Runtime ownership and execution contracts

For compiled plans, isolated sessions, deferred storage and backend contracts, see [execution architecture](execution_architecture.md).

## Tensor storage

Tensor aliases retain the same `shared_ptr<Buffer>`. Resetting or clearing a Tensor drops only that Tensor's ownership; the pool reuses a block after the final Buffer owner releases it. `BufferPool::deallocate` resets the supplied handle. `deallocate_all` rejects outstanding allocations. Buffers retain their owning pool, including pools other than the singleton.

Typed access and assignments validate element counts, byte multiplication, offsets and remaining capacity. ResetBuffer validates its range before changing storage. Invalid ranges raise the existing CHECK/logic_error exception. Clearing a Tensor makes IsInitialized false.

Mapped weights expose only their validated payload range. Their declared shape and dtype must match the payload size; INT4 uses two packed elements per byte. `mutable_data` makes a private copy of read-only mapped storage before returning a writable pointer. `raw_data` remains a low-level pointer API; callers must respect storage access and mutation rules.

## Model and graph validation

Model loading bounds metadata to 256 MiB, validates declared sizes before allocating collections, and validates every constant range before committing a newly loaded model. Failed parsing leaves the previous model and weight store in place. Collection counts for graph objects and strings are limited to 1,048,576; numeric vector lengths are bounded by remaining metadata bytes.

Graph construction rejects missing sources, duplicate values or nodes, multiple producers, missing outputs and cycles. Explicit SetTensor bindings can provide external values omitted from graph.inputs. Internally allocated placeholders do not count as external bindings. Anonymous nodes receive unique generated names. A stable topological ordering preserves the order of already sorted models.

After RuntimeGraph::Finalize, SetTensor may rebind compatible external inputs only. It preserves the original Tensor object used by kernel parameters and changes its storage. It rejects unknown names, produced outputs, immutable tensors, incompatible shape/dtype/layout and undersized storage. Dynamic shape changes require rebuilding the graph.

## CUDA sessions

Every RuntimeGraph owns a CudaExecutionContext. Device tensor caches, temporary block pools, stream, library handles and deferred host synchronization state belong to that context. Runners enter their graph's context for loading, inference, graph capture, state operations and teardown. Clearing one graph cannot free another graph's allocations.

Standalone kernel calls use a default context per host thread. Low-level callers inspecting or synchronizing a graph's device tensors must enter that graph's context:

```cpp
kernel::cuda_detail::ExecutionContextScope context(runtime.CudaContext());
{
    kernel::cuda_detail::DeferredHostSyncScope deferred;
    runtime.Run();
    // Explicitly sync externally consumed output tensors here.
}
```

Scopes restore the previous context and, when necessary, CUDA device. CPU graph workers inherit the graph's CUDA context for fallback transfers. A device-side read preserves a stale host copy's invalid state; only an actual transfer can establish consistency. Separate graph instances can use separate contexts concurrently; callers must serialize modifications and inference on the same graph/runner instance.

## CPU portability and builds

Native x86 compute units require AVX2, FMA and F16C. Their registration code is compiled in a separate baseline unit and checks host CPU capabilities before selecting native factories. Unsupported hosts and explicit X86 requests fall back to Common. `FEATHER_DISABLE_X86=1` forces that fallback for a process.

`FEATHER_ENABLE_X86=OFF` excludes native x86 compute sources and native benchmark executables. The default is off for non-x86 targets. Debug/Release follows the caller's CMAKE_BUILD_TYPE; Release is only a default for a single-configuration generator with no requested build type. FEATHER_DEBUG independently controls diagnostic logging.

## Regression coverage

`test/runtime_safety_test.cc` covers live storage aliases, pool lifetime, offsets, overflow, malformed model sizes, weight bounds and copy-on-write, external input rebinding, invalid graphs, unordered DAGs and forced Common fallback. CUDA builds additionally test context/flag isolation without a GPU, and cache isolation plus device-read/CPU-read consistency when a GPU is available. Existing graph cache tests explicitly select the graph context.
