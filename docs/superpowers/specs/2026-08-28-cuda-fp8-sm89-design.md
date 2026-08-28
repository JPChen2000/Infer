# CUDA FP8 SM89 Acceleration Design

## Goal

Make the CUDA FP8 E4M3 and E5M2 execution path for Qwen use Ada Tensor
Cores on the local RTX 4070 (SM89), while preserving the model's existing
FP8 storage and numerical contract. The steady-state decode path must avoid
the scalar FP32 tiled kernels and avoid copying a full vocabulary logits tensor
to the host.

## Environment and Constraints

- Deployment target: NVIDIA GeForce RTX 4070, compute capability 8.9.
- Driver: 596.49. The driver remains installed and is not modified.
- Build toolchain: install CUDA Toolkit 12.8 alongside CUDA 11.8 under
  `/usr/local/cuda-12.8`; the project configuration selects 12.8 explicitly.
- cuDNN: install a CUDA 12 compatible cuDNN package alongside the toolkit.
  Existing CUDA 11.8/cuDNN files are not reused by a CUDA 12 build.
- The model file format, `DataType::FP8E4M3`, `DataType::FP8E5M2`, and their
  per-tensor quantization scales remain unchanged.
- The Common and X86 implementations remain numerical references and fallbacks
  for their own backends.
- No silent CPU fallback is permitted for a CUDA FP8 linear operation.

## Evidence Driving the Design

The current CUDA build emits `sm_52` code even though the target GPU is SM89.
More importantly, FP8 `MatMul`, `Gemm`, and `FC` call a hand-written 16x16
kernel that decodes each FP8 element to float, accumulates scalar float
products, and re-encodes each output. BF16 instead calls cuBLAS Tensor Core
GEMM.

On the same Qwen decode workload, normalized runtime profile totals are:

| Operation | BF16 | FP8 E4M3 |
| --- | ---: | ---: |
| MatMul | 6.21 ms/token | 19.09 ms/token |
| Cast | 4.94 ms/token | 12.95 ms/token |
| lm-head path | 1.06 ms/token | 3.69 ms/token before host argmax |
| Runtime nodes | 1470 | 1893 |

cuDNN 8.9.7 is already enabled in the existing CUDA 11.8 build. It accelerates
supported convolution, pooling, and activation kernels, but it is not the
library used for Qwen's linear-layer bottleneck.

## Scope

### Included

1. Install and select CUDA Toolkit 12.8 and matching cuDNN without replacing
   the existing CUDA 11.8 installation or NVIDIA driver.
2. Build CUDA code for SM89 by default and permit an explicit caller override
   for another architecture.
3. Link and initialize cuBLASLt on the inference stream.
4. Use cuBLASLt native FP8 Tensor Core matmul for supported FP8 `MatMul`,
   `Gemm`, and `FC` shapes.
5. Cache cuBLASLt heuristic results, workspace, and device-side scale values
   outside the per-token hot path.
6. Add a CUDA FP8 Qwen lm-head plus greedy argmax kernel so one INT64 token,
   rather than complete logits, crosses the device-host boundary.
7. Preserve the existing cuDNN acceleration paths and make their CUDA 12
   discovery/linkage unambiguous.
8. Add regression tests, backend diagnostics, and a repeatable Qwen profile
   comparison.

### Deferred Until a Post-Change Profile

- CUDA Graph capture for the full decode graph.
- FP8 cuDNN frontend convolution support.
- Broad graph-wide fusion beyond the final lm-head.
- Rewriting every FP8 elementwise kernel.

These are candidate follow-up items, not part of the first correctness and
performance milestone. They proceed only when the new profile shows a material
remaining contribution.

## Toolchain Design

CUDA 12.8 and matching cuDNN are installed side by side with CUDA 11.8. The
build command explicitly supplies both of these settings:

```sh
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
-DCUDAToolkit_ROOT=/usr/local/cuda-12.8
```

The project adds a cache variable `FEATHER_CUDA_ARCHITECTURES`, defaulting to
`89`. A fresh configure initializes `CMAKE_CUDA_ARCHITECTURES` from that
variable only when the caller has not explicitly selected an architecture.
Existing build directories are reconfigured with
`-DCMAKE_CUDA_ARCHITECTURES=89`; an old cached `52` value is never silently
retained for the target build.

CMake locates cuDNN relative to the selected `CUDAToolkit_ROOT` before generic
`/usr/local/cuda` and system paths. It links `CUDA::cublasLt` in addition to
`CUDA::cublas` and defines `FEATHER_WITH_CUBLASLT` only when that imported
target is available. CUDA 12 is required for the native FP8 implementation;
the rest of the project can still configure without that feature when CUDA is
disabled.

Configuration output records the CUDA compiler version, selected toolkit root,
cuDNN library path, and CUDA architecture list. This makes accidental linkage
to CUDA 11.8 visible in a build log.

## FP8 Linear-Layer Design

### Capability Gate

Native FP8 dispatch is available only when all of the following are true:

1. The build has `FEATHER_WITH_CUBLASLT`.
2. The runtime device reports compute capability 8.9 or later.
3. The source FP8 format passes a 256-value byte/code-point compatibility test
   against the CUDA native E4M3 or E5M2 format.
4. cuBLASLt returns a valid heuristic for the exact matrix, layout, dtype, and
   epilogue combination.

The compatibility test is mandatory because `Fp8E4M3` and `Fp8E5M2` are
project-defined encoded byte formats. Native raw-byte use is enabled only if
their decoding and encoding agree with NVIDIA's format at every finite value,
zero, infinity, and NaN code point relevant to the type. If a format does not
match, the CUDA path uses a dedicated device conversion kernel before and after
the native operation rather than reinterpret-casting incompatible bytes.

### cuBLASLt Operation

The new launcher receives the existing raw FP8 device tensors, dimensions,
transpose flag, optional bias, `alpha`, `beta`, and the three existing tensor
quantization scales. It creates a cuBLASLt matmul descriptor with:

- `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2` matrix layouts for native-compatible
  FP8 storage;
- FP32 accumulation and the required FP8 output type;
- device-resident A, B, and D scale pointers;
- row-major-to-column-major operand mapping that preserves the repository's
  existing `trans_b` contract;
- a reusable device workspace and one selected heuristic algorithm per exact
  operation key.

The plan key contains source dtype, output dtype, M, N, K, transpose flags,
layout, bias mode, and epilogue. It intentionally excludes raw tensor addresses
and scale values so immutable model weights and repeated decode shapes reuse the
same selected algorithm. A separate scale cache is keyed by tensor identity and
mutation version; its device scalar is refreshed only when the source scale
changes.

Bias and alpha/beta behavior must remain bitwise compatible with the current
operator contract after its documented FP8 rounding point. When a cuBLASLt
epilogue cannot represent that exact contract, the launcher computes the
matmul with cuBLASLt and invokes a small existing-style CUDA post-operation
kernel. It must not return to the scalar tiled GEMM just because a bias is
present.

### Fallback Behavior

If the native compatibility or heuristic gate fails, the launcher records a
specific backend reason. It uses a GPU-only fallback: convert the required
operands to BF16 on the device and invoke the existing BF16 cuBLAS Tensor Core
path, then quantize the output according to the existing FP8 output scale.
The fallback may cache converted immutable weights by tensor identity/version,
but it must not cache mutable activations or alter their lifetime.

The original scalar FP8 CUDA implementation remains an explicit last-resort
correctness fallback for builds without cuBLASLt. The SM89/CUDA 12.8 Qwen
benchmark must report either a native cuBLASLt or BF16 Tensor Core backend;
using the scalar fallback is a failed performance acceptance result.

## Qwen FP8 lm-head and Argmax

The existing CUDA `QwenGemmArgmax` kernel accepts BF16 only. A pair of FP8
specializations is added for E4M3 and E5M2.

For the fused FP8 path:

1. The lm-head projection uses the FP8 cuBLASLt launcher or the GPU BF16 Tensor
   Core fallback.
2. The temporary logits retain the same effective FP8 output scale and rounding
   behavior as the current `FP8 Gemm -> Cast(BF16)` graph.
3. A CUDA reduction decodes/logically compares these logits, skips NaNs, keeps
   the first index for ties including signed zero, and writes one INT64 token.
4. The runner synchronizes only that one token.

After registration and numerical tests exist, the Qwen fusion pass fuses the
CUDA FP8 lm-head cast pattern into `QwenGemmArgmax`. The current CUDA FP8 guard
is retained until that exact point; it prevents a graph from lowering to a
kernel that does not exist.

## cuDNN Use

cuDNN remains enabled for its existing supported CUDA operations. CMake ensures
the CUDA 12 cuDNN headers and shared objects are paired with the CUDA 12.8
toolkit. The implementation adds a focused test that confirms supported
Conv2D, pool, and SiLU paths can select their cuDNN backend after the upgrade.

No FP8 Qwen linear layer is routed through cuDNN. No new FP8 convolution
frontend integration is added in this milestone because the current Qwen FP8
profile shows convolution as a small fraction of token latency. If its share is
material after the linear path is fixed, a separate profile-backed design will
choose between cuDNN frontend FP8 convolution and a specialized depthwise
kernel.

## Error Handling and Observability

- Runtime initialization reports a descriptive failure when cuBLASLt cannot be
  created or cannot be bound to the inference stream.
- The FP8 launcher exposes the last selected backend: native cuBLASLt, BF16
  Tensor Core fallback, or scalar fallback, plus a failure reason for the last
  unavailable native plan.
- Unsupported shapes remain correct through the defined GPU fallback and do not
  silently transfer their tensors to the host.
- A debug/profile build logs a one-time selected algorithm and workspace size
  per plan key, not one line per decode token.

## Testing

1. CMake configuration test: configure against CUDA 12.8 with SM89 and assert
   the generated CUDA flags contain `sm_89`, `FEATHER_WITH_CUBLASLT`, and the
   CUDA 12 cuDNN library path.
2. FP8 encoding test: compare all 256 project E4M3/E5M2 codes against the
   native CUDA format conversion behavior and select direct or conversion mode
   deterministically.
3. FP8 linear tests: compare `MatMul`, `Gemm`, and `FC` output bit patterns to
   the Common reference across E4M3/E5M2, scale values, tails, `trans_b`,
   optional bias, and alpha/beta behavior.
4. Backend-selection test: on an SM89 CUDA 12 device, assert a representative
   Qwen-shaped FP8 linear operation selects cuBLASLt. On unsupported devices or
   absent hardware, skip the assertion with a clear reason rather than passing
   under the scalar backend.
5. FP8 argmax tests: compare CUDA fused result with the existing unfused
   `Gemm + Cast + host greedy` reference for ordinary values, ties, signed
   zeroes, NaNs, E4M3, and E5M2.
6. Existing CUDA cuDNN tests: verify the selected CUDA 12 cuDNN library still
   executes the supported convolution, pooling, and activation paths.
7. End-to-end Qwen check: use a fixed token sequence to compare generated
   tokens before and after the change, then run the same fixed prompt/profile
   command used for the baseline. Report calls, aggregate operator time, and
   per-token timing separately from model load and warmup.

## Acceptance Criteria

- The CUDA build uses Toolkit 12.8, a compatible cuDNN library, and SM89 code.
- Representative Qwen FP8 linear operations select cuBLASLt native FP8 Tensor
  Core execution on the RTX 4070; if a shape cannot, it selects GPU BF16 Tensor
  Core fallback rather than scalar FP32.
- CUDA FP8 Qwen lm-head produces the same greedy token as the prior unfused
  graph and copies only one INT64 result to the host.
- Focused unit tests and the full unit test binary pass.
- On the fixed Qwen profile, FP8 `MatMul` aggregate time and full decode latency
  improve from the current baseline. BF16 correctness and latency do not
  regress.
- The final profile identifies the next bottleneck before CUDA Graph or
  additional fusion work is started.

## Non-goals

- Replacing the model format with a TensorRT engine.
- Removing CUDA 11.8 from the host.
- Changing the NVIDIA display/compute driver.
- Treating a cuDNN-linked build as proof that Qwen GEMM is optimized.
- Claiming a fixed tokens-per-second target before the post-change benchmark.
