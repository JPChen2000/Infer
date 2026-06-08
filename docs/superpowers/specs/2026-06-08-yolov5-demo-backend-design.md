# Yolov5 Demo Backend Selection Design

## Goal

Allow `yolov5_demo` to choose `common` or `x86` kernels from the command line while keeping the current host-default behavior when no backend is specified.

## Design

`StaticGraph` owns the kernel device used while building operators. It defaults to `GetHostRuntimeDevice()`, preserving current behavior. `Yolov5Runner::Load()` accepts a demo backend enum and maps it to `DeviceType::COMMON`, `DeviceType::X86`, or the host default.

The CLI adds `--backend <common|x86>`. Invalid backend strings fail argument parsing and print usage. Build summaries include the resolved backend name so benchmark/demo output makes the selected path visible.

## Testing

Unit tests cover parser behavior, runner summary output, and direct `StaticGraph` kernel selection by checking the concrete runtime kernel type after lowering.
