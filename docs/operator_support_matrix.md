# 算子支持表

## 统计口径

- 统计时间：2026-06-08
- 算子范围：`src/operator/*_op.cc` 中当前已接入框架的算子
- 后端支持判定：以 `KernelDispatcher::registerKernel(...)` 的实际注册结果为准
- `Arm` 列合并表示 `ARM32` 和 `ARM64`；当前两个后端都没有已注册 kernel
- `CUDA` 列只统计已注册到框架分发链路的能力；当前仓库里虽然存在 `src/kernel/cuda/fc_compute.cc`，但没有对应 dispatcher 注册，因此本表记为未支持

## 数据类型说明

- 框架当前注册的数据类型来自 `include/util/types.h` 中的 `DataType`
- `UINT16` 不是独立的框架 `DataType`
- 当前代码里 `uint16_t` 通过 `DataTypeTrait<uint16_t>` 映射到 `DataType::FP16`，因此它只作为 `FP16` 的存储表示出现，不属于可单独调度的 `UINT16` 类型

## 算子支持矩阵

说明：

- `Y` 表示该算子至少在一个已注册后端上支持该数据类型
- 后端列展示的是该算子在对应后端上已注册的数据类型集合
- `-` 表示当前没有已注册支持

| Operator | INT4 | INT8 | UINT8 | UINT16 | FP16 | FP32 | Common | x86 | Arm | CUDA | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Concat | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Conv2D | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| FC | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Flatten | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Gemm | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Identity | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| MatMul | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Mul | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| AvgPool | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| MaxPool | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Pow | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| ReLU | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Reshape | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Resize | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Sigmoid | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Slice | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Softmax | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Split | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |
| Transpose | - | - | - | - | Y | Y | FP16, FP32 | FP16, FP32 | - | - |  |

## 汇总统计

### 按数据类型统计

| DataType | 支持算子数 | 占比 |
| --- | --- | --- |
| INT4 | 0 / 20 | 0% |
| INT8 | 0 / 20 | 0% |
| UINT8 | 0 / 20 | 0% |
| UINT16 | 0 / 20 | 0% |
| FP16 | 20 / 20 | 100% |
| FP32 | 20 / 20 | 100% |

### 按后端统计

| Backend | 已注册算子数 | FP16 算子数 | FP32 算子数 | 备注 |
| --- | --- | --- | --- | --- |
| Common | 20 / 20 | 20 | 20 | 通用后端已补齐 FP16/FP32 |
| x86 | 20 / 20 | 20 | 20 | x86 后端已补齐 FP16/FP32 |
| Arm | 0 / 20 | 0 | 0 | `ARM32`、`ARM64` 均无已注册 kernel |
| CUDA | 0 / 20 | 0 | 0 | 存在局部源码，但没有 dispatcher 注册 |

## 后续开发建议

- `Common` 与 `x86` 的 `FP16/FP32` 基线已经补齐，后续可以把重心转向性能优化而不是算子覆盖率
- 量化方向如果要推进 `INT4`、`INT8`、`UINT8`，需要先补齐框架级 kernel 注册与张量/算子链路支持
- 如果后续要把 `UINT16` 作为独立类型推进，需要先扩展 `DataType` 枚举和相关分发逻辑
