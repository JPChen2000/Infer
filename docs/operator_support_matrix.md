# 算子支持表

## 统计口径

- 统计时间：2026-08-17
- 规范算子范围：`src/operator/*_op.cc` 中当前注册的算子类型；`Conv`、`AveragePool`、`Relu` 是已有规范算子的别名，不重复计数
- 算子总数：46 个。其中 44 个为数值/融合算子，`Shape` 与 `ConstantOfShape` 是控制流/元数据算子
- 后端支持：以 `KernelDispatcher::registerKernel(...)` 的直接注册结果为准；Common fallback 单独在备注中说明
- CUDA 原生实现：请求必须命中 `DeviceType::CUDA` 注册项；缺少 CUDA 设备时，数值测试会跳过，但编译、注册和图 lowering 测试仍可执行
- `ResizeConcat` 只由 `resize_concat_fusion_pass` 生成，属于融合后的内部算子，不是 ONNX 原始算子
- Transformer/LLM 图保持原子算子：`LayerNormalization`、RoPE、attention 和 MLP 均在导出图中展开，不引入聚合的 `Transformer`、`Qwen`、`RMSNorm` 或 attention 算子

## 数据类型说明

- `DataType` 当前包含 `INT4`、`INT8`、`UINT8`、`FP16`、`BF16`、`FP32`、`INT32`、`INT64`、`BOOL` 等类型
- `uint16_t` 是 `FP16` 的主机存储表示；`BFloat16` 是独立的 16 位 BF16 存储类型，两者不能互相重解释
- CUDA `Cast` 原生支持 FP32、FP16、BF16 互转，以及 INT32/INT64 到 FP32/FP16/BF16；其他控制类型转换保留 Common 通用内核
- CUDA BF16 的 MatMul/Gemm/FC 使用 cuBLAS BF16 输入与 FP32 累加；运行这些路径需要设备与 CUDA 库支持 BF16。BF16 Conv2D 固定使用项目的 CUDA fallback kernel，不依赖 cuDNN 的 BF16 支持
- CUDA 图允许缺少原生 CUDA kernel 的控制节点回退到 Common，并由 RuntimeGraph 负责设备边界同步；这不计入 CUDA 原生支持

## 算子支持矩阵

`Y` 表示该类型至少有一个后端的直接注册 kernel；`-` 表示没有直接注册。后端列列出该后端的直接注册浮点类型。

| Operator | INT4 | INT8 | UINT8 | FP16 | FP32 | BF16 | INT32 | INT64 | BOOL | Common | x86 | Arm | CUDA | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| BatchNormalization | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Sub | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Div | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Concat | - | - | - | Y | Y | Y | - | Y | - | FP16, FP32, BF16, INT64 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Conv2D | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | ONNX `Conv` 别名；BF16 CUDA 使用 fallback kernel |
| FC | - | - | - | Y | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32, BF16 | BF16 CUDA 线性层 |
| Flatten | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | 分类模型算子 |
| GlobalAveragePool | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Gemm | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | BF16 使用 FP32 累加 |
| Identity | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| MatMul | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | CUDA 支持高 rank/broadcast attention 形状 |
| Mul | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| AvgPool | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | ONNX `AveragePool` 别名 |
| MaxPool | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Pow | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32 | CUDA BF16 尚未实现 |
| ReLU | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32 | CUDA BF16 尚未实现 |
| Reshape | - | - | - | Y | Y | Y | - | - | Y | FP16, FP32, BF16, BOOL | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Resize | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| ResizeConcat | - | - | - | Y | Y | - | - | - | - | - | - | - | FP16, FP32 | 只由融合 pass 生成，当前没有 Common/x86 实现 |
| SiLU | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32 | `Sigmoid + Mul` 融合目标；CUDA BF16 尚未实现 |
| Sigmoid | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Slice | - | - | - | Y | Y | Y | - | Y | - | FP16, FP32, BF16, INT64 | FP16, FP32 | - | FP16, FP32 | CUDA BF16 尚未实现 |
| Softmax | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen attention 原子算子 |
| Split | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Transpose | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | FP16, FP32 | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Sqrt | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Tanh | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32 | CUDA BF16 尚未实现 |
| Erf | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32 | CUDA BF16 尚未实现 |
| Unsqueeze | - | - | - | Y | Y | Y | - | Y | Y | FP16, FP32, BF16, INT64, BOOL | - | - | FP16, FP32, BF16 | Qwen 原子算子；控制输入可在 Common 处理 |
| Squeeze | - | - | - | Y | Y | Y | - | Y | Y | FP16, FP32, BF16, INT64, BOOL | - | - | FP16, FP32, BF16 | Qwen 原子算子；控制输入可在 Common 处理 |
| Cast | - | - | - | Y | Y | Y | Y | Y | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16, INT32, INT64 | INT32/INT64 到浮点走 CUDA；其他控制转换走 Common |
| ReduceMean | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen 原子算子 |
| ReduceSum | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Gather | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | INT32/INT64 indices |
| Equal | - | - | - | Y | Y | Y | Y | Y | Y | FP16, FP32, BF16, INT32, INT64, BOOL | - | - | FP16, FP32 | 输出为 BOOL；CUDA BF16 尚未实现 |
| Shape | - | - | - | - | - | - | - | Y | - | INT64 | - | - | - | 控制流/元数据算子，按设计保留 Common |
| ConstantOfShape | - | - | - | Y | Y | - | Y | Y | Y | FP16, FP32, INT32, INT64, BOOL | - | - | - | 控制流/元数据算子，按设计保留 Common |
| Expand | - | - | - | Y | Y | Y | Y | Y | Y | FP16, FP32, BF16, INT32, INT64, BOOL | - | - | FP16, FP32, BF16 | Qwen 原子算子；shape 输入为 INT32/INT64 |
| Where | - | - | - | Y | Y | Y | Y | Y | Y | FP16, FP32, BF16, INT32, INT64, BOOL | - | - | FP16, FP32 | CUDA BF16 尚未实现 |
| YoloDecode | - | - | - | Y | Y | - | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | YOLOv5 head decode 融合算子 |
| Exp | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen RoPE 原子算子 |
| Sin | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen RoPE 原子算子 |
| Cos | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen RoPE 原子算子 |
| Neg | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen 原子算子 |
| Softplus | - | - | - | Y | Y | Y | - | - | - | FP16, FP32, BF16 | - | - | FP16, FP32, BF16 | Qwen MLP 原子算子 |

## 汇总统计

### 按数据类型统计

| DataType | 直接注册算子数 | 说明 |
| --- | --- | --- |
| INT4 | 0 / 46 | 当前没有算子 kernel |
| INT8 | 0 / 46 | `Cast` 可以生成 INT8 输出，但没有 INT8 执行 kernel |
| UINT8 | 0 / 46 | `Cast` 可以生成 UINT8 输出，但没有 UINT8 执行 kernel |
| FP16 | 45 / 46 | 除 `Shape` 外均有直接注册；`ResizeConcat` 仅 CUDA |
| FP32 | 45 / 46 | 除 `Shape` 外均有直接注册；`ResizeConcat` 仅 CUDA |
| BF16 | 36 / 46 | 35 个 Common、28 个 CUDA；`FC` 仅 CUDA BF16 |
| INT32 | 5 / 46 | `Cast`（CUDA 输入）、`ConstantOfShape`、`Equal`、`Expand`、`Where` |
| INT64 | 10 / 46 | `Cast`（CUDA 输入）、`Concat`、`Slice`、`Unsqueeze`、`Squeeze`、`Equal`、`Shape`、`ConstantOfShape`、`Expand`、`Where` |
| BOOL | 7 / 46 | `Reshape`、`Unsqueeze`、`Squeeze`、`Equal`、`ConstantOfShape`、`Expand`、`Where` |

### 按后端统计

| Backend | 直接注册算子类型 | FP16 直接注册 | FP32 直接注册 | BF16 直接注册 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| Common | 45 / 46 | 44 | 44 | 35 | `ResizeConcat` 仅 CUDA；`Shape`/`ConstantOfShape` 控制路径在 Common |
| x86 | 22 / 46 | 22 | 22 | 0 | 覆盖 YOLOv5 原始 22 个基础/融合算子，其余走 Common fallback |
| Arm | 0 / 46 | 0 | 0 | 0 | ARM32/ARM64 当前依赖 Common fallback |
| CUDA | 44 / 46 | 44 | 44 | 28 | 所有数值/融合算子均有 FP16/FP32 CUDA；`Shape`/`ConstantOfShape` 按设计走 Common |

## 分层边界

- 第一层是算子对齐：ONNX 导入或模型格式导入保留原始节点和输入输出关系，不在导入阶段做融合或改写
- 第二层是 pass：负责常量折叠、死节点消除、Identity/Reshape 链消除、`Sigmoid + Mul`、`MatMul + Add`、`Resize + Concat` 等图优化
- Kernel 层只负责对应原子算子的设备实现；当 CUDA 没有原生控制 kernel 时，dispatcher 可以选择 Common kernel，RuntimeGraph 处理设备同步

## 验证记录

- CUDA 编译：`cmake --build build_cuda_ops --target unit_tests -j2` 已通过
- 完整回归：`./build_cuda_ops/bin/unit_tests` 已通过，265 个测试通过；33 个依赖真实 CUDA device 的测试跳过
- CUDA 注册审计：`cuda_kernel_registration_test.*` 已通过；Qwen 所需的 26 个浮点原子算子在 FP32/FP16/BF16 下均命中 CUDA
- Qwen 图 lowering：`qwen_direct_export_test.LowersEveryQwenDecodeNodeToCuda` 已通过，真实 `qwen3.5-0.8b_decode_bf16_ctx128.fth` 的运行时节点没有 Common fallback
- CPU/BF16 回归：`bfloat16_test.*`、`qwen_operator_test.*`、`matmul_flatten_op_test.*` 已通过
- CUDA 数值用例：本机 WSL 当前没有可见 CUDA device，因此 `cuda_extended_ops_test.*` 的 FP32/FP16/BF16 数值测试按预期跳过；在具备 CUDA 设备的环境中运行这些用例即可完成设备侧数值验证
