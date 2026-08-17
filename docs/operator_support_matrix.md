# 算子支持表

## 统计口径

- 统计时间：2026-08-17
- 规范算子范围：`src/operator/*_op.cc` 中当前注册的算子类型；`Conv`、`AveragePool`、`Relu` 是已有规范算子的别名，不重复计数
- 算子总数：39 个，其中 38 个数值/融合算子，`Shape` 是控制流/元数据算子
- 后端支持：以 `KernelDispatcher::registerKernel(...)` 的直接注册结果为准；Common fallback 单独在备注中说明
- CUDA 原生实现：FP32/FP16 请求必须命中 `DeviceType::CUDA` 注册项；缺少 CUDA 设备时，数值测试会跳过，但编译和注册测试仍可执行
- `ResizeConcat` 只由 `resize_concat_fusion_pass` 生成，属于融合后的内部算子，不是 ONNX 原始算子
- 当前 Transformer 模型中的 `LayerNormalization` 在导出阶段展开为 `ReduceMean`、`Sub`、`Pow`、`Add`、`Sqrt`、`Div`、`Mul` 等独立算子，不引入聚合的 Transformer 算子

## 数据类型说明

- `DataType` 当前包含 `INT4`、`INT8`、`UINT8`、`FP16`、`FP32`、`INT32`、`INT64`、`BOOL` 等类型
- `uint16_t` 是 `FP16` 的主机存储表示，不是独立的 `UINT16` 数据类型
- `Cast` 的 FP32/FP16 浮点互转有 CUDA 原生实现；其他控制类型转换统一使用 Common 的通用 Cast 内核
- CUDA 图允许缺少原生 CUDA kernel 的控制节点回退到 Common，并由 RuntimeGraph 负责设备边界同步；这不计入 CUDA 原生支持

## 算子支持矩阵

`Y` 表示该类型在对应后端有直接注册 kernel；`-` 表示没有直接注册。

| Operator | INT4 | INT8 | UINT8 | FP16 | FP32 | INT32 | INT64 | BOOL | Common | x86 | Arm | CUDA | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| BatchNormalization | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Sub | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Div | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Concat | - | - | - | Y | Y | - | Y | - | FP16, FP32, INT64 | FP16, FP32 | - | FP16, FP32 |  |
| Conv2D | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | ONNX `Conv` 别名；CUDA 优先使用 cuDNN，不可用时使用 CUDA fallback kernel |
| FC | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Flatten | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| GlobalAveragePool | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 分类模型算子 |
| Gemm | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Identity | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| MatMul | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Mul | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| AvgPool | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | ONNX `AveragePool` 别名 |
| MaxPool | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Pow | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| ReLU | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | ONNX `Relu` 别名 |
| Reshape | - | - | - | Y | Y | - | - | Y | FP16, FP32, BOOL | FP16, FP32 | - | FP16, FP32 |  |
| Resize | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| ResizeConcat | - | - | - | Y | Y | - | - | - | - | - | - | FP16, FP32 | 只由融合 pass 生成，当前没有 Common/x86 实现 |
| SiLU | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | `Sigmoid + Mul` 融合目标 |
| Sigmoid | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Slice | - | - | - | Y | Y | - | - | - | FP16, FP32, INT64 | FP16, FP32 | - | FP16, FP32 |  |
| Softmax | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | Transformer 基础算子 |
| Split | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Transpose | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 |  |
| Sqrt | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | Transformer 基础算子 |
| Tanh | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | Transformer 基础算子 |
| Erf | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | Transformer 基础算子 |
| Unsqueeze | - | - | - | Y | Y | - | Y | Y | FP16, FP32, INT64, BOOL | - | - | FP16, FP32 | INT64/BOOL 控制输入回退到 Common |
| Squeeze | - | - | - | Y | Y | - | Y | Y | FP16, FP32, INT64, BOOL | - | - | FP16, FP32 | INT64/BOOL 控制输入回退到 Common |
| Cast | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | 非浮点目标统一走 Common 通用 Cast |
| ReduceMean | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | Transformer 基础算子 |
| Gather | - | - | - | Y | Y | - | - | - | FP16, FP32 | - | - | FP16, FP32 | INT32/INT64 indices |
| Equal | - | - | - | Y | Y | Y | Y | Y | FP16, FP32, INT32, INT64, BOOL | - | - | FP16, FP32 | 输出为 BOOL |
| Shape | - | - | - | - | - | - | Y | - | INT64 | - | - | - | 控制流/元数据算子，按设计保留 Common |
| Expand | - | - | - | Y | Y | Y | Y | Y | FP16, FP32, INT32, INT64, BOOL | - | - | FP16, FP32 | shape 输入为 INT32/INT64 |
| Where | - | - | - | Y | Y | Y | Y | Y | FP16, FP32, INT32, INT64, BOOL | - | - | FP16, FP32 | condition 输出为 BOOL |
| YoloDecode | - | - | - | Y | Y | - | - | - | FP16, FP32 | FP16, FP32 | - | FP16, FP32 | YOLOv5 head decode 融合算子 |

## 汇总统计

### 按数据类型统计

| DataType | 直接注册算子数 | 说明 |
| --- | --- | --- |
| INT4 | 0 / 39 | 当前没有算子 kernel |
| INT8 | 0 / 39 | `Cast` 可以生成 INT8 输出，但没有 INT8 执行 kernel |
| UINT8 | 0 / 39 | `Cast` 可以生成 UINT8 输出，但没有 UINT8 执行 kernel |
| FP16 | 38 / 39 | 除 `Shape` 外，所有数值/融合算子均支持 |
| FP32 | 38 / 39 | 除 `Shape` 外，所有数值/融合算子均支持 |
| INT32 | 3 / 39 | `Equal`、`Expand`、`Where` 的 Common 控制路径 |
| INT64 | 8 / 39 | `Concat`、`Slice`、`Unsqueeze`、`Squeeze`、`Equal`、`Shape`、`Expand`、`Where` |
| BOOL | 6 / 39 | `Reshape`、`Unsqueeze`、`Squeeze`、`Equal`、`Expand`、`Where` |

### 按后端统计

| Backend | 直接注册算子类型 | FP16 直接注册 | FP32 直接注册 | 备注 |
| --- | ---: | ---: | ---: | --- |
| Common | 38 / 39 | 37 | 37 | 另有 `Shape(INT64)`；`ResizeConcat` 仅 CUDA |
| x86 | 22 / 39 | 22 | 22 | 覆盖 YOLOv5 原始 22 个基础/融合算子，其余走 Common fallback |
| Arm | 0 / 39 | 0 | 0 | ARM32/ARM64 当前依赖 Common fallback |
| CUDA | 38 / 39 | 38 | 38 | 覆盖全部数值/融合算子；`Shape` 按设计走 Common |

## 分层边界

- 第一层是算子对齐：ONNX 导入保留原始节点和输入输出关系，不在导入阶段做融合或改写
- 第二层是 pass：负责常量折叠、死节点消除、Identity/Reshape 链消除、`Sigmoid + Mul`、`MatMul + Add`、`Resize + Concat` 等图优化
- Kernel 层只负责对应算子的设备实现；当 CUDA 没有原生控制 kernel 时，dispatcher 可以选择 Common kernel，RuntimeGraph 处理设备同步

## 验证记录

- CUDA 编译：`cmake --build build_cuda_ops -j2` 已通过
- CUDA 注册审计：`cuda_kernel_registration_test.*` 已通过，38 个数值/融合算子的 FP32/FP16 CUDA kernel 均可直接选择
- CUDA 数值测试：`cuda_kernel_test.*`、`cuda_extended_ops_test.*` 已通过，分类与 Transformer 新增算子均完成 FP32/FP16 验证
- CUDA 模型后端选择：ResNet50、RepVGG B0 的 `classification_real_models_test.*` 已通过，未发生 Host/Common fallback
- CPU 真实模型：ResNet50、RepVGG B0 以及 3 个 Transformer 模型均通过 ONNX Runtime 参考对比
