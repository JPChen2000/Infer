# 算子支持表

## 当前增量支持（2026-10-22）

下表按当前算子检查逻辑和 kernel 注册记录增量能力，不是全量算子清单，也不重新计算算子总数。下文的 46 算子矩阵、汇总数字和验证记录保留为 2026-08-17 历史快照；与本表不一致时，以本表列出的当前边界为准。这里的“支持”描述代码路径，不代表本次已运行数值测试或完成所有模型验证。

| 算子 / 能力 | 当前直接注册后端与类型 | 支持范围与限制 |
| --- | --- | --- |
| `RMSNormalization` / `LayerNormalization` | Common、CUDA：FP32、FP16、BF16；x86/Arm 可走 Common fallback | 从 `axis` 到末维归一化，支持负轴、广播 scale 和可选 bias；归约使用 FP32。只支持 `stash_type=1`；LayerNormalization 可按连续输出槽返回 FP32 Mean / InvStdDev，RMSNormalization 支持 `weight_offset` 扩展。拒绝不支持的属性、输出空洞及输入输出重叠。 |
| `Resize` | Common、CUDA：FP32、FP16、BF16；x86：FP32、FP16、INT8；x86 BF16 可走 Common fallback | nearest 可处理各轴；linear 限 rank 4 的 NCHW/NHWC 空间双线性，batch/channel 不变。支持完整 scales 或正 INT64 sizes 向量；scales 输出尺寸取 floor。支持 `half_pixel`、`pytorch_half_pixel`、`align_corners`、`asymmetric` 与四种 nearest_mode；拒绝 cubic、非空 ROI、axes 子集、antialias 等未实现语义。x86 INT8 要求 per-tensor 量化。 |
| 内部 `CachedAttention` | Common、CUDA：FP32、FP16、BF16 | 因果注意力与 GQA；按绝对 position 原地写入当前 K/V chunk，只读取有效前缀，不展开 KV heads。Q/K/V/cache 类型一致，Q head 数必须是 KV head 数的整数倍；cache/position 为外部输入，cache 独占、可变且持久保存，图执行限串行。CUDA 融合计算不分配全局 score 矩阵；不表示已实现 FlashAttention 或通用 ONNX Attention。 |
| `Slice` 的 BF16 路径 | Common、CUDA：BF16；x86 可走 Common fallback | 支持单轴、步长 1；一输入节点使用 `axis/start/end`，或通过控制张量输入给出单元素 starts/ends/axes（可选 steps）。不代表支持多轴切片或任意步长。 |
| `NonMaxSuppression` | Common、CUDA：FP32 boxes/scores，INT64 输出 | boxes `[B,N,4]`、scores `[B,C,N]`，支持 corner/center 两种 box 格式及可选 maximum/IoU/score 控制。输出逻辑形状 `[selected,3]`，底层保留容量；当前 ONNX 导入与 compiled graph 要求 NMS 输出直接作为图输出，不支持动态结果的后继计算。CUDA 原生路径有安全索引、CUB 数量和 workspace 限制。 |
| `ConvTranspose` | Common、CUDA：FP32 | rank 4 NCHW 数据、`[Cin,Cout/group,kH,kW]` 权重，支持 group、可选 bias、显式 strides/pads/dilations/output_padding。当前导入拒绝非 NOTSET auto_pad 和显式 output_shape；不支持 NHWC/3D/其他精度，CUDA 还要求相关尺寸与属性适合 INT32 索引。 |
| YOLO selection：`TopK` / `GatherElements` / `ReduceMax` / `Mod` | Common：FP32、FP16、BF16、FP8E4M3/E5M2、INT64、INT32、INT8、UINT8；GatherElements 另支持 BOOL | TopK 的 K 与 GatherElements 的 indices 使用 INT64；TopK 返回 INT64 indices，支持 axis/largest/sorted。ReduceMax 支持 axes/keepdims/noop_with_empty_axes；Mod 支持广播，浮点输入只接受 `fmod=1`。注册类型仍受各算子形状、控制值和量化检查约束。 |
| YOLO selection 的 CUDA 子集 | `TopK`、`ReduceMax`：FP32；`GatherElements`：FP32、INT64；`Mod`、整数 `Div`：INT64 | TopK 使用稳定排序，`sorted=0` 仍可返回有序结果；GatherElements indices 为 INT64。CUDA selection 限 rank ≤ 8，并受索引/workspace 检查约束；其他注册类型可走 Common fallback，不计为 CUDA 原生支持。 |

直接 Feather Resize 节点省略属性时保留历史 `nearest/asymmetric/floor` 默认；ONNX 导入显式保存相应 opset 默认，opset 10 为 `asymmetric/floor`，opset 11 起为 `half_pixel/round_prefer_floor`。优化后的 Qwen 导出使用版本化执行元数据和内部融合算子，旧原子导出仍保留；不能再将历史“所有 LLM 节点均保持展开”描述视为当前限制。

## 历史统计口径（2026-08-17）

- 统计时间：2026-08-17
- 规范算子范围：`src/operator/*_op.cc` 中当前注册的算子类型；`Conv`、`AveragePool`、`Relu` 是已有规范算子的别名，不重复计数
- 当时的算子总数：46 个。其中 44 个为数值/融合算子，`Shape` 与 `ConstantOfShape` 是控制流/元数据算子；此数字不代表 2026-10-22 当前总数
- 后端支持：以 `KernelDispatcher::registerKernel(...)` 的直接注册结果为准；Common fallback 单独在备注中说明
- CUDA 原生实现：请求必须命中 `DeviceType::CUDA` 注册项；缺少 CUDA 设备时，数值测试会跳过，但编译、注册和图 lowering 测试仍可执行
- `ResizeConcat` 只由 `resize_concat_fusion_pass` 生成，属于融合后的内部算子，不是 ONNX 原始算子
- 当时的 Transformer/LLM 导出保持原子算子：`LayerNormalization`、RoPE、attention 和 MLP 在导出图中展开；此描述仅适用于该历史快照

## 历史数据类型说明（2026-08-17）

- `DataType` 当前包含 `INT4`、`INT8`、`UINT8`、`FP16`、`BF16`、`FP32`、`INT32`、`INT64`、`BOOL` 等类型
- `uint16_t` 是 `FP16` 的主机存储表示；`BFloat16` 是独立的 16 位 BF16 存储类型，两者不能互相重解释
- CUDA `Cast` 原生支持 FP32、FP16、BF16 互转，以及 INT32/INT64 到 FP32/FP16/BF16；其他控制类型转换保留 Common 通用内核
- CUDA BF16 的 MatMul/Gemm/FC 使用 cuBLAS BF16 输入与 FP32 累加；运行这些路径需要设备与 CUDA 库支持 BF16。BF16 Conv2D 固定使用项目的 CUDA fallback kernel，不依赖 cuDNN 的 BF16 支持
- CUDA 图允许缺少原生 CUDA kernel 的控制节点回退到 Common，并由 RuntimeGraph 负责设备边界同步；这不计入 CUDA 原生支持

## 历史算子支持矩阵（2026-08-17）

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

## 历史汇总统计（2026-08-17）

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

## 历史分层边界（2026-08-17）

- 第一层是算子对齐：ONNX 导入或模型格式导入保留原始节点和输入输出关系，不在导入阶段做融合或改写
- 第二层是 pass：负责常量折叠、死节点消除、Identity/Reshape 链消除、`Sigmoid + Mul`、`MatMul + Add`、`Resize + Concat` 等图优化
- Kernel 层只负责对应原子算子的设备实现；当 CUDA 没有原生控制 kernel 时，dispatcher 可以选择 Common kernel，RuntimeGraph 处理设备同步

## 历史验证记录（2026-08-17）

以下命令、通过数和设备可见性均是当时记录，不代表当前环境或本次增量的验证结果；本次文档更新未执行构建、GPU 运行或测试。

- CUDA 编译：`cmake --build build_cuda_ops --target unit_tests -j2` 已通过
- 完整回归：`./build_cuda_ops/bin/unit_tests` 已通过，265 个测试通过；33 个依赖真实 CUDA device 的测试跳过
- CUDA 注册审计：`cuda_kernel_registration_test.*` 已通过；Qwen 所需的 26 个浮点原子算子在 FP32/FP16/BF16 下均命中 CUDA
- Qwen 图 lowering：`qwen_direct_export_test.LowersEveryQwenDecodeNodeToCuda` 已通过，真实 `qwen3.5-0.8b_decode_bf16_ctx128.fth` 的运行时节点没有 Common fallback
- CPU/BF16 回归：`bfloat16_test.*`、`qwen_operator_test.*`、`matmul_flatten_op_test.*` 已通过
- CUDA 数值用例：2026-08-17 的 WSL 环境当时没有可见 CUDA device，因此 `cuda_extended_ops_test.*` 的 FP32/FP16/BF16 数值测试跳过；这不是对当前环境 GPU 可用性的判断
