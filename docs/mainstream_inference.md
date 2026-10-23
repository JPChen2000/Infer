# 主流模型推理路径

2026-10-22：优先补现代检测、视觉 Transformer 与 dense/hybrid LLM 的高频算子。RoIAlign、RoIPool 等旧检测专用路径暂缓。已有 YOLO nano 范围见 [检测模型说明](yolo_detection.md)。支持基础算子并不代表任意新模型检查点可以直接运行，仍需对应导出、位置编码和状态契约。

## 新增算子

| 算子 | Common / CUDA | 语义与边界 |
| --- | --- | --- |
| RMSNormalization | FP32、FP16、BF16 | FP32 规约；axis、epsilon、广播 scale；可选 bias 和 weight_offset 扩展 |
| LayerNormalization | FP32、FP16、BF16 | Welford 统计、广播 affine；可选连续命名的 FP32 Mean / InvStdDev；stash_type=1 |
| Resize | FP32、FP16、BF16 | nearest；四维 NCHW/NHWC 空间 bilinear；half_pixel、pytorch_half_pixel、align_corners、asymmetric；四种 nearest 舍入模式 |
| CachedAttention | FP32、FP16、BF16 | causal MHA/GQA、FP32 累加、按绝对位置写 KV、只读有效因果前缀 |
| Slice | CUDA 新增 BF16 | 分块卷积状态切片不再回退 Common |

Norm CUDA 最大 rank 为 8，不支持输出槽空洞、重叠输出、残差融合。Resize 不支持 cubic、非空 ROI、antialias、axes、exclude_outside、外推与非默认宽高比策略；bilinear 不改变 batch/channel。旧 FTH 未指定属性时保留 nearest/asymmetric/floor；ONNX 导入根据 opset 写入显式默认值。ResizeConcat 只融合兼容的整数倍 nearest 语义。

CachedAttention 输入依次为 `Q [B,H,T,D]`、`K [B,Hkv,T,D]`、`V [B,Hkv,T,Dv]`、`cacheK [B,Hkv,C,D]`、`cacheV [B,Hkv,C,Dv]`、`position INT64 [1]`，输出 `[B,H,T,Dv]`。`H` 必须是 `Hkv` 的整数倍。缓存是可变外部输入、独占消费者；编译会拒绝与其他输入重叠的缓存。执行必须串行，设备缓存持久保留。`position` 表示本 chunk 起点，调用方必须保持历史连续性。运行图在执行任何节点前检查边界，捕获回放调用方也必须逐次检查。`round_intermediates=1` 保留 Qwen 的 BF16 score/probability 舍入边界。

CUDA attention 使用共享内存分块和 online softmax，不创建全局 score 矩阵；这是当前融合实现，没有宣称达到 FlashAttention 的吞吐。

## Qwen3.5 单图与分块预填充

保留原导出器和旧 FTH 路径，新增优化导出器：

```sh
OMP_NUM_THREADS=4 python3 -B tools/export_qwen35_optimized.py \
  --model-dir models/llm/qwen3.5-0.8b \
  --max-context 128 --prefill-chunk 16 \
  --output-fth models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_bf16.fth

./build_new/bin/qwen_demo \
  --model models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_bf16.fth \
  --tokenizer models/llm/qwen3.5-0.8b --backend cuda \
  --context-size 128 --max-new-tokens 64 --prompt '介绍一下你自己'
```

优化导出器只产生一个 FTH v4 文件，包含一个语义图和一份常量数据；不产生命名 variant、独立 prefill 图或配套权重文件。旧原子导出器继续写 v4，加载器保留旧 FTH v1–v5 以及旧 optimized v1 的兼容路径。`--max-context` 指定默认起始容量，默认 4096；`--prefill-chunk` 指定序列特化上限，默认 16，不要求必须为 2 的幂。示例中的 `--context-size 128` 设置硬容量上限；省略它或设为 0 才启用模型上限内的自动扩容。

优化图使用 `qwen.execution.version=2`、`qwen.execution.sequence=1`、`qwen.sequence.version/default/maximum`、`qwen.cache.layout=position` 和 `qwen.weights.id` 元数据。上下文与序列变化分别使用显式绑定，不根据维度数值相等推测符号。序列绑定覆盖 Tensor 维度、INT64 形状常量及整数标量/向量属性；RoPE 的 `qwen_chunk_offsets` 在加载特化时生成 `arange(T)`，同步更新描述与常量覆盖。权重指纹包含权重、配置、量化选项及精度策略版本，`qwen.precision.policy` 记录敏感权重保留策略；变换后的权重名称在不同序列长度间保持稳定。

runner 从同一加载图按需编译并缓存长度 1、2、4、8 等计划，最大不超过 `--prefill-chunk` 和当前容量。例如上限 16 时，23-token 输入可分解为 16+4+2+1，余数使用较小批量计划，无需 padding。投影、MLP 和 attention 按块计算；每个线性 attention 层使用一个 `QwenGatedDeltaSequence`，在算子内部按 token 顺序更新 FP32 recurrent 状态，长度 1 也使用同一图。

同一 runner 的所有长度计划绑定同一组可变状态 Tensor，并共享 CUDA 执行上下文/stream 和不可变权重分配；各自持有激活、Kernel 与捕获资源，执行保持串行。KV 按绝对位置原地写入；卷积与 recurrent 输出更新到地址固定的状态输入，长度切换无需复制状态。后端按图引用管理共享缓存，销毁一个计划不清除其他计划仍在使用的设备缓存。首次使用一个长度仍有编译、分配及预热开销。

`QwenRunner::ReserveContext(capacity)` 显式准备更大的上下文特化，在新会话准备和状态迁移成功后再替换旧会话。迁移保留 KV 有效前缀（按新行跨度复制）、卷积/recurrent 状态和已处理位置；地址变化会使旧捕获与序列计划失效，随后按需重建。失败时保留原会话。默认自动增长按需申请容量，始终不超过模型配置上限；显式正数 `--context-size` 是应用硬上限。

多轮控制器在规划裁剪前，为完整历史、新输入、结束标记和生成预算申请容量；`--max-new-tokens -1` 至少预留一个生成 token，并在继续生成时按容量策略处理。达到模型上限或显式容量后，仍使用保留 system 与最近完整轮次的裁剪/Reset/精确 token 重放策略。扩容失败直接返回错误并保留历史。执行失败可能部分写入 KV，runner 须 Reset 后再执行；聊天控制器尝试恢复完整历史。取消在完整块/token 边界处理。

可选 `--fp8-format e4m3|e5m2` 保留混合精度边界：主要线性投影的输入、权重和结果使用 FP8，再恢复 BF16；embedding 与共享输出头使用同一份原始 BF16 常量，深度卷积权重和小型 a/b 衰减、beta 控制投影也保留 BF16。残差、MLP 激活、RoPE、KV 和深度卷积计算使用 BF16；L2、衰减、门控归一化与 recurrent 更新使用 FP32。归一化权重和衰减参数保留原精度。该策略同时用于原子和优化导出器，避免先量化敏感权重再以 BF16 计算；已有 FP8 文件需要重新导出才能采用它。其余 FP8 投影仍会量化输入及结果，静态激活范围和长序列误差需要数值对照评估。

`LastPrefillMilliseconds()` 包括该次提示的所有块、余数与首次使用计划的准备开销；模型初始加载不计入。demo 在首个输出 token 处单独打印 `prompt_prefill_ms`，避免结束标记的 `Consume()` 覆盖指标。后续 `Consume()` 会更新最近提示指标，应用需在处理结束标记前读取。`CachedPlanCount()` 和 `ContextGrowthCount()` 分别提供当前缓存计划数与已完成扩容次数。

## 本次单图验证

后续逐 token 困惑度、生成性能和 E4M3 校准入口见 [Qwen 评测与校准](qwen_evaluation.md)。

在 RTX 4070 上运行实际 Qwen3.5-0.8B 单图文件。下面对照相同 teacher-forcing token 输入后的最后一个完整 logits 向量，共 248,320 个值；参考是官方 Hugging Face BF16 eager 顺序执行。每个长度只测一条输入，其中长输入包含重复技术文本，不能视为数据集精度、困惑度或模型完整上下文上限的验证。

| 输入 token 数 | BF16 logits 余弦相似度 | RMSE | 最大绝对差 | greedy token 一致 |
| --- | --- | --- | --- | --- |
| 25 | 0.999808 | 0.05882 | 0.28125 | 是 |
| 76 | 0.999695 | 0.06506 | 0.31250 | 是 |
| 128 | 0.999805 | 0.04957 | 0.26563 | 是 |
| 256 | 0.999864 | 0.05962 | 0.28125 | 是 |
| 512 | 0.999923 | 0.04248 | 0.23438 | 是 |
| 1024 | 0.999885 | 0.04458 | 0.28125 | 是 |
| 2048 | 0.999948 | 0.04287 | 0.25000 | 是 |

FP8 敏感权重恢复原始 BF16 后，76-token 输入的 E4M3 余弦相似度从 `0.971665` 提升到 `0.985674`，greedy 从错误恢复为一致；E5M2 从 `0.916540` 提升到 `0.947170`，greedy 仍不一致。2048-token 输入下，新 E4M3 的余弦相似度为 `0.994165`、RMSE 为 `0.51344`、最大绝对差为 `5.125`，greedy 一致。仍有明显量化损失；精度优先使用 BF16，需要 FP8 时优先 E4M3。这次没有实现激活校准或证明 FP8 与 BF16 等价。

实际运行同时检查所有 48 个状态 Tensor 和 CUDA 上下文在五个序列计划（1、2、4、8、16）之间共享。已有 CUDA Graph 捕获后，从逐 token 切到分块仍可继续生成。先在容量 64 中处理 48 token 并完成捕获，再扩到 128、处理剩余 28 token：48 份状态、greedy 输出和随后 24 个生成 token 与从容量 128 开始、采用相同输入分块的对照逐字节一致。分块 GEMM 与纯逐 token GEMM 可能出现 BF16 舍入差异，因此不同分块不要求状态逐字节相同。

最终 demo 三轮中文对话保留了前文信息，容量从 128 增长到 143、206，扩容两次。修复了回归验证暴露的两个 CUDA 问题：捕获所用的临时地址被其他计划复用，以及捕获失败后设备有效标记/别名未恢复。成功捕获现在持有分配租约；失败捕获在普通执行回退前恢复缓存元数据。

完整 C++ 回归 **693/693 通过**，包含最初列出的 FP8、Qwen fusion 和 YOLO persistent-device 三项；两个 Python 导出器测试集共 **14/14 通过**。完整回归还修正了 DistilBERT 的旧断言：静态 Shape 已折叠，测试现在校验六个替代 INT64 形状常量 `[1,12,8,8]`，并保留实际 masked logits 对 ONNX Runtime 的对照。

诊断程序提前准备长度计划后，76-token 分块输入单次执行为 `210.45 ms`，1024/2048-token BF16 输入分别为 `1223/2571 ms`；这些数值包含同步但不包含模型加载和提前准备计划，不是 demo 首次提示耗时或稳定态性能基准。诊断末尾的全卡已用显存约 2.95–2.98 GB，包含其他进程、CUDA 上下文和缓存，不能解释为模型独占峰值。原始日志、输入 IDs、logits、参考数组及状态哈希记录保存在本机忽略目录 `models/mainstream_validation/qwen_unified/`，汇总为 `qwen_unified_precision.json`、`qwen_unified_state.json`。25–512-token 精度记录在最终捕获分配修正前测得，1024/2048-token 和上述捕获/扩容检查使用最终构建；汇总中标明阶段。

可用持久化工具重新比较两个已导出的向量：

```sh
python3 -B tools/compare_qwen_logits.py \
  --actual models/mainstream_validation/qwen_unified/infer-unified-bf16_full2048_final.logits.bin \
  --reference models/mainstream_validation/qwen_unified/infer-qwen-unified-hf-extended_long_2048.npy
```

本机已生成 `models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_bf16_ctx128.fth` 和后缀为 `unified_fp8e4m3_bf16islands_ctx128.fth`、`unified_fp8e5m2_bf16islands_ctx128.fth` 的新文件；旧模型保留。需要自动扩容时省略运行参数 `--context-size` / `--max-context`；这里的文件名 `ctx128` 表示起始容量。

以下性能与精度记录来自旧版固定 decode/prefill 计划，保留为历史记录。它们不构成上述单图序列计划、共享状态或自动扩容行为的验证结果。

## 已测范围

RTX 4070、CUDA、Qwen3.5-0.8B BF16、context 128、chunk 16。同一 76-token 提示，进程内 Reset 后重复三次，以下预填充范围排除第一次预热；decode 为后续实际逐 token 执行，包含同步和状态更新。

| 路径 | 76-token 预填充 | decode |
| --- | --- | --- |
| 旧图逐 token | 620–630 ms | 中位数 7.69 ms/token |
| 优化图逐 token | 587–589 ms | 中位数 7.03 ms/token |
| 优化图 chunk16 | 284–295 ms | 中位数 6.95 ms/token |

分块预填充约快 2 倍。25-token 提示为 140–147 ms，对应逐 token 205–212 ms。旧版独立加载两个模型时，全设备已用显存约 4.60 GB，对应 decode 单独运行约 2.96 GB；这项重复权重分配已被共享方案替代。全设备数字包含 CUDA 上下文等开销，不能视为精确的模型独占峰值。

单文件共享权重修正后，同一 76-token 提示重复五次，排除首次预热的 prefill 为 234–272 ms，decode 中位数 6.77 ms/token；逐 token 对照为 553–561 ms。两个计划中 151 个大权重的主机地址、CUDA 地址全部一致，共用 1,502,609,408 字节的设备分配。全设备已用显存约 3.04 GB（含 prefill）与 2.96 GB（仅 decode），对照旧双模型约 4.60 GB。数值受系统基线和 GPU 频率影响，不作为跨框架基准。

共享前后同一 25-token 提示的 248,320 个 BF16 logits 逐字节一致。另实际验证双向权重修改会分离分配、保留对方数据，并在销毁 decode 会话后让 prefill 独立推理，得到相同输出；Reset/取消/越界拒绝/末尾执行失败恢复的 20 项真实模型诊断通过。记录见 `models/mainstream_validation/qwen_shared_*.json`（本机忽略目录）。

同一 25-token 提示的实际全量 logits 对照中，优化逐 token 与旧图逐元素一致；chunk16 对逐 token 的余弦相似度 `0.999860`，最大绝对差 `0.21875`。chunk16 对 Hugging Face BF16 eager 顺序参考的余弦相似度 `0.999836`，最大绝对差 `0.2578125`，greedy token 与 top10 集合一致。测量保留其他 Qwen passes，只关闭最终 greedy argmax 融合以读取全量 logits；这是一条提示的实测，不能替代完整数据集精度评估。

CachedAttention 使用真实 embedding 直接经过 layer3 投影、q/k norm、RoPE 得到的 Q/K/V（算子独立对照未先执行前 3 层），Common 与严格 CUDA 各完成 42 组独立对照，覆盖 GQA、batch 2、非零起点、chunk、容量边界及不同 D/Dv；每组验证旧前缀和尾部缓存不变、当前区间精确写入、只读输入不变。

真实 Qwen embedding/norm 权重与真实图片 crop，Common 和严格 CUDA 后端各进行 123 组独立数值对照。CUDA Norm Y 最大绝对误差分别为 FP32 `4.77e-7`、FP16 `4.88e-4`、BF16 `3.90625e-3`；Resize 为 `4.62e-7`、`2.44e-4`、`1.953125e-3`。两后端各 54 组 nearest 逐元素一致。低精度 Resize 参考为量化输入经 ORT FP32 插值后重新量化；不宣称等同 ORT 原生低精度 kernel。详细来源和原始测量保存在被 Git 忽略的 `models/mainstream_validation/`。另外已实际验证三轮对话的历史裁剪重放、运行容量 8/64/256、取消后状态一致、越界前拒绝及末尾算子失败后的 Reset 恢复。

当前没有 continuous batching、paged KV、MoE expert 路由、多 GPU 张量并行或服务调度器。现有测量说明小模型单请求路径得到改善，尚不能据此声称最新大模型全覆盖或性能达到 vLLM、TensorRT-LLM、SGLang 等框架。
