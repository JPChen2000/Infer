# Qwen 质量评测与 FP8 校准

这条离线路径用于比较 BF16、已有 FP8 和候选量化策略。它保留每个 token 的 loss、greedy 结果、各文档最后一个完整 logits，以及输入、模型、程序指纹。生成性能单独走正常 argmax 融合路径；完整词表评分的耗时不作为 decode 吞吐。

## 构建与评分

```sh
cmake -S . -B build_new
cmake --build build_new --target qwen_eval -j4

python3 -B tools/evaluate_qwen.py \
  --model models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_bf16_ctx128.fth \
  --tokenizer models/llm/qwen3.5-0.8b \
  --dataset tools/data/qwen/heldout.jsonl --context 256 \
  --output models/mainstream_validation/my_bf16

python3 -B tools/evaluate_qwen.py \
  --model models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_fp8e4m3_bf16islands_ctx128.fth \
  --tokenizer models/llm/qwen3.5-0.8b \
  --dataset tools/data/qwen/heldout.jsonl --context 256 \
  --reference models/mainstream_validation/my_bf16 \
  --max-relative-ppl-increase 0.05 \
  --output models/mainstream_validation/my_e4m3
```

需要已有 numpy、tokenizers；校准/导出和局部误差分析另外使用项目原有的 torch、safetensors、ml_dtypes。工具不安装软件包。`--output` 必须是新目录或空目录，避免覆盖旧测量。输入相同但程序/模型不同时会保留各自 SHA256；参考对照严格校验 token manifest。质量门槛显式由调用者提供，上例的 5% 只是示例，超出门槛时写完报告并返回状态码 2，运行错误返回非零。

`score` 对长度 N 的文档执行 N−1 个单 token 步骤。每次真实读取完整分布，使用 FP64 logsumexp 评分下一个已知 token。困惑度是 `exp(所有目标 token 的 NLL 总和 / 目标 token 总数)`，不平均各文档的困惑度。首 token 没有前文，不计 loss；文档间 Reset，文档内部保持全部 KV/卷积/recurrent 状态。聊天评分包含给定完整转录及分隔 token，并非只统计 assistant 内容。BF16 对照是同一运行时的 BF16 文件，不冒充另一框架或官方模型的逐 token 参考。

可用 `tools/data/qwen/heldout_context.jsonl` 和 `--context 4096` 评分 2048/1024-token 连续文档。`--max-tokens` 只在显式指定时截取前缀，manifest 记录截断。默认容量 0 允许自动扩容，显式正数是硬上限；越界拒绝，评测不裁剪或重放历史。支持用户提供更大的 JSONL 语料，格式见 [语料说明](../tools/data/qwen/README.md)。附带数据为自编和仓库文档快照，仅用于 smoke 对照，不能替代标准数据集或真实应用验证。

## 采集与导出

重新导出 BF16，以获得投影输入/输出和检查点指纹元数据：

```sh
OMP_NUM_THREADS=2 python3 -B tools/export_qwen35_optimized.py \
  --max-context 128 --prefill-chunk 16 \
  --output-fth models/llm/qwen3.5-0.8b/qwen_eval_bf16.fth

python3 -B tools/evaluate_qwen.py --mode calibrate \
  --model models/llm/qwen3.5-0.8b/qwen_eval_bf16.fth \
  --tokenizer models/llm/qwen3.5-0.8b \
  --dataset tools/data/qwen/calibration_context.jsonl --context 2048 \
  --output models/mainstream_validation/my_calibration

OMP_NUM_THREADS=2 python3 -B tools/export_qwen35_optimized.py \
  --fp8-format e4m3 --max-context 128 --prefill-chunk 16 \
  --fp8-calibration models/mainstream_validation/my_calibration/calibration.json \
  --output-fth models/llm/qwen3.5-0.8b/qwen_candidate_e4m3.fth
```

采集在未量化图上逐 token 执行，保留显式命名的投影 Tensor 后读取其真实值。评分及校准的诊断模式禁用 CUDA Graph 捕获，确保每步刷新主机观察缓存；正常生成基准仍采用捕获优化。150 个主要投影分别记录输入、输出绝对最大值、RMS、元素数；小型 a/b recurrent 控制投影继续保留原始 BF16。默认对观测 amax 加 10% 裕量，导出按各自范围生成独立的二次幂 scale。零激活使用极小正范围以保留有效描述。范围具有语料依赖，观察到的最大值不能保证覆盖未见数据，缩小范围可能导致饱和和质量退步。

校准 JSON version=1 包含 format、检查点/config/corpus/token SHA256、采样 token 数、各投影范围和可选 `keep_bf16`。导出检查配置、检查点、完整投影覆盖、有限正范围和观察数，拒绝错误模型、遗漏投影及无匹配的保留模式。优化文件的权重身份包含校准内容和保留策略。单文件/单图结构保持不变；旧文件需要重新导出才采用新策略。

局部敏感性分析使用每篇校准文档最后一个真实投影输入/输出向量，分别重放各投影：

```sh
OMP_NUM_THREADS=2 python3 -B tools/analyze_qwen_fp8.py \
  --model-dir models/llm/qwen3.5-0.8b \
  --calibration-dir models/mainstream_validation/my_calibration \
  --device cpu --keep-top 12 \
  --output models/mainstream_validation/my_sensitivity
```

输出 `sensitivity.json` 和带精确保留列表的候选 `calibration.json`。该估计不传播其他层的量化误差，因此局部排序不能保证整体质量提升；必须重新评分。报告还比较了权重逐输出通道的二次幂缩放误差，但当前导出/Kernel 仍使用逐 Tensor scale，不将这项分析表示为后端已支持逐通道量化。

也可不采用范围文件，仅显式保留若干 BF16 投影，例如：

```sh
OMP_NUM_THREADS=2 python3 -B tools/export_qwen35_optimized.py \
  --fp8-format e4m3 --max-context 128 --prefill-chunk 16 \
  --fp8-keep-bf16 '*.linear_attn.out_proj.weight' \
  --fp8-keep-bf16 '*.self_attn.o_proj.weight' \
  --fp8-keep-bf16 '*.mlp.down_proj.weight' \
  --output-fth models/llm/qwen3.5-0.8b/qwen_e4m3_bf16residual.fth
```

## 正常路径性能

```sh
python3 -B tools/evaluate_qwen.py --mode benchmark \
  --model models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_bf16_ctx128.fth \
  --tokenizer models/llm/qwen3.5-0.8b \
  --dataset tools/data/qwen/benchmark.jsonl --context 256 \
  --new-tokens 32 --repeats 3 \
  --output models/mainstream_validation/my_benchmark
```

每篇提示先运行一次，再 Reset 重复指定次数。run 0 包含首次使用长度计划的编译/分配；热运行保留计划，但 Reset 仍重新建立 CUDA 捕获。后续文档可能复用前面文档已准备的长度，因此 run 0 不是每篇文档完全独立冷启动。`load_ms` 单独记录 Load 调用，不包含后续首次执行才发生的上传/预热。

TTFT 从生成调用开始计到首个流式回调；prefill 为 runner 的提示处理耗时；decode 为后续回调间隔，包含同步、状态更新和输出选择。最后一次生成 token 的回填计算计入总耗时，未计入 callback-gap decode。固定生成数量，不因 EOS 提前停止；记录生成文本、ID 和重复运行一致性，文本供人工核查，不自动宣称代码执行或数学答案正确率。device-used 在运行后采样，含其他进程和 CUDA 上下文，属于全卡观测值，不是模型独占峰值或分配器峰值。

## C++ 入口

`QwenLoadOptions::retain_logits=true` 在加载时仅关闭最终 GemmArgmax 融合；`EvaluateToken(token, &logits)` 执行真实单 token 推理并返回 FP32 观察向量。`additional_outputs` 可显式保留诊断输出，`ReadOutput` 只允许读取最近一次成功 EvaluateToken 的声明输出。Reset、其他执行和上下文替换后观察值失效。选项传播到增长后的会话和序列计划。默认 Load/生成仍使用正常优化。诊断模式不捕获 CUDA Graph，避免重放时读取上一 token 的中间激活。

`qwen_eval` 可直接读取每行一篇文档的空格分隔 token IDs，详见 `build_new/bin/qwen_eval --help`。它在执行前检查词表和容量，拒绝目录中已有的运行产物，输出逐 token TSV、原始 final logits、运行摘要；Python 层负责语料/程序指纹、聚合、参考对照及校准 JSON。


## 2026-10-23 实测与决策

设备为 RTX 4070（12 GB），Qwen3.5-0.8B、CUDA、batch=1。校准为 14 篇、2234 tokens；独立短文本评分为 645 个目标 token，长文本为 3070 个，合计 3715 个。合并困惑度按目标 token 的 NLL 加权。参考为同一运行时 BF16；这些自编语料和仓库文本快照不能代表标准 benchmark、模型整体知识或答案正确率。

审查复现了诊断数据失效：CUDA Graph 重放后中间输出的 host-valid 标记未失效，读取激活会返回旧 token。当前离线诊断禁用捕获，并以连续变化 token 对照 Common，覆盖额外保留的投影和原有 recurrent-state 输出。修正后 300 项范围中有 235 项改变；旧校准及其派生策略已标记作废，原文件保留。校准重采、敏感性分析、候选导出和评分使用 `reviewed/` 新目录。

| 策略 | 短文本 PPL | 长文本 PPL | 合并 PPL | 相对 BF16 增幅 | greedy 与 BF16 一致率 |
|---|---:|---:|---:|---:|---:|
| BF16 | 13.332 | 45.831 | 36.987 | — | — |
| 原 E4M3 | 13.593 | 47.461 | 38.199 | 3.28% | 90.23% |
| 校准输入/输出范围 | 13.797 | 47.242 | 38.153 | 3.15% | 89.64% |
| 校准 + 敏感 12 投影 BF16 | 13.834 | 47.350 | 38.242 | 3.39% | 89.58% |
| 原范围 + 敏感 12 投影 BF16 | 13.745 | 47.161 | 38.073 | 2.94% | 90.85% |
| 原范围 + 48 个输出投影 BF16 | 13.790 | 46.838 | 37.879 | 2.41% | 91.49% |

当前候选均未同时超过原 E4M3 的短、长文本结果。默认配置保持不变，精度优先继续推荐 BF16；校准和 BF16 保留策略作为显式可选实验。局部敏感性误差、greedy 一致率和目标 NLL 并非同一指标，增加 BF16 投影不保证目标困惑度单调改善。

校准候选模型分别为 `qwen3.5-0.8b_e4m3_reviewed_calibrated_ctx128.fth`、`qwen3.5-0.8b_e4m3_reviewed_mixed_ctx128.fth`、`qwen3.5-0.8b_e4m3_reviewed_sensitive12_ctx128.fth`，位于原模型目录；已有文件未覆盖。48 投影方案为 `qwen3.5-0.8b_e4m3_bf16residual_ctx128.fth`。

正常生成各提示固定输出 32 tokens，Reset 后热运行 3 次。下表为热 TTFT 中位数 / decode 平均单 token 毫秒；诊断禁捕获不影响这条正常生成路径。不同提示的首次运行复用计划程度不同，首次与热运行原始数据均保留。

| 提示 tokens | BF16 TTFT / decode ms | 原 E4M3 | 校准 + 敏感 12 BF16 |
|---:|---:|---:|---:|
| 15 | 52.91 / 7.541 | 110.55 / 11.422 | 76.01 / 11.176 |
| 30 | 62.65 / 7.695 | 82.01 / 11.374 | 92.85 / 11.447 |
| 52 | 57.03 / 7.923 | 82.84 / 11.742 | 86.55 / 11.289 |

现有 BF16 的热 decode 约 7.5–7.9 ms/token，原 E4M3 约 11.4–11.7 ms/token。当前实现上 FP8 压缩没有转化为小批量解码加速，尚未完成 kernel 级归因。下一步应先 profile 小 M 投影、类型转换和同步成本，再决定融合或专用 GEMV；这些是待验证方向，不能作为已证实瓶颈。

敏感性报告的逐通道二次幂权重 scale 分析，相对逐 Tensor 的加权 MSE 比约为 1.000，本次没有支持投入逐通道 kernel 的收益证据。该分析不表示运行时已支持逐通道量化，也不能推论其他格式或模型没有收益。

原始报告、逐 token scores、最终 logits、范围与样本、模型和程序 SHA256 均位于 `models/mainstream_validation/qwen_eval_20261023/`；更正后的汇总为 `reviewed/comparison.json`，作废记录见 `calibration_review_audit.json`。旧 benchmark TSV 中 `captures` 实际表示 CUDA Graph launch 次数；新程序已将表头修正为 `cuda_graph_launches`。显存为全卡运行后采样使用量，未提供模型独占峰值或跨框架性能比较。

最终构建 `qwen_eval`、`qwen_demo`、`unit_tests` 成功。C++ 全量 **697/697 通过**，包含最初列出的三个失败项和新增观察缓存回归；Python 评测/两个导出器共 **21/21 通过**。直接 CLI 重用已有运行产物目录会被拒绝，避免样本追加混入旧数据。验证日志保存在 `reviewed/verification/`。


## 2026-10-23 FP8 解码性能跟进

本次继续使用 RTX 4070、同一 E4M3 模型和量化策略。当前环境为 CUDA 11.8、cuBLASLt 11.11.3（`cublasLtGetVersion()=111103`）；Ada 原生 FP8 GEMM 需要 cuBLAS 12.1 Update 1 或更新版本。当前矩阵乘法实际使用 BF16 Tensor Core 回退：每次调用先将 FP8 激活和完整权重转换为 BF16，再执行 GEMM，最后写回输出。小批量解码反复转换权重是额外成本；FP8 文件更小不能据此保证推理更快。

逐节点同步的运行时 profile 中，MatMul 和 Cast 累计耗时最高；这些同步会改变执行行为，不能当作正常 CUDA Graph 路径的吞吐。当前 WSL 环境中的 Nsight Systems 2022.4.2 导出失败（`No GPU associated to the given UUID`），没有可用的 GPU 时间线，不能据此宣称得到 kernel 耗时占比。

`src/kernel/cuda/kernel_io.cuh` 在 SM89 及以上使用 CUDA SDK 的 FP8 编解码，去掉原参考实现中的归一化循环等开销；旧架构保留原实现。有限值采用舍入到最近偶数和有限饱和，Inf/NaN 写入继续使用原语义，读取 NaN 保持规范化。CUDA 11.8 的 SDK 在 SM89 仍使用软件转换，SM90 才启用原生 FP8 转换指令；本次没有验证其他 GPU，也没有实现 Ada 原生 FP8 GEMM。使用已有模型文件即可生效，无需重新导出。

最终采用 SDK 路径。自定义 FP32 位运算转换虽然通过数值验证，但本次正常生成测量未超过 SDK 路径，因此未保留。专用 GEMV 原型在不同投影形状上也未表现出稳定收益，未纳入运行时。

同一提示集固定输出 32 tokens，Reset 后热运行 3 次，decode 为 callback 间隔均值；以下是本次优化前和 SDK 版本的首次对照，数值为 ms/token。结果只适用于这组短提示和当前设备，不是标准 benchmark 或跨框架性能结论。

| 提示 tokens | 优化前 | SDK 编解码 | 吞吐提升 |
|---:|---:|---:|---:|
| 15 | 11.184 | 10.853 | 3.04% |
| 30 | 11.497 | 11.190 | 2.74% |
| 52 | 11.859 | 11.354 | 4.45% |

最终重新构建后，相同配置再测一次：

| 提示 tokens | 最终 decode ms/token | 相对本次基线吞吐提升 | 最终热 TTFT ms |
|---:|---:|---:|---:|
| 15 | 10.678 | 4.73% | 73.71 |
| 30 | 10.826 | 6.20% | 85.06 |
| 52 | 10.860 | 9.20% | 81.59 |

两次 SDK 测量有时间波动，单次差值不能解释为固定 kernel 收益；当前证据支持小幅改善，未承诺固定加速比。`comparison.json` 保留各次结果、模型和二进制指纹。

三组提示的生成 token IDs 全部一致。短文本 645、长文本 3070 个目标 token 的困惑度相对原 E4M3 增幅为 0；逐 token `scores.tsv` 及 14 份文档最终 logits 均逐字节一致。这表示本次转换优化在已测样本中保持原 E4M3 结果，不消除 E4M3 相对 BF16 的量化误差。当前 FP8 解码仍慢于前述 BF16 约 7.5–7.9 ms/token，精度和此环境解码性能优先时继续推荐 BF16。

补充测试覆盖全部 65536 种 BF16 编码、两种 FP8 格式全部 256 种编码、相邻有限 FP8 值的 FP32 舍入中点与左右邻域、16384 组确定性 FP32 位模式，以及三个 scale。检查有限值/Inf 编码、NaN 类别、signed zero 和反量化值。

性能与质量原始报告位于 `models/mainstream_validation/qwen_fp8_perf_20261023/`。`baseline` 为原参考转换，`native_codec` 是 SDK 转换的历史目录名，在 SM89 上仍是软件；`bit_codec` 与 `bit_quality_*` 是未采用的位运算实验。`final_codec` 为最终构建复测，`final_quality_short`、`final_quality_long` 为最终精度复核；这两项的 3715 个评分 token 与全部 14 份最终 logits 已再次确认逐字节一致。`runtime_profile_before.log` 为同步 profile，`nsys_failure.log` 保留失败的时间线导出记录。

最终构建成功，C++ 全量 **699/699 通过**，包含两项新转换测试和最初报告的三个失败项。SM80、SM90（开启 cuBLASLt）及 SM89（关闭 cuBLASLt）的独立转换 TU 编译通过，属于编译检查，未在相应其他设备运行。构建、回归、编译日志保存在本次产物的 `verification/` 目录。


## 2026-10-23 FP8 回退权重缓存

针对上述重复转换，CUDA BF16 Tensor Core 回退支持有限预算的只读 FP8 权重缓存。首次在正常执行中将权重解码到 BF16 并完成同步，随后重复投影直接复用该设备缓冲。激活和输出量化边界保持原有数值计算。只缓存 `immutable` 权重，源设备分配的 generation、格式、元素数或 scale 改变后重新准备。普通 FP8 输出与融合 BF16 输出的两条回退路径均接入；原生 FP8 GEMM 不创建 BF16 缓存。

缓存由执行上下文持有，并通过已有共享设备权重的分配身份复用；跨上下文仅发布已完成转换的数据。最终权重注册释放、上下文销毁或关闭缓存时释放对应引用。CUDA Graph 捕获只使用预热后的本地命中，未命中仍捕获原临时转换，不在捕获中分配或同步。捕获图独立保活所引用的 BF16 Storage，直到图销毁。因此预算约束当前活动缓存，已捕获图保留的旧缓冲在调低预算后仍可能驻留。

这是额外显存换解码时间的运行时选项：原 FP8 设备权重仍保留，模型文件和量化策略不变，无需重新导出。通用 `SessionOptions::fp8_weight_cache_limit` 默认 0；Qwen 的 `QwenLoadOptions::fp8_weight_cache_bytes` 默认 256 MiB，序列计划和上下文增长沿用加载选项。Qwen demo 和评测 CLI 均支持 `--fp8-weight-cache-mb`，单位实际为 MiB，0 关闭缓存。预算耗尽或可选缓存分配失败时沿用原临时转换路径。

例如为 RTX 4070 上的现有 E4M3 模型显式提供 1 GiB 缓存：

```sh
./build_new/bin/qwen_demo \
  --model models/llm/qwen3.5-0.8b/qwen3.5-0.8b_unified_fp8e4m3_bf16islands_ctx128.fth \
  --tokenizer models/llm/qwen3.5-0.8b --backend cuda \
  --prompt '介绍一下你自己' --max-context 256 --max-new-tokens 128 \
  --fp8-weight-cache-mb 1024
```

正常生成比较使用同一二进制、模型、提示集和上下文 256，三组提示分别含 15、30、52 tokens，各固定生成 32 tokens、Reset 后热运行 3 次。首轮数据如下，decode 为 ms/token：

| 缓存预算 MiB | 15 token 提示 | 30 token 提示 | 52 token 提示 | 全卡采样最大使用 MiB |
|---:|---:|---:|---:|---:|
| 0 | 10.900 | 10.964 | 10.918 | 2419.5 |
| 256 | 9.947 | 10.307 | 10.580 | 2693.5 |
| 1024 | 8.279 | 8.423 | 8.286 | 3407.5 |

1 GiB 档相对关闭缓存的吞吐提升约 30%–32%，全卡采样增量约 988 MiB；256 MiB 默认档提升约 3%–10%。全卡采样包含 CUDA 上下文和其他进程，不能当作缓存自身大小或模型独占峰值。收益取决于权重规模、可缓存比例及设备；当前 CUDA 11.8 的回退仍没有使用 Ada 原生 FP8 GEMM，BF16 仍是此环境下更省转换开销的选择。

三档生成 token IDs 全部一致；1 GiB 档对照原 E4M3 的短文本 645、长文本 3070 个目标 token，`scores.tsv` 和全部 14 份最终 logits 均逐字节一致。新增回归覆盖捕获图少一个权重转换 kernel、两种 FP8 格式的预算/数据/scale 更新、跨上下文共享和源上下文释放、首次捕获不发布未执行转换、关闭缓存并释放权重后旧图继续重放。

原始结果位于 `models/mainstream_validation/qwen_fp8_cache_20261023/`，`cache_0`、`cache_256`、`cache_1024` 为首轮比较，`quality_short`、`quality_long` 为质量复核，`verification/quality_byte_comparison.json` 记录逐文件字节比对和 SHA256。首个转换 kernel 回归在旧实现上实际失败，修复后通过；相关日志保存在 `verification/`。


最终构建每档增加到 5 次热运行后复测：

| 缓存预算 MiB | 15 token 提示 decode ms | 30 token 提示 decode ms | 52 token 提示 decode ms |
|---:|---:|---:|---:|
| 0 | 10.889 | 10.949 | 10.919 |
| 256 | 9.755 | 10.225 | 10.204 |
| 1024 | 7.869 | 8.339 | 8.427 |

1 GiB 档本次吞吐提升约 30%–38%，各档输出 IDs 均一致；两轮测量均保留，不能承诺固定加速比。最终数据与指纹见 `final_comparison.json`、`final_cache_*`。重新构建及 C++ 全量 **704/704 通过**，Python 评测/导出器 **21/21 通过**；`git diff --check` 无空白错误。缓存关闭/释放及捕获边界已只读审查，无阻塞问题。
