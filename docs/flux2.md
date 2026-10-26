# FLUX.2 Klein 4B 文生图

当前入口支持 **FLUX.2 Klein 4B 的四步蒸馏版**，batch=1，导出时固定分辨率与文本长度，提供 FP32 / BF16 的 Common、CUDA 路径。运行时为纯 C++，不依赖 PyTorch 或 Diffusers。尚未支持 Klein base、9B、FLUX.2 dev、图像编辑或 CFG。

## 准备权重

导出器使用原生 BFL 键名，不能直接输入 Diffusers 的 `transformer/` 或 `vae/` 转换权重：

| 组件 | 来源 | 文件 |
| --- | --- | --- |
| 去噪网络 | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 根目录 `flux-2-klein-4b.safetensors`，约 7.75 GB |
| VAE | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 根目录 `ae.safetensors`，约 336 MB；按源仓库要求申请访问 |
| 文本编码器 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 未量化 HF snapshot，包含三个 `model-*.safetensors`、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` |

三个组件的源权重合计约 15 GiB；另外预留导出文件空间。模型权重遵循各自源仓库的许可证。Qwen3.5 或量化版 Qwen3 不能替代上述文本编码器。

## 导出

Python 依赖仅用于转换和参考验证，建议安装在独立虚拟环境中：

```sh
python -m pip install numpy ml_dtypes safetensors torch
python tools/export_flux2_model.py \
  --denoiser /path/to/flux-2-klein-4b.safetensors \
  --vae /path/to/ae.safetensors \
  --text-encoder-dir /path/to/Qwen3-4B \
  --output-dir models/flux2-klein-4b \
  --height 512 --width 512 --text-length 512 --dtype bf16
```

宽、高必须是 16 的正整数倍。改变分辨率、文本长度或精度需要重新导出。导出器逐个读取权重，生成 `text_encoder.fth`、`denoiser.fth`、`vae_decoder.fth` 并复制 tokenizer 文件。默认使用 BF16；`--dtype fp32` 可用于数值诊断，但权重空间约翻倍。

文本编码器拼接 Qwen3 第 9、18、27 层隐藏状态；去噪网络使用 FLUX.2 四轴 RoPE 和联合注意力；VAE 解码包含潜变量反归一化与解包。

## 构建和运行

按 [README](../README.md) 准备 Conan 1.x 和 CMake 依赖。Qwen3 tokenizer 还需要 ICU 的 `uc`、`i18n` 开发库（Ubuntu / Debian 可安装 `libicu-dev`），用于 Unicode 正则分词和 NFC 规范化；CMake 自动检测。缺少 ICU 时可使用预计算 embeddings，但原始 Qwen3 提示词入口会明确报错。以下 CUDA 架构值 `89` 适用于 RTX 4070，请按设备调整：

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DFEATHER_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_UNITTEST=ON
cmake --build build-cuda --parallel 4
build-cuda/bin/flux2_demo \
  --model-dir models/flux2-klein-4b \
  --backend cuda --seed 42 --steps 4 \
  --prompt 'A small red panda reading a book under a tree, watercolor' \
  --output flux2.png
```

CPU 可使用 `--backend common` 或 `host`，相应构建可关闭 CUDA。显式选择 CUDA 时，缺少 CUDA 实现会报错。Runner 依次加载并释放文本编码器、去噪网络和 VAE，降低同时驻留权重的需求；CUDA 会话仅按需将输出同步到主机，避免累积全部中间激活的主机副本。实际峰值内存和显存仍取决于分辨率、文本长度和后端临时空间。

默认采样为官方分辨率相关时间步调度的四步 Euler；`--steps` 可调整步数。潜变量采用 FP32 累加，各组件边界按导出 dtype 转换。初始噪声使用原生 `mt19937` 和 Box–Muller；相同 seed 与 PyTorch 的噪声不同，FP32 累加也与官方 BF16 原地采样存在舍入差异，因此不承诺与官方输出逐像素相同。

提示词使用 Qwen3 关闭思考的聊天模板，超长输入从尾部截断，按 tokenizer 配置填充。C++ 接入见 `demo/flux2_runner.h`：`Load()` 后调用 `Generate()`；已有完整固定长度隐藏状态可调用 `GenerateFromEmbeddings()`。同一 runner 的调用需串行执行。

## 验证范围

当前验证覆盖缩小但完整的 Qwen3 → FLUX.2 → VAE 网络：各组件与 PyTorch 数值对照，FP32 / BF16、Common / CUDA，以及完整 C++ PNG 输出、同 seed 确定性和错误路径。额外使用官方 BFL Python 结构及匹配的合成权重核对去噪网络、四轴位置编码和 VAE。合成网络验证结构与执行正确性，不代表真实权重出图质量。

2026-10-23 已下载原生 Klein 4B、FLUX.2 VAE 和 HF Qwen3-4B 权重，逐文件校验源仓库 SHA256，并成功导出 BF16、512×512、文本长度 512 的三个 FTH。当前工作区的输出位于 `models/flux2-klein-4b/`，原始权重位于 `models/flux2-source/`；`export_info.json` 记录导出文件尺寸和 SHA256，`../flux2-source/sources.json` 记录源文件版本及校验值。这些本地模型文件由 Git 忽略。

首次真实 CUDA 执行因会话未启用延迟主机同步，将中间激活逐个回传并保留，进程最大 RSS 约 14.3 GiB，随后被主机 OOM killer 终止。此失败属于执行实现缺陷，不能据此判断显存不足或模型无法在这台机器运行。

修复后，在同一台 RTX 4070 12 GiB 显存、约 16 GiB 主机内存的机器上，BF16、512×512、文本长度 512、四步采样已完成两次真实出图。提示词为 `A small red panda reading a book under a tree, watercolor`，seed 为 42。复跑耗时约 15.1 秒（包含组件加载），系统记录主机 RSS 峰值约 8.22 GiB、进程 swap 为 0；每秒采样的整卡显存峰值约 8.83 GiB（包含约 1.06 GiB 桌面基线，并非精确进程峰值）。输出为 `models/flux2-klein-4b/sample.png`，记录见同目录 `sample_fixed_metrics.json` 和 `sample_attempt_fixed.log`。已检查图片包含树下读书的小熊猫及水彩风格；这只验证该配置和提示词，尚未评估其他分辨率或系统性的画质指标。

复现组件和流水线检查（构建目录不同时设置环境变量）：

```sh
export FEATHER_FLUX2_PROBE="$PWD/build-cuda/bin/flux2_component_probe"
export FLUX2_COMPONENT_PROBE="$FEATHER_FLUX2_PROBE"
export FEATHER_FLUX2_DEMO="$PWD/build-cuda/bin/flux2_demo"
python tools/flux2_graph_test.py
python tools/flux2_text_export_test.py
python tools/flux2_export_test.py
python tools/flux2_vae_export_test.py
python tools/flux2_pipeline_test.py
```

默认强制执行 CUDA 对照；纯 CPU 环境可设置 `FEATHER_TEST_CUDA=0`。该设置只省略 CUDA 组合，不能用于声称 CUDA 已通过验证。
