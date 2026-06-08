# Infer 开发路线图

## 1. 路线图目的

这份路线图回答一个核心问题：

**从当前仓库状态出发，下一步应该优先做什么，才能把项目稳步推进成一个真正可扩展的推理引擎。**

当前项目已经有了一条正确的主链路：

`ModelDesc -> StaticGraph(Op DAG) -> GraphLowering -> RuntimeGraph(Kernel DAG)`

这意味着现在最重要的事，不再是继续往 `RuntimeGraph` 里堆逻辑，而是围绕这条主链路补齐算子层、后端层和后续图优化层。

---

## 2. 总体开发原则

### 2.1 先守住分层，再扩功能

后面无论加多少算子，都不应该再回到：

- 中央 `if/else` 分发
- `RuntimeGraph` 直接创建具体 `Op`
- 图优化和运行时执行混在一起

正确边界必须固定住：

- `StaticGraph`：语义图、shape inference、静态优化、融合前后图变换
- `GraphLowering`：把静态图下沉为 runtime 可执行节点
- `RuntimeGraph`：只执行 lowered kernel DAG

同时要固定住 backend 分层：

- `src/kernel/common`：通用 kernel 基线
- `src/kernel/x86`：x86 指令集优化版本
- `src/kernel/cuda`：CUDA backend 专用实现

后端绑定遵循统一规则：

- 先查本机/目标设备对应 backend
- 查不到再回退到 `COMMON`

### 2.2 先做可验证闭环，再做高级能力

不要一开始把重心放在：

- CUDA kernel 大量铺开
- 大模型专用算子
- INT4 量化
- autotune

当前最缺的是：

- 更高性能的 `Conv2D / MatMul / Gemm` kernel
- 更真实的多精度与量化抽象
- CUDA 显存、stream/event 与 DAG 调度抽象

### 2.3 每一阶段都必须有可运行验收模型

建议里程碑模型：

- 阶段 1：`Input -> FC -> Output`
- 阶段 2：`Input -> Conv2D -> ReLU -> FC -> Output`
- 阶段 3：带常量折叠/死节点删除的小型 CV 子图
- 阶段 4：支持 `FP16/INT8` 的基础子图
- 阶段 5：CUDA 版 CV 子图
- 阶段 6：最小大模型 block

---

## 3. 当前阶段判断

### P0：已经建立的基础

当前已经落下来的关键架构点：

- `Op` 是静态图语义节点，不是花瓶
- 权重导入后先构建 `StaticGraph`
- 算子构建通过 `OperatorRegistry` 完成，不走中央 `if/else`
- kernel 绑定发生在静态图构建阶段
- `RuntimeGraph` 只保存 lowered runtime node 并执行
- `Common / X86 / CUDA` 已经按统一注册表接入 `FP16 / FP32` 常用算子
- `KernelDispatcher` 会记录 kernel 的实际绑定 backend，`RuntimeGraph` 可识别 CUDA 请求退化到 `COMMON/X86` 的节点
- CUDA 图执行到 CPU fallback 节点时，RuntimeGraph 会在节点边界统一处理必要的 D2H/H2D cache 同步
- `SigmoidMulFusionPass` 与 `YoloDecodeFusionPass` 已经在静态图阶段落地

这部分是后面所有工作的地基，不能再回退。

### P1：下一步必须优先推进

现在最应该补的是：

- 针对 YOLOv5 当前瓶颈继续优化 `Conv2D`
- 给 `CUDA` 增加显存 allocator、workspace、stream/event 依赖管理
- 继续做 CUDA 算子本体优化，fusion pass 后续统一规划，不作为当前主要提速手段
- 继续补 `x86` 目录里的高频 AVX 加速版本
- 保持数值正确性、图级测试、demo/benchmark 闭环

### 4.1 现阶段 benchmark 口径

为了让后续 x86 优化有稳定参照，当前 benchmark 口径固定为：

- warmup 与 measure 分离
- 默认输出 min/max/avg
- profiling 只统计测量阶段的运行时节点
- 图级执行默认保持串行，避免和 kernel 内部并行互相干扰

### P2：中期扩展

- `INT8`
- `INT4`
- CUDA multistream runtime
- CUDA memory planning
- 更完整的外部模型导入

### P3：中长期目标

- 大模型推理算子体系
- KV cache
- attention / rope / rmsnorm
- 动态 shape
- 多设备调度
- autotune / kernel selection

---

## 4. 下一步最应该做的事

如果只看“现在立刻开始做什么”，最应该聚焦的是：

### 目标：把 `Common -> X86/CUDA` 这条后端演进路径从“能跑”推进到“可优化”

原因很直接：

- 当前 Host/CUDA kernel 已经支持“目标 backend 优先、`COMMON` 兜底”
- 这给 `ARM` 运行通用 kernel、`x86` 运行 AVX kernel、`CUDA` 运行 GPU kernel 留出了统一接入点
- 如果现在还把通用实现继续塞进 `x86`，后面 backend 扩展会越来越乱

所以当前最优先的工程动作应该是：

1. 对 YOLOv5 实测热点继续优化 `Conv2D`
2. 逐步把性能敏感算子迁移到 `src/kernel/x86/` 下实现 `AVX` 特化
3. 给 `CUDA` 补显存管理、workspace、event 依赖和多 stream 调度
4. 保持 `Op` 侧只做查表绑定，不掺入 backend 细节

---

## 5. 分阶段开发建议

## 阶段一：完成 Common / X86 后端分层

### 目标

让通用实现和平台特化实现彻底解耦，固定内核绑定规范。

### 当前已完成的部分

1. 新增 `src/kernel/common/` 存放通用 CPU kernel
2. Host 侧通过本机设备信息选择优先 backend
3. `KernelDispatcher` 支持“精确 backend 优先，`COMMON` 兜底”
4. 大部分 `Op` 已改成通过统一 helper 完成 kernel 绑定
5. CUDA kernel 已按算子文件拆分并注册到 dispatcher

### 接下来需要完成的内容

1. 清理仍然带有 `x86` 语义的旧命名和日志
2. 在 `src/kernel/x86/` 下逐步增加 AVX 实现
3. 给每个高频算子补 backend 选择与正确性测试
4. 建立 CUDA benchmark 与 profile 口径，避免 profile 同步误读

### 阶段验收标准

- `Common` 基线能在 `x86` 和 `ARM` 上跑通
- 请求 `X86` kernel 时可优先命中特化实现，否则自动回退 `Common`
- 新增 backend 不需要改 `RuntimeGraph` 或中央 `if/else`

---

## 阶段二：补齐 Common 基础算子集

### 目标

把当前通用 kernel 扩展成可支撑主流 CV baseline 的公共子集。

### 推荐优先顺序

1. `MatMul/Gemm`
2. `Sigmoid`
3. `Pooling`
4. `Reshape`
5. `Concat / Split`

### 实现要求

每个算子都应该具备：

- `OperatorRegistry` builder
- `CheckShape`
- `InferOutputShapes`
- `Common` kernel
- 单算子数值测试
- 图级执行测试

### 阶段验收标准

- 跑通更完整的 CV 子图
- 所有核心算子有单测
- 新算子不需要改中央分发逻辑

---

## 阶段三：开始做 x86 AVX 加速

### 目标

把性能敏感算子从 `Common` baseline 中分离出来，在 `src/kernel/x86/` 下做真正的指令集优化。

### 第一批建议

1. `Conv2D`
2. `Gemm / MatMul`
3. `Add / Mul`
4. `Pooling`

当前已完成的第一批 `X86 AVX2`：

- `Add`
- `Mul`
- `ReLU`
- `MatMul`
- `Gemm`

### 原因

- 这些算子是 CPU 侧最主要的热点
- 做成 `X86` 特化后，可以直接复用现有绑定逻辑
- 有利于后续对比 `Common` 与 `X86` 的性能收益

### 阶段验收标准

- `X86` kernel 命中时性能优于 `Common`
- 数值结果与 `Common` 保持一致
- fallback 路径持续可用

---

## 阶段四：优化 CUDA backend

### 目标

在已接入的 CUDA kernel 之上补齐真正可扩展的 GPU runtime 能力。

### 推荐顺序

1. `CUDA Conv2D` 性能优化
2. `CUDA MatMul/Gemm` 性能优化
3. `CUDA` 显存 allocator、workspace 与 tensor 生命周期
4. 多 stream/event 依赖调度

### 需要补齐的能力

- GPU tensor / allocator 抽象
- CUDA event dependency
- device-resident Tensor 设计
- Host -> CUDA lowering 与数据搬运；当前已支持 CUDA 节点退化到 CPU 节点时的 D2H/H2D 边界同步
- 同样遵循“目标 backend 优先、`COMMON` 兜底”的统一分发思想

### 当前已完成的 CUDA 算子本体优化

- `Conv2D` 增加 `1x1 pointwise` 与 `depthwise/grouped depthwise` CUDA 快路径
- `MatMul / Gemm / FC` 切换到 `16x16 shared-memory tiled` CUDA kernel
- CUDA 输出 tensor 在设备分配阶段统一标记 dtype，避免 fallback 同步和 demo 输出读取时依赖 UNKNOWN dtype

### 仍有优化空间

- `Conv2D` 还需要继续补 `3x3 stride1/stride2` 专用路径、im2col+GEMM 或 cuDNN/cuBLAS 接入策略
- `FP16 MatMul/Gemm/Conv2D` 目前仍以 FP32 accumulator 的普通 CUDA kernel 为主，还没有 Tensor Core / WMMA 路径
- `Softmax` 仍是一行一个线程的串行 axis 规约，长 axis 下需要 block-level reduction
- `Resize / Pool / Transpose / Slice` 仍有访存合并、向量化 load/store、shared-memory tile 的优化空间

### 阶段验收标准

- 至少一个 CV 子图可在 CUDA backend 跑通
- backend 扩展不破坏现有 Host 路径
- 有 CPU/GPU 结果一致性测试

---

## 阶段五：接入 CUDA 后端

### 目标

把后端能力从“只有 x86”扩展到“x86 + CUDA”。

### 需要先补的基础设施

1. `DeviceContext`
2. CUDA allocator
3. H2D / D2H copy
4. stream 管理
5. Tensor 所属设备语义

### 然后再补的 kernel

1. `MatMul/FC`
2. `Conv2D`
3. `ReLU`

### 阶段验收标准

- 至少一个小型 CV 子图可在 CUDA 上运行
- CPU/GPU 结果误差可控
- dispatcher 可选不同 backend

---

## 阶段六：转向大模型能力

### 目标

在 CPU/CUDA 基础引擎稳定后，再补大模型所需的专用算子与运行时。

### 第一批应该支持的能力

- `MatMul`
- `RMSNorm`
- `RoPE`
- `Attention`
- `KV Cache`
- `Softmax`
- `Sampling`

### 需要新增的系统能力

- token 级推理调度
- cache 生命周期管理
- 大权重分片加载
- 更适合 LLM 的模型格式接入

---

## 6. 建议的近期里程碑

### 里程碑 M1

**主题：静态图优化起步**

完成标准：

- pass manager 建立
- 常量折叠和死节点删除可运行

### 里程碑 M2

**主题：CPU 基础算子扩充**

完成标准：

- `MatMul/Gemm + Sigmoid + Pooling` 接入
- 新算子均走注册表和 lowering

### 里程碑 M3

**主题：简单融合落地**

完成标准：

- 至少一个融合 pass
- 融合后 runtime node 减少

### 里程碑 M4

**主题：多精度起步**

完成标准：

- `FP16` 路径打通
- 至少 1 到 2 个算子支持低精度

### 里程碑 M5

**主题：CUDA 起步**

完成标准：

- CUDA runtime 抽象建立
- 至少一个图在 GPU 上可运行

---

## 7. 当前最不建议优先做的事情

### 7.1 回到 `RuntimeGraph` 中央分发

这是最不应该再做的事。新算子、融合、backend 扩展都不应该通过改 `RuntimeGraph` 的 `if/else` 来完成。

### 7.2 直接做大模型推理

原因：

- 静态图优化层虽然已经有骨架，但还没有足够多的真实优化 pass
- CPU 基础算子还不全
- 多精度和 CUDA 都没落地

### 7.3 一开始就大规模铺 CUDA kernel

原因：

- 设备抽象还不完整
- lowering 还没有 backend-specific 分支

### 7.4 过早推进 INT4

原因：

- 工程复杂度高
- 对现阶段收益不如 `FP16 / INT8`

---

## 8. 一句话结论

**从整体项目开发来看，下一步最应该做的是：围绕已经建立好的 `StaticGraph -> GraphLowering -> RuntimeGraph` 主链路，把 `StaticGraph` 的 pass/优化框架补起来，然后继续扩 CPU 基础算子。**

只有这一步站稳以后，再做：

- 算子融合
- 多精度
- CUDA
- 大模型支持

才会顺，而且不会频繁返工。
