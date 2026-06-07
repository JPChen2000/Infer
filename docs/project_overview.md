# Infer 项目总览

## 1. 项目定位

这个项目当前是一个 C++ 推理引擎雏形，目标方向已经明确：

- 支持普通 CV 模型推理
- 支持当前流行的大模型推理
- 兼顾多种数据精度
- 支持计算图构建、算子优化、算子融合、图优化
- 主要面向两类后端：
  - `x86 CPU`
  - `CUDA GPU`

从现有代码来看，项目已经具备了一些关键底座：

- 张量与形状表示
- 内存池
- Kernel 注册与分发机制
- 基础模型存储/加载格式
- 一组已经接入执行主链路的 `Common + FP32/FP16` 常用算子与测试
- 新的两阶段图执行骨架

当前阶段不再适合把所有职责都塞进一个运行时类里。项目现在已经明确采用两阶段图架构：

`ModelDesc -> StaticGraph(Op DAG) -> GraphLowering -> RuntimeGraph(Kernel DAG)`

这也是后续做图优化、算子融合、多后端扩展时最稳的主线。

---

## 2. 当前仓库结构

### 顶层目录

- `include/`: 公共头文件
- `src/`: 主要实现代码
- `test/`: 单元测试
- `docs/`: 架构与路线文档
- `cmake/`: CMake 辅助脚本
- `main.cc`: 简单示例入口
- `CMakeLists.txt`: 构建入口
- `conanfile.py`: 依赖管理配置

### 模块分层

1. **基础数据层**
   - `include/core/tensor.h`
   - `include/core/dim.h`
   - `include/core/memory.h`
   - `src/core/*.cc`

2. **图与执行层**
   - `include/core/static_graph.h`
   - `include/core/graph_lowering.h`
   - `include/core/graph.h`
   - `include/core/operator_registry.h`
   - `include/core/operator.h`

3. **算子与参数层**
   - `src/operator/*.h`
   - `src/operator/*.cc`

4. **后端 Kernel 层**
   - `src/kernel/common/*`：不依赖特定指令集的通用 CPU kernel 基线
   - `src/kernel/x86/*`：只放 `AVX / AVX2 / AVX512` 等 x86 特化实现
   - `src/kernel/cuda/*`：CUDA kernel 与 GPU backend 相关实现

5. **模型格式与权重加载层**
   - `include/model/*`
   - `src/model/*`

6. **测试层**
   - `test/*.cc`

---

## 3. 构建系统与工程组织

项目使用：

- `CMake`
- `Conan`
- `C++17`

当前 `infer` 静态库已经纳入主构建链路的核心模块包括：

- `Tensor / DDim / BufferPool`
- `StaticGraph / GraphLowering / RuntimeGraph`
- `OperatorRegistry`
- `FC / Gemm / Conv2D / ReLU / Sigmoid / Reshape`
- `Common` kernel 基线实现
- 模型 IO 与权重映射

这说明项目已经从“只有零散模块”推进到“有统一执行主线”的阶段。

---

## 4. 核心数据结构

### 4.1 DDim

文件：

- `include/core/dim.h`
- `src/core/dim.cc`

作用：

- 表示 Tensor 维度
- 计算元素总数
- 提供简单切片和 reshape 辅助

这部分已经足够支撑当前最小闭环，但距离大模型常见的动态 shape、stride/view、KV cache 布局还差很远。

### 4.2 Tensor

文件：

- `include/core/tensor.h`
- `src/core/tensor.cc`

作用：

- 封装数据指针、维度、数据类型、偏移量、内存大小
- 提供赋值、共享数据、重置 Buffer 等能力

当前问题也很明显：

- 使用习惯仍然明显偏 `FP32`
- 还没有系统的多精度访问语义
- 缺少更强的视图/布局/变换抽象

### 4.3 Buffer / BufferPool

文件：

- `include/core/memory.h`
- `src/core/memory.cc`

这一层是当前项目最像“推理引擎底座”的部分之一，适合继续扩展到：

- 张量复用
- workspace 管理
- allocator 抽象
- CPU/GPU 分离内存管理

但目前仍然只有 CPU 侧内存管理，没有设备内存抽象。

---

## 5. 图架构与执行框架

### 5.1 StaticGraph：语义图阶段

文件：

- `include/core/static_graph.h`
- `src/core/static_graph.cc`

`StaticGraph` 是现在真正的“构图层”。

它负责：

- 接收 `ModelDesc`
- 管理 graph input/output 对应的 tensor
- 物化 value/tensor map
- 通过 `OperatorRegistry` 创建具体 `Op`
- 在构图阶段完成 shape 检查、输出推导、kernel 绑定
- 预留 `ApplyPasses()` 作为图优化/算子融合入口

也就是说，权重导入后得到的不是 runtime 执行图，而是一个携带 `Op` 的静态图。

### 5.2 OperatorRegistry：去掉中央 `if/else`

文件：

- `include/core/operator_registry.h`
- `src/core/operator_registry.cc`

这里是这次架构调整最关键的一步之一。

现在具体算子不再由 `RuntimeGraph` 通过 `if/else` 创建，而是：

- 每个算子自己注册 builder
- builder 负责解析 `NodeDesc`
- builder 负责构造参数对象
- builder 负责创建 `Op`
- builder 负责完成 shape inference 和 kernel 绑定

这样带来的结果是：

- `RuntimeGraph` 不再依赖具体算子头文件
- 增加新算子时不需要去改中央分发逻辑
- 算子语义构建和运行时执行被真正拆开

### 5.3 OpBase：静态图里的语义节点

文件：

- `include/core/operator.h`

当前 `OpBase` 的职责边界已经明确：

- 表达算子语义
- 持有输入/输出 tensor 绑定
- 做 `CheckShape()`
- 做 `InferOutputShapes()`
- 接收/移交 kernel

这里的 `Op` 不再是花瓶，也不是最终执行图本身，而是静态图优化和 lowering 之前的语义节点。

### 5.4 GraphLowering：静态图到运行时图

文件：

- `include/core/graph_lowering.h`
- `src/core/graph_lowering.cc`

`GraphLowering` 负责把 `StaticGraph` 下沉为 `RuntimeGraph`。

当前实现是最小版本：

- 复制 tensor map
- 遍历静态图中的 `Op`
- 把绑定好的 kernel 从 `Op` 移交到 runtime node
- 生成按执行顺序排列的 runtime node

后续如果要做：

- 融合后 kernel 生成
- backend-specific lowering
- memory planning
- execution scheduling

都应该继续放在这一层扩展，而不是回退到 `RuntimeGraph` 里硬写。

### 5.5 RuntimeGraph：执行期 Kernel DAG

文件：

- `include/core/graph.h`
- `src/core/graph.cc`

现在的 `RuntimeGraph` 已经不再负责：

- 解析 `ModelDesc`
- 创建具体 `Op`
- 通过 `if/else` 选择算子和 kernel

它只负责：

- 保存 lowered runtime node
- 保存运行时 tensor map
- 顺序执行 runtime node
- 暴露图中 tensor 给调用方和测试读取

当前 `RuntimeGraph` 还是顺序执行容器，还不是完整 DAG 调度器，但职责边界已经对了。

### 5.6 Kernel 绑定与后端选择

当前 kernel 绑定规则已经切到“后端优先 + 通用兜底”这条主线：

- `Op` 在静态图构建阶段完成 kernel 绑定
- Host 侧先根据本机信息识别目标设备
  - `x86` 机器优先请求 `DeviceType::X86`
  - `ARM32 / ARM64` 机器优先请求对应 ARM backend
- `KernelDispatcher` 按查表方式选择实现
  - 先查目标 backend
  - 查不到再回退到 `DeviceType::COMMON`

这意味着：

- 当前 `Common` 目录下的 kernel 可以同时服务 `x86` 和 `ARM`
- 后续新增 `x86/AVX` kernel 时，不需要改 `Op` 绑定流程，只需要补充注册表
- 未来新增 `CUDA` backend 时，也复用同一套“精确 backend 优先、Common 兜底”的分发机制

所以静态图和运行时图的边界现在是：

- `StaticGraph(Op DAG)`：携带语义和已绑定 backend kernel 的静态图
- `RuntimeGraph(Kernel DAG)`：经过 lowering 后只保留可执行 kernel 节点的 DAG

---

## 6. Kernel 注册与后端分发

### 6.1 KernelBase 与 KernelDispatcher

文件：

- `include/core/kernel.h`

当前机制：

- `KernelBase` 作为执行单元抽象
- `KernelDispatcher` 按 `DeviceType / DataType / op_type` 分发
- `REGISTER_KERNEL` 完成静态注册
- `Ensure*Registered()` 保证静态库场景下 kernel 可见

这个设计现在不再服务于 `RuntimeGraph` 里的硬编码分发，而是服务于：

- 算子构建阶段的 kernel 绑定
- lowering 前的静态图准备
- 后续不同 backend/dtype 的替换

### 6.2 DeviceType / DataType

文件：

- `include/util/types.h`

枚举层已经为：

- `FP16 / FP32 / INT8 / INT4`
- `COMMON / X86 / ARM32 / ARM64 / CUDA / ASCEND`

预留了方向，当前真正落地的主链路是 `Common` baseline，`X86` 作为宿主优先 backend，未来再逐步补平台特化实现。

---

## 7. 算子层现状

### 7.1 FC

文件：

- `src/operator/fc_op.h`
- `src/operator/fc_op.cc`
- `src/kernel/fc.h`
- `src/kernel/common/fc.cc`

当前这条执行主链路已经不只覆盖 `FC`，同样覆盖：

- `Gemm`
- `Sigmoid`
- `Reshape`

也就是说，`OperatorRegistry -> StaticGraph -> GraphLowering -> RuntimeGraph` 这条链路已经不是单算子验证，而是开始具备了常见图算子的可复用性。

### 7.2 Conv2D

文件：

- `src/operator/conv2d_op.h`
- `src/operator/conv2d_op.cc`
- `src/kernel/conv2d.h`
- `src/kernel/common/conv2d.cc`

当前已经具备：

- `Conv2dOp`
- 当前 `Conv2D` 仍主要走 `Common` kernel
- `Add / Mul / ReLU / MatMul / Gemm` 已经补上第一批 `X86 AVX2` 特化实现
- 静态图注册和运行时 lowering

但还远没到通用卷积，batch/channel/group/layout 都还缺。

### 7.3 ReLU

文件：

- `src/operator/relu_op.h`
- `src/operator/relu_op.cc`
- `src/kernel/relu.h`
- `src/kernel/common/relu.cc`

`ReLU` 已接入统一的静态图构建和 runtime lowering 流程。

---

## 8. 模型格式与权重管理

### 8.1 ModelDesc / GraphDesc / NodeDesc

文件：

- `include/model/model_format.h`

这层负责描述模型语义，不直接等同于运行时执行图。当前它正好对接到 `StaticGraph`，这是更合理的边界。

### 8.2 ModelIO / WeightStore

文件：

- `include/model/model_io.h`
- `src/model/model_io.cc`
- `include/model/weight_store.h`
- `src/model/weight_store.cc`

这一层已经有一些适合继续扩展到大模型场景的特征：

- 自定义模型 metadata
- 权重 blob 存储
- `mmap` shard 映射
- Tensor 视图绑定

但距离主流模型格式和大模型分片体系还差很远。

---

## 9. 测试覆盖情况

当前测试已经覆盖了：

- `BufferPool`
- `Tensor`
- `FC` kernel/op
- `Gemm`
- `Conv2D -> ReLU`
- `Sigmoid`
- `Reshape`
- 模型 IO
- `WeightStore`
- `StaticGraph` 构图
- `GraphLowering`
- `RuntimeGraph` 执行
- `Conv2D -> ReLU -> FC`
- `Gemm -> Sigmoid`
- `Reshape`

这说明现在不只是“单算子能跑”，而是新的两阶段图架构已经有回归保护。

---

## 10. 当前已经具备的能力总结

当前项目已经具备：

1. 基础 `Tensor / Shape / Memory` 抽象
2. 一个简单可扩展的内存池
3. `KernelDispatcher`
4. 统一 `Op` 抽象
5. `StaticGraph -> GraphLowering -> RuntimeGraph` 主链路
6. 最小可用的 `FC`
7. 最小可用的 `Gemm`
8. 最小可用的 `Sigmoid`
9. 最小可用的 `Reshape`
10. 最小可用的 `Conv2D -> ReLU -> FC` 与 `Gemm -> Sigmoid` 子图执行链路
11. 自定义模型格式与权重映射加载机制
12. 一组围绕新图架构的单元测试

也就是说，项目已经从“推理引擎底座原型”推进到了“有明确图分层的执行原型”。

---

## 11. 与目标的差距

差距仍然主要集中在：

### 11.1 CV 模型支持

- `Conv2D` 还很初级
- 常见 CV 算子不完整
- layout、padding、group、pooling 等能力不足

### 11.2 大模型支持

- 没有 attention、rope、rmsnorm、kv cache、sampling
- 没有 token 级运行时
- 没有大模型主流格式接入

### 11.3 数据精度支持

- 类型枚举有了，但计算路径仍然几乎只有 `FP32`

### 11.4 图优化

- `StaticGraph::ApplyPasses()` 仍然是空壳
- 没有常量折叠、死节点删除、融合、内存规划

### 11.5 多后端支持

- `CUDA` 仍然只有雏形代码
- 缺少设备上下文、显存管理、stream 抽象

---

## 12. 建议的后续演进顺序

### 第一阶段：补强静态图优化层

- 实现 `StaticGraph::ApplyPasses()`
- 引入 pass 基类与 pass manager
- 先做常量折叠、死节点删除、简单融合

### 第二阶段：继续补 CPU 基础算子

- `MatMul/Gemm`
- `Sigmoid`
- `Pooling`
- `Reshape`
- `Concat / Split`

### 第三阶段：补多精度与多后端

- `FP16`
- `INT8`
- `INT4`
- `CUDA` runtime 与 kernel

### 第四阶段：转向模型生态与大模型能力

- ONNX 或其他外部模型导入
- 大模型权重格式接入
- attention / kv cache / decoding runtime

---

## 13. 一句话判断当前项目阶段

这个仓库目前已经不只是“算子 + kernel 的松散集合”，而是一个带有明确静态图/运行时图分层的推理引擎原型。

现在最值得继续扩展的基础是：

- `Tensor + BufferPool`
- `OperatorRegistry`
- `StaticGraph`
- `GraphLowering`
- `KernelDispatcher`
- `ModelIO + WeightStore`

而下一步最应该优先补强的是：

- `StaticGraph` 优化框架
- CPU 基础算子集
- 多精度与 CUDA 的抽象边界
