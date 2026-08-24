# Infer 工程基线与核心边界收敛设计

## 1. 背景与目标

Infer 已经具备 `ModelDesc -> StaticGraph -> GraphLowering -> RuntimeGraph` 的两阶段执行主线，并覆盖 Common、x86、CUDA 等 kernel 后端。当前主要风险不在单个算子功能，而在工程边界：构建目标过度集中、图和 runtime 重复保存状态、kernel 参数依赖 operator 生命周期、注册过程隐式且可能重复、错误处理混用返回值/异常/进程退出。

本阶段目标是建立可持续演进的工程基线：明确依赖方向，收敛所有权，统一错误边界，保证每个重构增量都能独立验证和回滚。不修改模型文件格式，不改变算子数学语义，不以一次性重写核心运行时为目标。

## 2. 方案选择

采用渐进式迁移方案。每个阶段只改变一个主要风险面，并保留编译和运行闭环。

不采用一次性重写，因为它会同时影响图语义、CUDA 同步、模型 demo 和 kernel 生命周期，回归定位成本过高。也不采用只治理 CMake 的方案，因为它无法解决 runtime 所有权和重复状态问题。

## 3. 架构与依赖边界

第一阶段只建立四个物理 CMake target，逻辑模块仍由目录和源文件表达：

- `infer_core`：Tensor、Dim、Memory、图/算子/kernel 基础接口以及线程和日志基础设施。
- `infer_backend`：Common、x86、CUDA kernel 实现，按探测结果条件编译。
- `infer_model`：Model IO、WeightStore、LayoutTransform。
- `infer`：兼容聚合库，链接上述三个 target；现有 demo 和外部调用方继续依赖它。

依赖方向如下：模型层不依赖具体 backend；图层只依赖 operator 和 kernel 抽象；operator 通过显式 `BuildContext{device, data_type, layout}` 选择 kernel；runtime 只拥有执行所需的 kernel、Tensor handle、`ValueId` 和调度元数据，不拥有 `OpBase`。

名字只在模型导入、调试、profile 和公开查询边界保留。runtime 内部逐步迁移到连续的 `ValueId`，并通过边界映射支持 `GetTensor(name)`。

## 4. 所有权与注册机制

kernel 参数必须由 kernel 或独立的 `KernelArgs` 对象拥有。kernel 不得借用 `OpBase` 内部参数的裸地址；完成参数独立拥有后，删除 `RuntimeNode::owner` 以及仅用于生命周期转移的 `AttachKernel/DetachKernel/HasKernel` 运行时接口。

算子和 kernel 统一使用显式注册协议：每个实现提供普通 `RegisterXxx(...)` 函数，由中央 register-all 入口按固定顺序调用。删除静态 lambda、宏静态对象和重复的 `Ensure*` 初始化路径。builder 接收显式构建上下文，删除 thread-local backend 选择状态。

## 5. 错误处理契约

新增 `include/core/status.h`，定义 `StatusCode`：`kOk`、`kInvalidArgument`、`kNotFound`、`kShapeMismatch`、`kUnsupported`、`kBuildFailed`、`kExecutionFailed`、`kInternal`；`Status` 保存 code 和诊断 message。

核心 graph/model/runtime 入口逐步从裸 `int32_t` 迁移到 `Status`。kernel `compute()` 第一阶段保留 `int32_t`，由 runtime 统一转换。库代码禁止 `std::exit`；日志只记录，不结束进程。参数错误、未知 op、缺失 tensor、shape 不匹配和未实现能力必须返回明确失败状态。未实现的 runtime load stub 直接删除。demo 层负责将 `Status` 转为命令行退出码。

机器判断只依赖 `StatusCode`，message 只用于诊断；错误上下文应包含节点名和 value 名（如果可用）。

## 6. 测试与质量门禁

单元测试覆盖 Status、Tensor/Memory、registry 错误路径、重复 producer、缺失输入、环依赖、节点删除后的索引一致性，以及 lowering 后 runtime 不持有 `OpBase`。

图级测试保留现有 graph、pass、pipeline 测试，并增加最小 Add 和 MatMul/Relu 图，覆盖 Common 与 x86 fallback。CUDA 可用时运行同一图的 CUDA 路径，不可用时由 CTest 跳过而不是配置失败。

构建门禁至少包括 CPU+OpenMP、CPU 无 OpenMP、CPU debug 和可选 CUDA。CTest 统一注册；依赖外部模型或大文件的测试标记为集成测试，不进入默认单元测试集合。demo 只链接 `infer`。

## 7. 迁移阶段

1. 记录 CPU 构建、测试基线和公共 API 引用清单。
2. 拆分 CMake target，保留 `infer` 聚合 target。
3. 删除无调用者 API/stub，统一 registry 初始化，补 registry 测试。
4. 引入 Status，先覆盖 graph/model/runtime 边界，再迁移 demo。
5. lowering 引入 ValueId；保留名字映射和公开查询 API。
6. 独立 kernel 参数所有权，删除 RuntimeNode owner 和相关转移接口。
7. 执行完整测试矩阵，更新架构文档和迁移说明。

每个阶段必须通过针对性测试和完整 CPU 构建后才能进入下一阶段。模型格式和算子数值语义保持不变；内部头文件允许不兼容调整，已确认无仓库调用者的旧 API 可删除。

## 8. 验收标准

- 源码目录不再被已有 build 目录作为构建前提。
- CPU 配置下 `ctest --output-on-failure` 全部通过。
- demo 和测试只依赖兼容聚合 target `infer`。
- 未知 op、缺失 tensor、非法图和执行失败均返回可断言的 StatusCode。
- runtime 生命周期不依赖 StaticGraph/OpBase 的存活。
- Common、x86、CUDA backend 的选择由显式上下文决定，不依赖 thread-local 隐式状态。
- YOLO、分类和 Qwen 的最小回归图在重构前后保持输出形状和数值容差。
