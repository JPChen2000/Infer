# 编译计划、存储与执行会话

## 使用入口

已有的模型导入和 StaticGraph 构建方式保持兼容。需要多个独立请求时，构建和优化一次，再创建会话：

```cpp
#include "core/execution_plan.h"

// graph 已绑定输入、权重，且 Build / ApplyPasses 成功。
std::shared_ptr<const feather::CompiledGraph> compiled;
auto status = feather::CompiledGraph::Compile(graph, &compiled);
if (!status.ok()) return status;

std::unique_ptr<feather::ExecutionSession> first, second;
status = compiled->CreateSession(&first);
if (!status.ok()) return status;
status = compiled->CreateSession(&second);
if (!status.ok()) return status;

status = first->BindInput("input", request_tensor);
if (!status.ok()) return status;
status = first->Run();
if (!status.ok()) return status;
std::shared_ptr<feather::Tensor> output;
return first->ReadOutput("output", &output);
```

CompiledGraph 保存固定 shape 的描述、优化后的节点顺序、选定实现的工厂、可重绑定的参数快照和内存计划；不持有可变 Kernel 实例或设备上下文。编译后可以销毁源 StaticGraph。初始化输入被复制成快照，映射权重保持共享映射；普通可写权重在编译时复制成只读快照。

每个 ExecutionSession 独立创建 Kernel、输入输出、激活存储、线程池和后端上下文。不同会话可以并发，同一会话的绑定、执行和修改由调用者串行化。权重 Buffer 只读共享，各会话使用自己的 Storage 别名组；mutable_data 写入时复制整个别名组，保留本会话的视图关系。BindInput 将数据复制到会话自己的输入存储，保持已有 view 的关系；输入 shape、dtype、layout 必须与编译计划一致。

ClassificationRunner、Yolov5Runner 和 QwenRunner 默认使用这一编译与会话入口。会话创建后锁定图结构，禁止通过 runtime() 添加节点、清空图、修改输出或重新 Finalize；线程配置也固定。输入绑定同时校验完整量化参数。即使调用者通过可变 Tensor 句柄改写描述，运行图也会在执行前校验编译时约定的形状、类型、布局和容量，拒绝继续使用失效的计划。旧 RuntimeGraph 在结构或线程配置变更后必须重新 Finalize，RunStatus 会拒绝执行尚未成功初始化的图。

## 静态图与旧接口

StaticGraph 的构图和优化阶段采用延迟分配：Tensor 元数据与逻辑容量立即可用，激活存储在实际数据访问时才分配。形状控制子图需要计算具体值时，会按需物化。这个阶段保留现有 Operator 的形状推导接口；绑定结果是 DeferredKernel 配方，实际后端 Kernel 在 lowering 或会话实例化时才创建。编译期控制值计算按需创建 Common 实例。

GraphLowering::Lower 接受 const StaticGraph，复制 Kernel 实例及其参数，不再通过 DetachKernel 消耗源图，可以重复 lowering。此兼容接口继续引用调用者绑定的 Tensor，因此两个旧 RuntimeGraph 并不保证激活隔离；独立请求应使用 CompiledGraph::CreateSession。

## Storage 和内存计划

Tensor 是 shape、dtype、layout、量化参数与 offset 的 view；Storage 负责共享 Buffer、容量、设备位置、延迟分配及共享 mutation version。Clear/Reset 只改变该 view 的引用。主机和 CUDA 分配均通过 Storage 管理实际所有权；CUDA 的有效性和同步信息由所属 BackendContext 的副本缓存管理。

MemoryPlan 按共享 Storage 合并别名的生命周期，保留外部输入、常量、编译期控制值和可见输出的必要寿命。一个节点的输入和输出不复用同一空间；只有前一生命周期完全结束后，后续值才可以使用其槽位。arena 槽位大小按 64 字节取整，实际对齐遵循底层 Buffer 分配器。

默认的串行主机会话应用 arena 计划。图并行、CUDA 或显式设置 reuse_memory=false 时，使用独立分配。重用内存的会话不保留中间值的历史内容，只保证图输出在下一次执行前有效。后续如改变线程模式，必须重新创建会话，不能绕过 SessionOptions 修改计划的执行前提。

编译排序和串行执行统一采用最小就绪节点索引，确保分支图的实际执行顺序符合内存生命周期；不能改成 FIFO 就绪队列而继续使用当前计划。回归覆盖两条独立分支的交错就绪及多次输入重绑。

SessionOptions::host_memory_limit 限制会话可变 Tensor 的主机存储，在分配 arena 前检查；只读共享权重另计，不包含 Kernel 自行申请的主机打包缓冲或编译快照。memory_usage() 返回创建会话时规划的可变主机存储、共享权重和 arena 字节数，便于比较复用开关；它不跟踪后续写时复制、Kernel 内部打包和动态工作区带来的变化。StaticGraph::SetWorkspaceLimit 将预算传入所有构图/融合中的候选筛选；SessionOptions::device_workspace_limit 则限制 CUDA 临时工作区同时借出的总容量，按分配器取整后的容量计费。两者默认不限；临时块释放后归还预算，缓存池保留的块不算借出，CUDA/cuBLAS/cuDNN 自有内部内存不计入该临时工作区预算。

StatefulSession 的状态绑定复用同一套校验和更新实现。有内存计划时，同形状状态复制到持久输入，避免交换 arena 槽位破坏下一次运行的生命周期；兼容运行图可以交换存储，append 模式按指定轴更新。QwenRunner 已调用该通用状态更新逻辑。

## 后端边界

RuntimeGraph 和 GraphLowering 通过 BackendContext 进行进入上下文、同步、主机读写标记、持久化、回收、状态更新与执行捕获，不包含 CUDA runtime 头文件或调用。CUDA 适配实现位于 src/kernel/backend.cc；其他后端可实现同一接口，并通过 RuntimeGraph 的上下文构造参数或 SessionOptions::backend_factory 注入，无需修改通用执行器。

BackendContext::Capabilities 按设备声明支持情况、是否使用主机内存、是否允许图并行和是否适用当前激活复用计划。同步、状态推进和内存复用均使用这些能力，未注册支持的设备会被拒绝，不能把所有非 CUDA 设备视作 CPU。后端传输及执行准备中的异常转为带节点上下文的 Status，图并行工作线程也遵循该约定。

ExecutionCapture 管理捕获结果的生命周期及提交。QwenRunner 保留双缓冲选择和 logits 等模型语义，实际捕获、实例化、提交、销毁由后端负责。低层 CUDA cache API 和 RuntimeGraph::CudaContext 仍保留为兼容入口。

## Kernel 与 Pass 契约

KernelRequest 传递设备、dtype、layout、Tensor 描述、节点属性来源和 workspace 上限。RegisterCandidate 支持候选实现的能力谓词和显式优先级；不满足条件的候选不会被选用。默认回退 Common，StaticGraph::SetFallbackPolicy(kRequireRequested) 可禁止后端回退。已有 CUDA INT8 严格策略保持一致。编译计划保存选中的工厂，后续注册变化不改变已编译的实现。

自定义 Kernel 可重写 Supports 做绑定后检查。SetParamOwner 记录参数实际类型；内置 Kernel 通过 ParamAs<T> 校验后访问。参数 schema 的 VisitTensors 明确列出 Tensor 字段，编译和会话实例化时按映射重新绑定，缺失映射会失败。原始 SetParam(void*) 继续用于低层兼容，不用于编译快照。

CreateDefaultPassManager 只包含通用 Pass；CreateQwenPassManager 与 CreateYoloPassManager 显式加入模型专用优化。Pass 可以声明 Requires 和 SupportsDevice；PassManager 检查前置依赖，并在每个 Pass 后验证活跃图的生产者、来源、输出和无环约束。Status 包含失败 Pass 或运行节点的上下文；整数返回值保留为兼容包装。

## 当前边界

当前计划针对固定 shape，改变 shape 需要重新编译。算子的独立纯形状函数、符号 shape、跨设备复制节点的编译期调度、CUDA arena/workspace 全局规划、多设备并行和自动代价模型尚未实现。当前架构为这些能力提供明确扩展点，不将它们描述为已有功能。

## 内存基线

`execution_plan_benchmark` 构建 32 个 Add、每个值含 65536 个 FP32 元素的 Common 图，对复用开关分别预热、执行 100 次并校验全部输出。它输出创建会话时的存储预算和平均时延，可直接运行构建目录的 `bin/execution_plan_benchmark`。

2026-10-21 本机 Release 构建的可变 Tensor 主机存储从 8,650,752 字节降至 786,432 字节，下降约 90.9%；两者共享权重均为 262,144 字节，复用 arena 为 524,288 字节。这些存储数字不代表进程 RSS 或 GPU 峰值显存；时延需在空闲系统上重复测量，不能由此推断实际模型的加速幅度。

## 测试资源配置

Python 导入测试使用 PATH 中的 python3，也可通过 FEATHER_TEST_PYTHON 指定安装了 numpy、onnx、onnxruntime 的解释器。YOLO Demo 测试可通过 FEATHER_YOLOV5_MODEL 指定已有模型；可选模型资源缺失会明确跳过，模型存在但执行失败则仍然报错。本机本轮使用独立的 `/tmp/infer-validation-python` 环境及仓库已有的 yolov5n_fp32.fth，无需修改系统 Python 或复制模型到固定路径。

回归见 test/execution_plan_test.cc；测试覆盖源图销毁、重复 lowering、共享权重的写时复制、多会话并发、别名输入重绑、生命周期复用与并行退回独立分配、参数类型错误、能力约束、Pass 校验和通用状态更新。

## 后续修复验证（2026-10-21）

- CPU 全目标构建通过；直接运行完整测试套件，540 项中 493 项通过、47 项因无 GPU 或可选模型缺失跳过，无失败。包含真实 YOLOv5 ONNX 导入与 ONNX Runtime 输出对照，以及 FP32 Runner 推理；另行验证了 NHWC FP16 模型入口。
- Common-only Debug 构建通过，架构与安全测试 29/29 通过。
- CPU 项目代码（包括 Common/x86 Kernel、模型加载与 Runner）全量启用 ASan/UBSan；同一完整套件 493 项通过、47 项跳过，无 sanitizer 或泄漏报告。该检查发现并修复了 bundled stb JPEG 写入器有符号位缓冲左移的未定义行为。
- CUDA/cuBLASLt/cuDNN 按 RTX 5090 的 sm_120 构建通过；关闭设备后的完整主机回归 602 项中 514 项通过、88 项跳过，无失败。此结果不表示设备计算测试通过。
- 真实 GPU 诊断使用匹配已加载模块的用户态库，可以识别两张 RTX 5090，但两卡上下文创建均报显存不足；GPU 计算、多会话并发和捕获回放测试仍需释放显存后执行。详情与运行入口见 [CUDA 运行与排障](cuda_runtime.md)。
- 新增回归覆盖分支生命周期顺序、可选空输入、结构和 Tensor 描述变更、量化参数、权重别名组写时复制、主机内存及 CUDA 工作区预算、其他加速设备的状态策略、并行传输异常。

## 首轮架构验证

此处记录第一轮架构改造的验证；后续修复与 CUDA 环境状态见 [CUDA 运行与排障](cuda_runtime.md)。

- CPU 全目标构建通过；所选回归共 520 项，479 项通过、41 项跳过。
- Common-only Debug 库构建通过；架构与安全测试共 21 项全部通过。
- 对图执行、编译计划、状态更新、内存与模型加载源文件启用 ASan/UBSan，21 项测试全部通过，无 sanitizer 报告；静态库中的其他后端代码未全量插桩。
- CUDA（含 cuBLASLt/cuDNN）构建通过；相关 65 项测试中 55 项通过，10 项因无可用 GPU 跳过。当前机器驱动与库版本不匹配，未完成 GPU 实机执行验证。
- CPU 回归排除了依赖外部 real_models、direct_export、onnx_import 的测试及缺少 models/yolov5n.fth 的 LoadFth 测试；其余跳过主要来自 CUDA 或可选模型资源。
