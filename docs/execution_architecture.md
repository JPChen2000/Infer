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

ExecutionSession 默认独立创建 Kernel、输入输出、激活存储、线程池和后端上下文。`SessionOptions::mutable_state` 可显式绑定调用者持有的同名可变输入 Tensor，使同一请求的形状计划使用同一状态对象；共享这些对象的计划需串行执行。`SessionOptions::immutable_weight_source` 可显式指定另一会话，从中共享同名且描述、量化参数和内容一致的只读常量；GPU 先完成源上传，再共享带所有权的分配。源会话销毁不影响目标会话；共享分配不进入激活复用池，修改时分离。不同会话可以并发，同一会话的绑定、执行和修改由调用者串行化。权重 Buffer 只读共享，各会话使用自己的 Storage 别名组；mutable_data 写入时复制整个别名组，保留本会话的视图关系。BindInput 将数据复制到会话自己的输入存储，保持已有 view 的关系；输入 shape、dtype、layout 必须与编译计划一致。构图和 Pass 阶段实际读取的形状控制 Tensor（如 Reshape shape、Slice 的 bounds/axes/steps）及其传递的数值依赖也保存值快照；编译、绑定和执行分别校验，改变其内容必须重新构图编译。Shape 只依赖输入描述，不冻结输入数据。由非静态子图计算出的形状控制值会明确拒绝，普通 token 与 position 输入不受此限制。

ClassificationRunner、Yolov5Runner 和 QwenRunner 默认使用这一编译与会话入口。会话创建后锁定图结构，禁止通过 runtime() 添加节点、清空图、修改输出或重新 Finalize；线程配置也固定。输入绑定同时校验完整量化参数。即使调用者通过可变 Tensor 句柄改写描述，运行图也会在执行前校验编译时约定的形状、类型、布局和容量，拒绝继续使用失效的计划。旧 RuntimeGraph 在结构或线程配置变更后必须重新 Finalize，RunStatus 会拒绝执行尚未成功初始化的图。

会话同时冻结 Tensor 对象身份及其 Storage 身份/offset，执行、输入绑定和状态重置前校验，拒绝通过 `ShareDataWith`、`ResetStorage` 等途径改变已计划的别名关系。写时复制保留 Storage 身份，因此正常的跟踪写入仍可使用。旧 RuntimeGraph 在 Kernel 绑定后保持 Tensor 句柄，即使线程配置或新增节点使调度失效，`SetTensor` 仍复制兼容输入数据；重新 Finalize 不能把外部改写的存储关系登记为合法关系。通用状态更新器只在完成批次预检后，为实际交换的 Tensor 私下更新存储约定。

## 静态图与旧接口

StaticGraph 的构图和优化阶段采用延迟分配：Tensor 元数据与逻辑容量立即可用，激活存储在实际数据访问时才分配。形状控制子图需要计算具体值时，会按需物化。这个阶段保留现有 Operator 的形状推导接口；绑定结果是 DeferredKernel 配方，实际后端 Kernel 在 lowering 或会话实例化时才创建。编译期控制值计算按需创建 Common 实例。

Build 成功后保存 Tensor 对象和 shape/dtype/layout/量化描述的快照，Compile 与 ApplyPasses 入口拒绝构图后直接替换句柄或改写描述。修改相同 Tensor 的普通输入数据或存储仍可编译；改变句柄或描述需重新 Build。Pass 通过受支持的重建接口同步修改描述与 Kernel，成功后保存新快照。声明的输出 dtype 与算子推导结果冲突时在构建阶段返回错误，避免首次执行改写类型、后续执行才失败。

GraphLowering::Lower 接受 const StaticGraph，复制 Kernel 实例及其参数，不再通过 DetachKernel 消耗源图，可以重复 lowering。此兼容接口继续引用调用者绑定的 Tensor，因此两个旧 RuntimeGraph 并不保证激活隔离；独立请求应使用 CompiledGraph::CreateSession。

## Storage 和内存计划

Tensor 的 `raw_data()` 始终返回只读指针，包括通过可变 Tensor 句柄访问。调用者和 Kernel 写入必须使用 `mutable_data(bytes)` 或类型化 `mutable_data<T>()`，使只读共享存储先执行写时复制，并更新整个别名组的版本。CUDA 下载仅在实际复制时取得可写主机地址，保持非重叠待下载别名的缓存有效性，并使重叠别名中的旧设备副本失效。无法安全保留部分重叠视图的待下载数据时，在复制前拒绝同步；零字节同步不改变缓存状态。

Tensor 是 shape、dtype、layout、量化参数与 offset 的 view；Storage 负责共享 Buffer、容量、设备位置、延迟分配及共享 mutation version。Clear/Reset 只改变该 view 的引用。主机和 CUDA 分配均通过 Storage 管理实际所有权；CUDA 的有效性和同步信息由所属 BackendContext 的副本缓存管理。

MemoryPlan 按共享 Storage 合并别名的生命周期，保留外部输入、常量、编译期控制值和可见输出的必要寿命。一个节点的输入和输出不复用同一空间；只有前一生命周期完全结束后，后续值才可以使用其槽位。arena 槽位大小按 64 字节取整，实际对齐遵循底层 Buffer 分配器。

默认的串行主机会话应用 arena 计划。图并行、CUDA 或显式设置 reuse_memory=false 时，使用独立分配。重用内存的会话不保留中间值的历史内容，只保证图输出在下一次执行前有效。后续如改变线程模式，必须重新创建会话，不能绕过 SessionOptions 修改计划的执行前提。

编译排序和串行执行统一采用最小就绪节点索引，确保分支图的实际执行顺序符合内存生命周期；不能改成 FIFO 就绪队列而继续使用当前计划。回归覆盖两条独立分支的交错就绪及多次输入重绑。

SessionOptions::host_memory_limit 限制会话可变 Tensor 的主机存储，在分配 arena 前检查；只读共享权重另计，不包含 Kernel 自行申请的主机打包缓冲或编译快照。memory_usage() 返回创建会话时规划的可变主机存储、共享权重、从指定会话复用的权重和 arena 字节数，便于比较复用开关；它不跟踪后续写时复制、Kernel 内部打包和动态工作区带来的变化。StaticGraph::SetWorkspaceLimit 将预算传入所有构图/融合中的候选筛选；SessionOptions::device_workspace_limit 则限制 CUDA 临时工作区同时借出的总容量，按分配器取整后的容量计费。两者默认不限；临时块释放后归还预算，缓存池保留的块不算借出，CUDA/cuBLAS/cuDNN 自有内部内存不计入该临时工作区预算。

StatefulSession 的状态绑定复用同一套校验和更新实现。有内存计划时，同形状状态复制到持久输入，避免交换 arena 槽位破坏下一次运行的生命周期；兼容运行图可以交换存储；常量状态源始终复制到可变输入，避免交换改写权重，append 模式按指定轴更新。QwenRunner 已调用该通用状态更新逻辑。`StatefulSession::Advance` 先校验整个绑定批次，再快照可能被其他状态写入覆盖的主机输出，最后提交，保证交叉状态的读取使用更新前的值。重叠状态目标会被拒绝；当前后端尚无设备状态快照接口，因此存在跨绑定设备分配别名的危险批次，以及源与目标设备分配重叠的追加操作，会在任何写入前返回不支持。单绑定入口使用相同的校验与别名处理。

## Qwen 上下文与会话

FTH v4 在原有图和权重描述后附加 `ModelDesc.metadata`；v5 的命名计划仍可加载。当前优化 Qwen 导出器写一个 v4 图，默认序列长度为 1；通过元数据进行上下文和序列特化，不在文件中保存多个形状图。原子导出器和旧 FTH v1–v5 加载路径保留。

上下文使用 `qwen.context.version/default/maximum`；新优化图使用 `qwen.execution.version=2` 和 `qwen.sequence.version/default/maximum`。Tensor 维度、INT64 形状常量和整数属性中的变化元素通过 `(index, offset)` 显式标注，特化分别解析为 `context + offset` 或 `sequence + offset`，不按数值相等猜测形状。序列属性绑定同时支持标量与向量，标量的索引为 0。上下文特化先执行，序列特化合并已有常量覆盖；两者均在编译前完成。`qwen_chunk_offsets` 通过 `qwen.sequence.arange` 标记运行时生成 FP32 `arange(T)`，其 Tensor 描述和覆盖 payload 必须一致，映射权重保持只读。

优化图每个线性 attention 层使用一个 `QwenGatedDeltaSequence`，输入为状态 `[1,H,K,V]`、K/Q `[1,T,H,K]`、V `[1,T,H,V]`、beta/decay `[1,T,H]`，输出最终状态与所有 token 的 core；FP32 算子内部按 token 顺序递推。投影和其他逐 token 子图按块执行。runner 从同一图按需缓存不超过提示块上限的 2 的幂长度计划，余数组合使用较小计划，长度 1 也是这张图的特化。

一个 Qwen 请求持有一组可变状态 Tensor。所有长度计划绑定同一组对象，并共享不可变权重和 CUDA 执行上下文/stream；Kernel、激活和捕获资源按计划保存。KV 原地写入，卷积/recurrent 输出复制到稳定的状态输入，切换长度不传递状态。共享后端缓存按图引用计数，销毁一个计划只释放其引用。计划之间的执行由 runner 串行化。

CUDA 捕获在开始前快照设备有效性、别名、分配池和工作区元数据。成功捕获持有其使用地址的分配租约，避免临时块进入共享池后被其他计划覆盖；销毁捕获后才允许安全复用。捕获失败恢复快照，使普通执行回退不会误认为未执行的上传或写入已经完成。租约与运行图的所有权引用分别管理，销毁一个捕获不释放仍由其他图或 Tensor 持有的分配。

单个编译计划的 shape 和地址固定；运行中的容量增长通过 `QwenRunner::ReserveContext(capacity)` 准备更大特化完成。新状态按新缓存行跨度保留有效 KV 前缀，同时迁移卷积/recurrent 状态与已处理位置，准备成功后再提交会话替换。地址变化使已有捕获和长度计划失效，之后按需重建；扩容失败保留原会话。默认 `context_size=0` 从导出默认容量开始按需自动增长；正数显式容量是应用硬上限，自动与手动申请都受模型配置上限约束。旧模型没有上下文绑定时保持固定容量。

QwenChatSession 保存精确的完整轮次 token，先申请完整历史、新输入、生成预算和结束标记所需容量，再决定是否裁剪。`max_new_tokens=-1` 至少为一个生成 token 留出空间。达到模型上限或显式硬上限后，保留 system 与最近完整轮次，统一 Reset KV、卷积和 recurrent 状态，再重放保留 token。扩容失败保留历史；新输入自身超过可用上限时返回错误。每个 Chat 保存 runner 状态修订号；外部 Consume/Reset/Load/Evaluate/Generate 或另一 Chat 改变状态后，下次有效请求会重置并重放自身精确历史，不能继承其他会话的 KV/recurrent 状态。共享同一 runner 的 Chat 调用必须串行，runner 必须比 Chat 存活更久。Qwen `Load` 失败后 runner 处于 unloaded 状态，后续推理、消费和重置会返回错误，需再次成功加载后使用。

`CachedAttention` 使用可变外部缓存与 position，缓存消费者独占，串行执行并持久保留设备分配，执行前检查块边界；失败可能部分修改 KV，恢复执行前需要 Reset。以上描述是新执行契约，历史测量及其适用范围见 [主流模型推理路径](mainstream_inference.md)。

Qwen 的生成和消费期间，回调重入 `Load`、`Reset`、`Generate`、`ReserveContext` 等公共操作会返回错误，保持外层操作的状态与诊断；无返回值的 profiling 修改在此期间不生效。回调异常转换为错误返回，RAII 保护在所有退出路径释放，Chat 仍可恢复完整历史。`CreateRequest` 拒绝以持有当前 runner 的 unique_ptr 作为输出，避免调用中销毁自身。普通 `Generate` 允许 prompt、stop IDs 与输出向量相同，清空输出前保存所需输入。单个 runner 仍由调用者保证多线程串行使用。

## 后端边界

RuntimeGraph 和 GraphLowering 通过 BackendContext 进行进入上下文、同步、主机读写标记、持久化、回收、状态更新与执行捕获，不包含 CUDA runtime 头文件或调用。CUDA 适配实现位于 src/kernel/backend.cc；其他后端可实现同一接口，并通过 RuntimeGraph 的上下文构造参数或 SessionOptions::backend_factory 注入，无需修改通用执行器。

BackendContext::Capabilities 按设备声明支持情况、是否使用主机内存、是否允许图并行和是否适用当前激活复用计划。同步、状态推进和内存复用均使用这些能力，未注册支持的设备会被拒绝，不能把所有非 CUDA 设备视作 CPU。后端传输及执行准备中的异常转为带节点上下文的 Status，图并行工作线程也遵循该约定。

ExecutionCapture 管理捕获结果的生命周期及提交。QwenRunner 保留双缓冲选择和 logits 等模型语义，实际捕获、实例化、提交、销毁由后端负责。低层 CUDA cache API 和 RuntimeGraph::CudaContext 仍保留为兼容入口。标准 CUDA INT8 算子的输入、输出、临时缓冲和 launch 统一使用当前上下文的缓存、工作区预算和 inference stream，支持与 Conv/Gemm/MatMul 在延迟主机同步模式下串联。局部主机向量生成的辅助元数据先完成上传；此类不具备稳定主机地址的上传在 CUDA 捕获时会明确拒绝，由调用者回退到普通执行。

## Kernel 与 Pass 契约

KernelRequest 传递设备、dtype、layout、Tensor 描述、节点属性来源和 workspace 上限。RegisterCandidate 支持候选实现的能力谓词和显式优先级；不满足条件的候选不会被选用。默认回退 Common，StaticGraph::SetFallbackPolicy(kRequireRequested) 可禁止后端回退。已有 CUDA INT8 严格策略保持一致。编译计划保存选中的工厂，后续注册变化不改变已编译的实现。

算子构造在参数绑定后调用 Supports，只有 kUnsupported 可以继续尝试后续候选或允许的 Common 回退；参数错误和执行错误直接返回，开始执行后不会重选。低层 KernelDispatcher::create 返回待绑定候选，调用者在 SetParamOwner 后通过 KernelBase::FinalizeSelection 完成选择。自定义 Kernel 可重写 Supports 做绑定后检查。SetParamOwner 记录参数实际类型；内置 Kernel 通过 ParamAs<T> 校验后访问。参数 schema 的 VisitTensors 明确列出 Tensor 字段，编译和会话实例化时按映射重新绑定，缺失映射会失败。原始 SetParam(void*) 继续用于低层兼容，不用于编译快照。

OperatorRegistry 的注册表在锁内取得不可变 Builder 的共享所有权，在锁外执行构造函数及销毁被替换的 Builder。并发注册不会释放正在执行的 callable，构造函数及其捕获对象析构期间可以再次注册/创建算子。

CreateDefaultPassManager 只包含通用 Pass；CreateQwenPassManager 与 CreateYoloPassManager 显式加入模型专用优化。Pass 可以声明 Requires 和 SupportsDevice；PassManager 检查前置依赖，并在每个 Pass 后验证活跃图的生产者、来源、输出和无环约束。Status 包含失败 Pass 或运行节点的上下文；整数返回值保留为兼容包装。

## 算子语义与兼容入口

`OperatorSemantics` 集中声明 shape-value 输入、可在构建时求值的控制算子、只读取描述符的求值、可写/外部/独占输入、输出别名、受容量约束的输出与串行要求。扩展算子通过 `RegisterOperatorSemantics` 注册；StaticNode、CompiledGraph 与 RuntimeNode 保存独立快照，注册变化不能改写已有计划。`Shape` 的输入数值和 `Pow` 的指数仍可变。控制值的传递依赖在构建和 Pass 后冻结，CachedAttention 的位置算法检查留在算子中，通用执行器消费声明完成状态预检和写入通知。

`GraphLowering` 先完成同一 CompiledGraph 校验，再复用节点实例化和 Tensor 契约安装，保留原 StaticGraph 的 Tensor 绑定。旧入口同样拒绝描述符和 shape-value 变化，并约束可变状态与终端选择输出；可以重复 lowering，不消费静态 Kernel。

公开 `Tensor::storage()`、`Storage::buffer()` 和底层 `data()` 均提供传递只读访问，写入经 `Tensor::mutable_data` 完成 COW 和版本更新。`Storage::id()` 表示稳定的分配/别名组身份，与写入版本和主机地址无关。CUDA 主缓存以 `(storage id, offset, bytes)` 为键，Tensor 地址只登记使用者；同范围别名共享副本，下载只遍历该 Storage 的局部索引，释放最后一个登记使用者后才可回收。捕获回滚同时恢复副本、索引、使用者和持久标记。

QwenRunner 负责请求编排与生成；`QwenModelResources` 保留共享且只读的原始模型和映射，`QwenExecutionPlan` 保存每个 shape 的会话、Kernel 与捕获，`QwenDecodeState` 保存请求状态、位置和修订号。长度计划不再包含完整 runner。`CreateRequest` 从相同资源创建独立请求，即使模型文件已移除、原请求已销毁仍可使用；请求之间的 KV、recurrent 状态和修订号独立。

x86 BF16/FP8 线性算子的只读权重打包结果可跨请求和长度计划共享。后端弱缓存以物理只读字节视图、dtype、K/N 和转置格式为键，存活的打包结果保留原始 Buffer 所有权；最后一个使用者销毁后释放打包和源权重引用。每个 Kernel 保留自己的源版本检查和可变工作区；仅设置 immutable 标记的可写 Buffer 不进入共享缓存，COW 后重新准备本地权重。FP8 scale 在计算时单独应用，不包含在解码后的打包字节中。

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

## 架构边界重构验证（2026-10-23）

本轮封闭底层 Storage/Buffer 写入路径，统一声明式算子语义和传统 lowering 契约，改用稳定存储视图身份组织 CUDA 副本，拆分 Qwen 资源/计划/请求所有权，并在绑定参数后完成候选选择。全部 CUDA 目标构建通过；完整 762 项测试中 656 项通过、106 项明确跳过，无失败。新增回归涵盖旧入口执行前拒绝、语义快照与融合刷新、后绑定状态别名、候选回退与错误传播、设备副本索引/捕获回滚以及模型文件移除后的独立 Qwen 请求。当前没有可用 GPU，设备计算和真实捕获回放仍需实机验证。

## 已复现问题修复验证（2026-10-23）

修复调度失效后的绑定替换、运行时存储/别名改写、构图与编译间描述不一致、Qwen 回调重入及参数别名、注册表 callable 并发生命周期，并共享 x86 BF16/FP8 只读打包权重。构建和候选检查拒绝推导或实际 MatMul 实现无法生成的输出类型；自定义候选仍可声明支持异构输出。直接图重写只刷新实际重建输出的快照，不能使无关的外部描述改写通过校验。

全部目标（含 CUDA）构建通过。最终完整 797 项测试中 691 项通过、106 项明确跳过，零失败；相关集成回归 132 项中 125 项通过、7 项因无可用 GPU 跳过。原注册表并发 UAF 复现通过 ASan 与泄漏检查。2048×2048 BF16 合成模型的新增 Qwen 请求分配量由约 8.45 MB 降至约 61–67 KB，此测量仅验证该模型的打包结果复用，不代表所有模型的进程内存。当前没有可用 GPU，设备执行未验证。具体证据见 [修复计划与验证记录](superpowers/plans/2026-10-23-confirmed-architecture-fixes.md)。
