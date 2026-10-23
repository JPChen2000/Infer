# YOLO 检测模型

统一入口 `yolo_demo` 接收带检测头元数据的 Feather 模型，覆盖官方 YOLOv6、YOLOv8、YOLOv10、YOLO11 和 YOLO26 的 nano 目标检测检查点。当前适配范围为 COCO 80 类、batch 1、固定正方形输入、NCHW、FP32，默认输入尺寸为 640。分割、姿态、旋转框、动态尺寸、多 batch 和低精度导出不在本次范围内。

## 导出官方模型

在已配置 PyTorch、torchvision、NumPy、ONNX 的 Python 环境中，从仓库根目录运行：

```sh
python tools/export_yolo_models.py \
  --families yolov6 yolov8 yolov10 yolo11 yolo26 \
  --size 640 --layout nchw
```

工具默认下载官方预训练 nano 检查点，使用 CPU 和一个 PyTorch 线程导出静态 ONNX opset 13，再转换为带元数据的 FTH v4。YOLOv6 使用 Meituan YOLOv6 `0.4.0` 检测版本的源码和 `yolov6n.pt`；其源码保存在 `/tmp/infer-yolov6-0.4.0`。其他四个模型使用 Ultralytics `8.4.183` 和官方 assets `v8.4.0` 检查点。工具通过 `pip --no-deps --target /tmp/infer-yolo-export-deps` 准备该版本的 Ultralytics；已有环境需要提供其运行依赖。`--ultralytics-deps` 和 `--yolov6-source` 可调整临时目录。

YOLOv6 的官方导出脚本加载时使用 `fuse=True`，并对 RepVGG 块调用 `switch_to_deploy()` 完成重参数化。当前 YOLOv6n ONNX 包含 35 个 `rbr_reparam` 卷积，不含 `rbr_dense`、`rbr_1x1`、`rbr_identity` 训练分支或 BatchNormalization 节点。

`--variant` 选择官方模型规格，默认为 `n`。所有选中的家族必须提供该规格，否则工具在下载前报错。

| 家族 | 可选择的官方规格 |
| --- | --- |
| YOLOv6 0.4.0 | `n,s,m,l` |
| YOLOv8、YOLO11、YOLO26 | `n,s,m,l,x` |
| YOLOv10 | `n,s,m,b,l,x` |

例如 `python tools/export_yolo_models.py --families yolov10 --variant b` 生成 `models/yolo/yolov10/yolov10b.pt`、`.onnx` 和 `.fth`。本次实际导出与运行记录针对 `n`；其他规格提供导出选项，尚未在这里验证其完整图转换与运行。非 nano 导出的 provenance 使用检查点名，例如 `yolov10b.provenance.json`；YOLOv6 非 nano 导出日志同样使用检查点名前缀，保留现有 nano 记录。

导出产物为：

```text
models/yolo/yolov6/yolov6n.pt   yolov6n.onnx   yolov6n.fth
models/yolo/yolov8/yolov8n.pt   yolov8n.onnx   yolov8n.fth
models/yolo/yolov10/yolov10n.pt yolov10n.onnx  yolov10n.fth
models/yolo/yolo11/yolo11n.pt   yolo11n.onnx   yolo11n.fth
models/yolo/yolo26/yolo26n.pt   yolo26n.onnx   yolo26n.fth
```

每个目录另有 `provenance.json`，记录官方下载地址、检查点与 ONNX 的 SHA256、导出器版本、实际输入输出形状、算子清单和检测元数据。YOLOv6 的官方导出日志保存在 `export.log`。这些真实模型已经生成；`models/` 被 Git 忽略，其他工作区通过上面的命令重新获取。模型文件与源码各自遵循上游许可。

只准备 ONNX 可添加 `--skip-convert`；已有 ONNX 可用以下命令重新校验并转换，避免重新导出：

```sh
python tools/export_yolo_models.py --reuse-onnx
```

## 检测头契约

以下是上述 640 输入模型实际导出的输出。`N` 为候选框数量，`C` 为每个候选的字段数量。

YOLOv10/26 最多输出 300 个候选；较小输入下，官方检测头可能限制为可用候选数量，输出为 `[1,N,6]`（`0<N<=300`），例如 64 输入为 `[1,84,6]`。

| 家族 | 官方检查点 | 实际输出 | 字段格式 | 布局 | 应用 NMS |
| --- | --- | --- | --- | --- | --- |
| YOLOv6 | `yolov6n.pt` | `[1,8400,85]` | `cx,cy,w,h,objectness,80 class scores` | `bnc` | 按类别执行 |
| YOLOv8 | `yolov8n.pt` | `[1,84,8400]` | `cx,cy,w,h,80 class scores` | `bcn` | 按类别执行 |
| YOLOv10 | `yolov10n.pt` | `[1,300,6]` | `x1,y1,x2,y2,score,class_id` | `bnc` | 跳过 |
| YOLO11 | `yolo11n.pt` | `[1,84,8400]` | `cx,cy,w,h,80 class scores` | `bcn` | 按类别执行 |
| YOLO26 | `yolo26n.pt` | `[1,300,6]` | `x1,y1,x2,y2,score,class_id` | `bnc` | 跳过 |

YOLOv6 的最终置信度为 objectness 乘以类别分数；YOLOv8/11 直接使用类别分数。YOLOv10/26 保留官方 one-to-one 检测头的最终选择结果，导出时不添加 NMS 包装；运行时按置信度过滤并恢复图片坐标。`--iou-thresh` 仅影响需要 NMS 的模型。

FTH v4 元数据显式记录以下字段，运行时验证它们与输出描述是否一致：

| 元数据键 | 类型 / 含义 |
| --- | --- |
| `yolo.version` | int64，当前为 `1` |
| `yolo.family` | string：`yolov6`、`yolov8`、`yolov10`、`yolo11`、`yolo26` |
| `yolo.num_classes` | int64，当前官方 COCO 模型为 `80` |
| `yolo.output_format` | string：`cxcywh_objectness`、`cxcywh_scores`、`xyxy_score_class` |
| `yolo.output_layout` | string：`bnc` 或 `bcn` |
| `yolo.nms_required` | int64：`0` 或 `1` |

旧版无此元数据的 YOLOv5 文件继续使用原有检测后处理。家族模型应通过上述导出工具生成 FTH，避免丢失输出契约。

## 运行图片检测

使用 README 的构建流程配置 CPU 或 CUDA 目录后，构建统一入口：

```sh
cmake --build build-cpu --target yolo_demo --parallel 4
build-cpu/bin/yolo_demo \
  --model models/yolo/yolov8/yolov8n.fth \
  --image /path/to/image.jpg \
  --backend host --layout auto \
  --conf-thresh 0.25 --iou-thresh 0.45 --max-detections 300 \
  --output yolov8-output.jpg
```

替换模型路径即可选择表中其他家族。预处理沿用图片 runner 的 letterbox、RGB 转换和 `[0,1]` 归一化；输出框坐标映射回原图，保存带标注的图片。`--backend host` 选择宿主 CPU，`common` 选择通用实现，`x86` 请求 x86 实现。当前家族模型使用 `--layout auto` 或 `--layout nchw`。

CUDA 构建对应的命令为：

```sh
cmake --build build-cuda --target yolo_demo --parallel 4
build-cuda/bin/yolo_demo \
  --model models/yolo/yolo26/yolo26n.fth \
  --image /path/to/image.jpg --backend cuda --layout auto \
  --output yolo26-cuda-output.jpg --profile
```

CUDA 路径允许回退到 Common。当前 FP32 `TopK`、`ReduceMax`，FP32/INT64 `GatherElements` 和 INT64 `Mod`/`Div` 已有 CUDA 实现；索引链上的 INT64 `Expand`、`Unsqueeze` 和 INT32/INT64 互转也在设备端执行。TopK 使用稳定排序，相同分数保留较小的源索引；整数索引处理保持整数精度。上述选择算子的 CUDA 实现支持最多 8 维，TopK 还要求输入元素数不超过 `INT_MAX`，其他类型或超出支持范围时回退到 Common。GatherElements、Mod 和 Div 仍会同步一个很小的错误标志以检查索引越界、除零等错误，输入和结果张量保留在 GPU。

YOLOv6n/10n/26n 的当前导出图在 CUDA 上没有 Common 节点。FP32 NCHW `ConvTranspose` 使用 cuDNN backward-data 接口处理对称 padding，保留 `CUDNN_FMA_MATH` 精度；无 cuDNN、参数不适用或工作区不可用时使用通用 CUDA 内核。通用 CUDA 路径覆盖二维 stride、dilation、非对称显式 pads、group、output_padding 和可选 bias，权重保持 ONNX `[Cin,Cout/group,kH,kW]` 顺序；维度和整数属性需在 int32 范围内，超出时允许 Common 回退。暂不接收 `auto_pad` 和显式 `output_shape` 导出。`--backend cuda` 表示请求 CUDA 后端；编译与运行摘要中的 `cuda_nodes`、`common_nodes` 和 `common_ops` 显示实际节点设备。

## 保留同一输入进行数值对照

以下命令保存预处理后的 FP32 NCHW 输入、原始输出及最终检测结果：

```sh
build-cpu/bin/yolo_demo \
  --model models/yolo/yolov8/yolov8n.fth \
  --image /path/to/image.jpg --backend host \
  --output yolov8-output.jpg \
  --dump-input yolov8-input.bin --json yolov8-result.json
```

JSON 包含 `input_shape`、原图尺寸、letterbox 参数、`raw_output.shape`、`raw_output.data` 和 `detections`，可与同目录 ONNX 通过 ONNX Runtime 对照。参考执行应读取同一份输入 dump，以避免图片解码或缩放差异。YOLOv10/26 的 TopK 在相同分数处可能存在候选顺序差异；最终检测对照应同时检查类别、分数和框坐标。


## 本次实际运行结果

`yolo_demo` 和原有 `yolov5_demo` 已在现有 CUDA 构建目录 `build_new` 编译成功。五个官方 nano FP32 模型分别使用 Ultralytics 的 `bus.jpg`、`zidane.jpg`，在 x86 和 CUDA 上运行，共 20 组推理。ONNX Runtime CPU 读取 Feather 导出的同一份输入进行对照，所有组的原始输出及最终检测框对照通过。

| 模型 | 两张图 × 两个后端 | 最终框坐标最大绝对误差（原图像素） |
| --- | --- | --- |
| YOLOv6n | 4/4 | 0.000306 |
| YOLOv8n | 4/4 | 0.000214 |
| YOLOv10n | 4/4 | 0.000132 |
| YOLO11n | 4/4 | 0.000366 |
| YOLO26n | 4/4 | 0.000549 |

原始输出使用 `atol=1e-3, rtol=1e-3`；最终检测框使用 0.1 像素阈值、分数使用 `atol=1e-3`。v10/26 的完整输出按类别、分数、坐标进行一一匹配，同时核查分数排序；报告保留逐行对照结果。部分低分候选的分数差异约为 `1e-7`，会交换相邻候选顺序。所有候选均参与比较，未按置信度跳过低分输出。

在对照过程中修正了 CUDA FP32 卷积的计算精度：cuDNN 原先允许 TF32，现在 FP32 使用 `CUDNN_FMA_MATH`，FP16 路径仍允许 Tensor Core 计算。此精度策略作用于共享卷积实现，可能影响其他 FP32 模型的吞吐；下面的选择算子优化保留这一策略。原有 YOLOv5 FP32 模型通过统一入口完成 CUDA 图片推理。

可复用以下命令对任意一组进行数值对照（需 ONNX Runtime）：

```sh
OMP_NUM_THREADS=4 python tools/compare_yolo_onnx.py \
  --model models/yolo/yolo26/yolo26n.fth \
  --onnx models/yolo/yolo26/yolo26n.onnx \
  --image /path/to/image.jpg \
  --demo build_new/bin/yolo_demo --backend x86 cuda \
  --output-dir /tmp/yolo26-compare
```

本工作区汇总保存在 `models/yolo/validation.json`。详细输入、输出、图片和对照报告位于 `/tmp/infer-yolo-compare/<family>/` 与 `/tmp/infer-yolo-zidane/<family>/`。本次执行的是实际模型导出、图片推理和数值对照，未新增或运行单元测试套件；上述结果仅覆盖列出的 nano 检测模型和两张图片。

## YOLOv10n / YOLO26n CUDA 性能优化

原先检测头的 `ReduceMax` 回退到 Common，单节点约 23 ms；两次 Common TopK 合计约 2 ms，选择链还触发 GPU 与 CPU 之间的数据传输和同步。这是两个 nano 模型 `rungraph_ms` 达到 20～30 ms 的主要原因。

补齐上述 CUDA 选择与整数索引算子后，在本机 RTX 4070、batch 1、640×640、FP32、`bus.jpg` 下测得：

| 模型 | 优化前 rungraph_ms 中位数 | 优化后中位数 | 加速 | CUDA / Common 节点 |
| --- | --- | --- | --- | --- |
| YOLOv10n | 32.2864 ms | 4.43091 ms | 7.29× | 241 / 0 |
| YOLO26n | 31.4596 ms | 4.97451 ms | 6.32× | 299 / 0 |

每项取三次独立进程的测量，各进程在加载时执行内置 CUDA warmup，关闭 `--profile`。优化后原始样本分别为 `[4.55987, 4.34450, 4.43091]` ms 和 `[4.83759, 4.97451, 7.70399]` ms；这些是本机观测，GPU 频率和其他负载会影响波动。`rungraph_ms` 现在在 CUDA 推理流执行完成后停止计时，包含 GPU 实际执行及图调度，不包含图片预处理和最终输出下载。优化前计时未显式等待图末尾，Common 回退会强制大部分工作同步，因此对照存在计时边界差异。`--profile` 会同步各节点，增加耗时，只用于定位瓶颈。

完整 CUDA 选择链重新完成 YOLOv10n/26n × `bus.jpg`/`zidane.jpg` 的四组 ONNX Runtime 对照，全部通过；每组比较完整 300 行输出，沿用上面的误差阈值。最终框坐标最大绝对误差为 0.000244 像素。精度与性能汇总保存在 `models/yolo/cuda_selection_validation.json`，详细数据位于 `/tmp/infer-yolo-cuda-selection/`。`yolo_demo` 和原有 `yolov5_demo` 已重新编译。

## YOLOv6n CUDA 转置卷积优化

YOLOv6n 已在导出阶段完成重参数化；此前慢的主要原因是 neck 中两个 `ConvTranspose` 回退 Common，profiling 分别约 16.88 和 17.54 ms。补齐 CUDA 实现后，全图 165 个节点均运行在 CUDA，两个转置卷积的 profiling 耗时分别约 0.18 和 0.33 ms。

在同一 RTX 4070、batch 1、640×640、FP32、`bus.jpg` 下，关闭 profiling，内置加载 warmup 后取三次独立进程测量：

| 模型 | 优化前 rungraph_ms 中位数 | 优化后中位数 | 加速 | CUDA / Common 节点 |
| --- | --- | --- | --- | --- |
| YOLOv6n | 35.6087 ms | 3.28202 ms | 10.85× | 165 / 0 |

优化前样本为 `[34.1965, 35.6087, 38.6297]` ms，优化后为 `[3.10591, 3.28202, 3.69373]` ms。前后均使用等待 CUDA 图执行完成的计时边界；预处理、最终输出下载与 NMS 不计入 `rungraph_ms`。此阶段的 CPU NMS 仍约 6 ms；下面的 CUDA 后处理进一步优化这部分耗时。

最终构建在 `bus.jpg`、`zidane.jpg` 上重新完成同输入 ONNX Runtime CPU 对照，两组均通过原有 `atol=1e-3, rtol=1e-3` 原始输出阈值与检测框阈值。每组检查完整 `[1,8400,85]` 输出，原始值最大绝对误差约 0.01163（大坐标仍满足相对误差阈值），最终检测框最大误差约 0.000306 像素，分数最大误差约 `1.73e-06`。本次实际模型验证覆盖官方 YOLOv6n 使用的对称 padding 转置卷积；其他 CUDA 参数组合未在此做模型验证。汇总位于 `models/yolo/cuda_conv_transpose_validation.json`，详细输入、输出与图片位于 `/tmp/infer-yolov6-cuda-conv-transpose/`。未新增或运行单元测试套件。

## ONNX NonMaxSuppression 算子

ONNX 导入器、Common 和 CUDA 后端现已支持 `NonMaxSuppression`。输入契约为 FP32 `boxes[B,N,4]` 和 FP32 `scores[B,C,N]`，输出 INT64 `[K,3]`，每行依次为 batch、类别、源候选框索引。支持多 batch、多类别，以及 `center_point_box=0` 的 `y1,x1,y2,x2` 和 `center_point_box=1` 的 `cx,cy,w,h`。

可选输入依次是 INT64 `max_output_boxes_per_class`、FP32 `iou_threshold` 和 FP32 `score_threshold`，均接收标量或 `[1]`；默认值均为 0，因此省略 maximum 时输出为空。控制量可在同一会话的不同调用间修改。筛选采用 `score > score_threshold`，抑制采用 `IoU > iou_threshold`；分数相同时保留源索引较小的候选。NaN 分数和非有限坐标会被跳过，正无穷分数可参与排序。

当前采用有容量上限的动态输出：编译时预留 `B*C*N` 行，执行后实际输出形状为 `[K,3]`，不会将填充行作为有效结果暴露。**NMS 必须直接作为图输出，暂不支持其后连接 Gather 等动态后继算子**；导入器和图编译器会明确拒绝该用法。boxes/scores 仍要求静态形状，图内 NMS 暂不支持 FP16 输入。

CUDA 使用稳定 radix sort、候选框位掩码和设备端贪心选择，仅下载计数与最终选中索引。掩码工作区随通过阈值的候选数平方增长；工作区或共享内存不足时返回错误，不会静默截断候选。图内 NMS 的最终索引输出存于主机。

实际验证使用 YOLOv6 的两张图片输出组成 batch 2、8400 框、80 类的真实输入，分别转换两种坐标格式，并在 Common/CUDA 上运行。同一会话的 maximum 按 `300 → 0 → 300` 变化，实际输出为 `16 → 0 → 16` 行；共 12 组结果与 ONNX Runtime 的 INT64 索引逐项一致。该验证覆盖了导入、编译、重复执行和读取动态输出。

## YOLO CUDA 解码与 NMS

`yolo_demo` 和 `yolov5_demo` 的 CUDA 路径现在对需要 NMS 的检测头，在 GPU 上执行候选解码、阈值过滤、按类别 NMS 和结果收集。支持 FP32/FP16 原始输出；普通图片检测仅下载最终选中记录，不再下载完整原始检测张量。请求 JSON 或调用 `LastRawOutput()` 时才同步原始输出。YOLOv5 沿用坐标恢复和裁剪后执行 NMS 的行为；YOLOv6/8/11 在模型坐标中执行 NMS，再恢复图片坐标。YOLOv10/26 的最终检测头继续跳过 NMS。

在 RTX 4070、batch 1、640×640、FP32、`bus.jpg` 下，每项取三次独立进程的中位数，关闭 profiling 和 JSON：

| 模型 | 原 CPU 后处理 | CUDA 后处理 | 加速 |
| --- | --- | --- | --- |
| YOLOv6n | 5.20939 ms | 1.29278 ms | 4.03× |
| YOLOv8n | 5.16344 ms | 1.52143 ms | 3.39× |
| YOLO11n | 5.01063 ms | 1.38908 ms | 3.61× |

这里统计 `postprocess_ms`，包含解码与 NMS。原路径的完整原始输出下载位于该计时区间之前；新路径的计时还包含 GPU 后处理和选中结果下载。每个进程仅执行加载时的图 warmup，首次后处理的工作区初始化也计入上述新路径；未另做 NMS warmup。图推理时间 `rungraph_ms` 单独统计。GPU 频率和其他负载会影响这些本机观测。

接入后，五个 nano FP32 家族模型 × 两张图片共 10 组 ONNX Runtime 对照通过，包含完整原始输出与最终检测；v10/26 按完整候选匹配处理近似同分项的排序差异，沿用前文阈值。最终检测框坐标最大误差约 0.000184 像素。精度、图内 NMS 和性能汇总保存在 `models/yolo/nms_validation.json`；详细图片与输出位于 `/tmp/infer-yolo-cuda-nms/`，图内 NMS 输入与索引位于 `/tmp/infer-nms-real-model/`。未新增或运行单元测试套件。

旧 YOLOv5 FP32/FP16 模型也各运行两张图片。对每次实际输出的同一份原始张量执行 NumPy CPU 解码与 NMS，四组最终结果均与 CUDA 后处理一致，框坐标最大误差约 `4.4e-6` 像素。FP16 的优化前后独立进程原始输出存在数值差异，其中 `zidane.jpg` 的一个最终框相差 0.125 像素，因此不将这组跨进程结果计作通过 0.1 像素阈值；该现象不出现在同原始张量的后处理对照中。相关原始输出、跨进程对照和同输入后处理对照均保存在 `legacy/comparison.json`。
