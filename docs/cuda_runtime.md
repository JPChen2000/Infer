# CUDA 运行与排障

CUDA 是否可执行需要分别检查：驱动/用户态库是否匹配、是否能创建上下文、目标架构是否正确，以及实际 Kernel 和模型结果。`cudaGetDeviceCount` 成功只能证明设备枚举成功。

## 本机环境

2026-10-21 检测到两张 RTX 5090，计算架构 `sm_120`，CUDA Toolkit 12.8。已加载的 NVIDIA 内核模块为 `580.173.02`，系统用户态库为 `580.178.04`；直接执行 nvidia-smi 报 Driver/library version mismatch。

已从 NVIDIA 官方下载同版本驱动包，仅提取计算所需的用户态库到 `~/.cache/infer/nvidia/580.173.02/lib`。项目提供包装入口，按已加载模块版本选择匹配库；系统库和其他进程的环境不变：

```sh
tools/with_cuda.sh -- nvidia-smi
tools/with_cuda.sh -- /tmp/infer-fix-cuda-build/bin/cuda_diagnostics
tools/with_cuda.sh -- /tmp/infer-fix-cuda-build/bin/unit_tests \
  --gtest_filter='execution_plan_test.*:runtime_graph_test.*:runtime_safety_test.*'
```

缓存不存在时可使用 `tools/with_cuda.sh --prepare -- command ...`，下载并提取与当前模块匹配的 NVIDIA 用户态库。模块与系统库已经匹配时直接运行原命令，不再注入旧库。该办法解决当前进程的库匹配问题；系统整体的版本统一仍应由主机管理员完成，例如在维护窗口重启，加载已安装的新模块。

## 构建

RTX 5090 使用以下配置；其他设备应选择对应架构：

```sh
cmake -S . -B build-cuda -DFEATHER_ENABLE_CUDA=ON \
  -DBUILD_UNITTEST=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda -j 10
tools/with_cuda.sh -- build-cuda/bin/cuda_diagnostics
```

`cuda_diagnostics` 在每张可见设备上创建上下文、查询显存、分配并传输数据、运行 Kernel，验证结果为 2。任一步失败都会打印错误并返回非零；可通过 CUDA_VISIBLE_DEVICES 指定设备。

## 显存不足

匹配库后可以枚举本机两张 GPU，但检测时现有 root 用户的 vLLM 服务占用了各卡绝大部分显存。最小独立程序的 cudaSetDevice/cuCtxCreate 即返回 out of memory，因此尚不能据此宣称 GPU 计算验证通过。显存剩余量会随服务负载变化；需先提供足够的上下文及测试工作区容量，建议先腾出至少 2 GiB，再按模型规模增加。

这与驱动版本不匹配是两个独立问题。修复匹配后仍需运行诊断，不能仅看 nvidia-smi 成功或 CUDA 构建成功。项目不会为测试而自动终止现有模型服务。
