# CMake 模块化与动态库 (DLL) 后端隔离设计

## 1. 为什么需要后端 DLL 隔离设计？

在本项目中，`ggml` 启用了 **动态库加载模式 (`GGML_BACKEND_DL=ON`)**。这意味着：
* `ggml` 的 CPU 后端、CUDA 后端、SYCL 后端在编译时各自独立生成独立的动态库（例如 `ggml-cpu.dll`、`ggml-cuda.dll`）。
* 在运行时，上层的 `ggml` 引擎会通过 `ggml_backend_load` 动态寻址加载这些二进制文件。

### 静态链接的致命缺陷：
如果我们把扩展算子（如 `ggml_ops_ext`）设计成静态库并直接打入 `tts.dll`：
1. **CPU 侧**：调用 `ggml-cpu` 声明的系统检测函数（如 `ggml_cpu_has_avx2`）在静态链接时会因为找不到符号而报 `LNK2019`。
2. **CUDA 侧**：调用 `ggml-cuda` 内部的 `ggml_cuda_error` 和 `ggml_cuda_set_device` 会报 `LNK2019`，因为这些符号都在动态加载的 `ggml-cuda.dll` 里，无法进行静态链接。
3. **SYCL 侧**：同理，依赖 `ggml-sycl` 导出的底层加速逻辑。

---

## 2. 后端隔离 DLL 的架构设计 (像 GGML 一样)

仿照 `ggml` 的后端隔离设计，我们将扩展算子库 `ggml_ops_ext` 重构为**动态库分离模式**。

### A. 库与组件拓扑：

1. **`ggml_ops_ext.dll` (接口与图构建层)**：
   * 包含 `src/ops/ops.cpp` 以及 `src/ops/core_ops/` 下的构图代码。
   * 它为应用层提供标准 C/C++ 接口（如 `ggml_ops_ext_conv_1d`、`ggml_ops_ext_mish`）。
   * 它不包含任何具体的底层硬件计算代码，仅负责将算子节点组装进 GGML 计算图。

2. **`ggml_ops_ext_cpu.dll` (CPU 计算层)**：
   * 对应 `src/ops/ops-cpu/`。
   * 包含所有的 CPU 优化计算（Mish AVX2/512、Gated SIMD、Conv Transpose 1D L1 Cache 累加）。
   * 编译时显式链接到 `ggml-cpu` 导入库，以便解析 `ggml_cpu_has_avx2` 等符号。

3. **`ggml_ops_ext_cuda.dll` (CUDA 计算层)**：
   * 对应 `src/ops/ops-cuda/`。
   * 包含所有 CUDA kernel 实现。
   * 编译时显式链接到 `ggml-cuda` 和 `cublas`，以便解析设备设置和矩阵乘法符号。

4. **`ggml_ops_ext_sycl.dll` (SYCL 计算层)**：
   * 对应 `src/ops/ops-sycl/`。
   * 编译时显式链接到 `ggml-sycl`。

---

## 3. 模块化 CMake 结构

我们在 `src/ops/` 下重构目录，让每个后端拥有自己独立的 `CMakeLists.txt`：

```
src/ops/
├── CMakeLists.txt              # 主接口层 CMake，添加各个子文件夹并生成 ggml_ops_ext.dll
├── core_ops/                   # 构图构建代码
├── ops-cpu/
│   ├── CMakeLists.txt          # 生成 ggml_ops_ext_cpu.dll，管理 CPU 编译优化标志
│   ├── conv_transpose_1d.cpp
│   ├── layer_norm.cpp
│   ├── mish/                   # 模块化 Mish 子文件夹
│   └── gated_tanh_sigmoid/     # 模块化 Gated 子文件夹
├── ops-cuda/
│   ├── CMakeLists.txt          # 生成 ggml_ops_ext_cuda.dll，管理 CUDA/cuBLAS 依赖
│   └── ...
└── ops-sycl/
    ├── CMakeLists.txt          # 生成 ggml_ops_ext_sycl.dll，管理 Intel oneAPI DPC++ / MKL / oneDNN 依赖
    └── ...
```

### 这一架构的收益：
* **符号自动解析**：由于各后端 DLL 显式链接到对应的 `ggml` 后端 DLL，操作系统加载器（Loader）在运行时会自动在依赖链中定位并解析所有导出的符号，完美解决 `LNK2019` 错误。
* **极速增量编译**：修改某一后端的优化实现（例如改动 CUDA 里的卷积 Kernel），只需重新编译并热替换对应的 `ggml_ops_ext_cuda.dll`，完全不需要重新链接主工程或其他后端，对于大型项目调试非常高效。
* **低内存占用**：运行时仅加载与当前运行硬件匹配的 DLL。如果用户在纯 CPU 环境下运行，则 CUDA/SYCL 的算子 DLL 根本不会被加载到内存中。
