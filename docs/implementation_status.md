# 架构实施现状与待办事项

> 最后更新: 2026-06-23（bridge 切换完成，旧钩子彻底清理）

## 1. 已完成：旧钩子清理 + Bridge 桥接层纯净化

### 1.1 变更概要

将旧的双钩子机制（`g_ggml_ops_ext_hook` + `g_ggml_custom_op_hook`，定义在 `ggml-backend.cpp`，约 50 行注入）彻底替换为统一的 bridge 单钩子（`g_ggml_bridge_hook`，12 行注入）。

**修改的文件：**

| 文件 | 变更 |
|---|---|
| `src/ops/ops.cpp` | `extern` 声明 → `#include "ggml-ops-ext-bridge.h"`；调用 `ggml_ops_ext_bridge_set_hook` |
| `ggml/src/ggml-backend.cpp` | 删除末尾两个旧钩子定义块（~22 行） |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 旧 extern 块 + 旧循环 → bridge include + bridge 分发 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 同上 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 同上 |
| `src/ops/ops-cuda/CMakeLists.txt` | 添加 post-build 步骤用 `lib.exe` 从 `.def` 生成 import library |

### 1.2 当前 ggml 注入量

总共 **12 行**（4 个文件，每个后端 3 个注入点：include、supports_op、graph_compute dispatch），全部基于单行锚点精确匹配。旧符号 `g_ggml_ops_ext_hook`、`g_ggml_custom_op_hook` 在整个项目中**零残留**。

### 1.1 问题

自定义算子需要注入 ggml 计算图（单次构图 + 后端分发），但 ggml 没有提供扩展接口。之前通过对 ggml 源码 5 个文件做 ~100 行正则 patch 实现，每次 sync 上游都可能静默失败。

### 1.2 方案

新增桥接层（`scripts/ggml-bridge/`），通过 `scripts/apply_ggml_patches.py` 自动注入 ggml：

```
scripts/ggml-bridge/                     ← 本项目源码，跟 git
├── ggml-ops-ext-bridge.h               ← 公开 API（钩子指针 + resource getter 声明）
├── ggml-ops-ext-bridge.cpp             ← include 内联进 ggml.cpp（ggml-base.dll）
├── ggml-ops-ext-bridge-cuda.cu         ← CUDA getter（编译进 ggml-cuda.dll，GLOB 自动）
└── ggml-ops-ext-bridge-sycl.cpp        ← SYCL getter（编译进 ggml-sycl.dll，GLOB 自动）
```

**注入 ggml 的改动（10 行，全部单行锚点精确匹配）：**

| 文件 | 注入 | 内容 |
|---|---|---|
| `ggml.cpp` | include bridge 实现 | `#include "ggml-ops-ext-bridge.cpp"` |
| `ggml-cpu.cpp` | 3 行 | include + supports_op + graph_compute dispatch |
| `ggml-cuda.cu` | 3 行 | 同上 |
| `ggml-sycl.cpp` | 3 行 | 同上 |

**零 CMakeLists 注入**：CUDA/SYCL bridge 利用后端已有的 `file(GLOB *.cu / *.cpp)` 自动编译。

### 1.3 脚本

```
python scripts/apply_ggml_patches.py           # COPY + INJECT + VERIFY
python scripts/apply_ggml_patches.py --revert  # git checkout 还原
```

每次 `git pull` ggml 后执行一次即可。

---

## 2. DLL 后端隔离（架构不变，仅供记录）

### 2.1 架构

```
tts.dll
├── ggml-ops-ext.dll           ← 接口层（计算图构建）
├── ggml-ops-ext-cpu.dll       ← CPU 算子
├── ggml-ops-ext-cuda.dll      ← CUDA 算子
│   └── bridge getter → GetProcAddress → ggml-cuda.dll
└── ggml-ops-ext-sycl.dll      ← SYCL 算子
    └── bridge getter → GetProcAddress → ggml-sycl.dll

ggml.dll / ggml-base.dll       ← ggml 底座（bridge 钩子指针导出）
ggml-cuda.dll                  ← ggml CUDA 后端（bridge getter 编译在内）
ggml-sycl.dll                  ← ggml SYCL 后端（bridge getter 编译在内）
ggml-cpu.dll                   ← ggml CPU 后端
```

### 2.2 ops 层与 ggml 的解耦

- **CUDA**: `ops_cuda_common.cuh` 不再 include `ggml-cuda/common.cuh`。bridge getter 通过 `GetProcAddress` 从 `ggml-cuda.dll` 运行时动态加载。
- **SYCL**: `ops_sycl.h` 同理，通过 `GetProcAddress` 从 `ggml-sycl.dll` 加载。
- **hook 指针**: `g_ggml_bridge_hook` 是 ggml-base.dll 导出的唯一新符号，后端和内联 dispatch 代码直接引用。

### 2.3 cuDNN 状态

- **默认 OFF**（`GGML_CUDNN=OFF`）
- preset `cuda` 中显式设置 `"GGML_CUDNN": "ON"` 启用
- 非 cuDNN 路径使用 im2col + cuBLAS GEMM

---

## 3. 已完成：Conv1D / ConvTranspose1D 内存优化

### 3.1 col_buf 移除（2026-06-23）

**问题**：`core_ops/ops_conv_1d.cpp` 和 `ops_conv_transpose_1d.cpp` 在计算图构建时无条件分配 im2col/col2im 中间张量。

VITS 最后一层上采样：`col_elements = 1 × 512 × 5 × 16000 = 40,960,000 floats = 164 MB`。4-8 层叠加，峰值显存数百 MB 到上 GB。cuDNN 路径用 implicit GEMM 根本不需要这个缓冲区。

**修复**：删除 ggml 层的 col_buf 分配。CUDA handler 内部用 `ops_cuda_alloc` 按需分配（cuDNN 路径零额外显存，非 cuDNN 路径按需 cudaMallocAsync）。

---

## 4. 当前状态：CUDA Conv 实现分析（非 cuDNN 路径）

### 4.1 实现概要

| 文件 | 路径 | 说明 |
|---|---|---|
| `conv_1d.cu` | cuBLAS + im2col | `im2col_1d_kernel` → `cublasSgemm` |
| `conv_transpose_1d.cu` | cuBLAS + col2im | `cublasSgemm` → `col2im_1d_kernel` |

1×1 conv 有 shortcut：直接 GEMM，零 workspace。

### 4.2 待优化点

| # | 问题 | 影响 | 优先级 |
|---|---|---|---|
| 1 | **权重 F16→F32 转换每层都做** | 每层 GPU kernel + cudaMallocAsync（1-5MB） | **高** |
| 2 | `cudaSetDevice` 每层调用 | 极低 | 低 |
| 3 | `ops_cuda_alloc` 每层 alloc/free | 内存碎片 | 中 |
| 4 | im2col/col2im kernel 无 shared memory | 带宽利用不足 | 低 |
| 5 | cuDNN descriptor 每层创建/销毁（cuDNN 路径） | 每层 ~0.1ms | 中 |

### 4.3 优化建议

**最高优先：权重预转换。** VITS 权重总共几十 MB，模型加载时一次性转为 F32，推理时零开销。

**中优先：workspace 池化。** 维护一个 per-stream workspace 池，避免反复 cudaMallocAsync/cudaFreeAsync。

---

## 5. 已知问题

### 5.1 nvcc 链接 `ggml_ops_ext_cuda` 不产出 import library

**现象**：`BUILD_SHARED_LIBS=ON` 时，nvcc 驱动的 MSVC linker 不会为 CUDA 目标生成 `.lib`，导致 `tts.dll` 链接失败 `LNK1181`。

**根因**：CMake 的 `CUDA_SHARED_LIBRARY_LINKER` 规则未像 `CXX_SHARED_LIBRARY_LINKER` 那样自动添加 `/IMPLIB:` 标志。

**修复**：`src/ops/ops-cuda/CMakeLists.txt` 中添加 post-build 步骤，用 MSVC `lib.exe` 从 CMake 自动生成的 `.def` 文件手动创建 import library。

### 5.2 `ggml-sycl.dll` 运行时加载失败（0xC0000135）

**现象**：`ggml-sycl.dll` 作为 MODULE 库被 `ggml-base.dll` 通过 `LoadLibrary` 动态加载时失败（error 126），导致所有依赖 `tts.dll` 的可执行文件启动即崩溃。

**根因**：`ggml-sycl.dll` 链接了多个 Intel oneAPI 运行时 DLL（`tbb12.dll`、`mkl_core.2.dll`、`ze_loader.dll`、`sycl8.dll`、`dnnl.dll` 等）。这些 DLL 的搜索依赖 `AddDllDirectory` API 注册的路径，而 `setvars.bat` 仅设置 `PATH` 环境变量——对于 `LoadLibrary` 加载的 MODULE 库的**依赖解析**，`PATH` 不够。

**验证**：Python 脚本中使用 `os.add_dll_directory()`（底层调用 `AddDllDirectory`）+ `LoadLibraryExW` 可成功加载。

**影响范围**：仅影响 `gpt-sovits-test-pipeline.exe`（及其它链接 `tts.dll` 的可执行文件）。不链接 `tts.dll` 的可执行文件（`test_ops.exe`、`bert-alignment.exe`、`test-backend.exe` 等）正常运行。

**可能的方向**：
- 在 `ggml-base.dll` 的 `ggml_backend_load` 中使用 `LoadLibraryExW` + `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` 或 `SetDllDirectory` 改善 MODULE 库的依赖搜索
- 或者将所需的 Intel 运行时 DLL 复制到 `bin/` 目录

### 5.3 SYCL SoftMax NaN Bug

已定位修复方案（clamp -80.0f），待提 PR 到 ggml 上游。详见 `docs/ggml_patch_and_upstream_plan.md` 第 4 节。

### 5.4 CUDA/SYCL convT 的 CPU padding workaround

`ops_conv_transpose_1d.cpp` fallback 路径仍有 `padding=0 + view 裁剪` 规避 ggml CPU bug。CUDA/SYCL 路径不受影响。

---

## 6. 下一步

1. 解决 `ggml-sycl.dll` 运行时加载问题（见 5.2）
2. 权重 F16→F32 预转换（VITS 模型加载时）
3. workspace 池化
4. SYCL SoftMax PR 提交到 llama.cpp / ggml
5. 跑完整的 CPU/CUDA/SYCL 三后端测试
