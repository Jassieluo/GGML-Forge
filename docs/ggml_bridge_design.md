# ggml Bridge 桥接层设计

## 1. 目标

在不修改 ggml 已有源码的前提下，通过**新增桥接文件 + 最少量行级注入**，为 GPT-SoVITS.cpp 的自定义算子提供：

- 钩子注册机制（替代当前对 `ggml-backend.cpp` 的 patch）
- 原生后端资源访问（cudaStream / cublasHandle / sycl::queue）
- 计算图调度（supports_op 放行 + graph_compute 劫持）

所有桥接文件源码归属本项目（`scripts/ggml-bridge/`），ggml 目录保持 git-clean。每次 `git pull` ggml 后执行 `apply_ggml_patches.py` 一键完成拷贝 + 注入。

## 2. 文件布局

```
GPT-SoVITS.cpp/
├── scripts/
│   ├── apply_ggml_patches.py          ← 重写：拷贝 + 注入 + 验证 + --revert
│   └── ggml-bridge/                   ← 桥接文件源码（跟本项目 git）
│       ├── ggml-ops-ext-bridge.h      ← 公开头文件
│       ├── ggml-ops-ext-bridge.cpp    ← 钩子全局变量 + setter（无后端依赖）
│       ├── ggml-ops-ext-bridge-cuda.cu← CUDA 资源 getter（编译进 ggml-cuda）
│       └── ggml-ops-ext-bridge-sycl.cpp← SYCL 资源 getter（编译进 ggml-sycl）

ggml/                                  ← 上游 git submodule，保持纯净
├── src/
│   ├── ggml-ops-ext-bridge.h          ← [COPY]  从 scripts/ggml-bridge/ 拷贝
│   ├── ggml-ops-ext-bridge.cpp        ← [COPY]  编译进 ggml target
│   ├── CMakeLists.txt                 ← [INJECT] 1 行：在 ggml-backend-reg.cpp 后追加源文件
│   │
│   ├── ggml-cpu/
│   │   ├── ggml-cpu.cpp               ← [INJECT] 3 行
│   │   └── (CPU 无需独立 bridge，getter 逻辑简单)
│   │
│   ├── ggml-cuda/
│   │   ├── ggml-ops-ext-bridge-cuda.cu← [COPY] → GLOB "*.cu" 自动编译
│   │   └── ggml-cuda.cu               ← [INJECT] 3 行
│   │
│   └── ggml-sycl/
│       ├── ggml-ops-ext-bridge-sycl.cpp←[COPY] → GLOB "*.cpp" 自动编译
│       └── ggml-sycl.cpp              ← [INJECT] 3 行
```

## 3. 注入点明细

### 3.1 总览

| # | 目标文件 | 操作 | 锚点 | 内容 |
|---|---|---|---|---|
| 1 | `ggml/src/CMakeLists.txt` | INJECT | `ggml-backend-reg.cpp` | 追加 bridge 源文件到 ggml target |
| 2 | `ggml-cpu.cpp` | INJECT | `#include "ggml-impl.h"` | 追加 `#include` |
| 3 | `ggml-cpu.cpp` | INJECT | `device_supports_op` 函数体 `{` | `if (bridge_supports_op(...)) return true;` |
| 4 | `ggml-cpu.cpp` | INJECT | `cgraph->n_nodes` 循环体 `{` | `if (bridge_try_dispatch(...)) continue;` |
| 5 | `ggml-cuda.cu` | INJECT | `#include "ggml-cuda/common.cuh"` | 追加 `#include` |
| 6 | `ggml-cuda.cu` | INJECT | `device_supports_op` 函数体 `{` | 同 #3 |
| 7 | `ggml-cuda.cu` | INJECT | `cgraph->n_nodes` 循环体 `{` | 同 #4 |
| 8 | `ggml-sycl.cpp` | INJECT | `#include "common.hpp"` | 追加 `#include` |
| 9 | `ggml-sycl.cpp` | INJECT | `device_supports_op` 函数体 `{` | 同 #3 |
| 10 | `ggml-sycl.cpp` | INJECT | `cgraph->n_nodes` 循环体 `{` | 同 #4 |
| — | `ggml/src/` | COPY | — | `ggml-ops-ext-bridge.h` + `.cpp` |
| — | `ggml/src/ggml-cuda/` | COPY | — | `ggml-ops-ext-bridge-cuda.cu` |
| — | `ggml/src/ggml-sycl/` | COPY | — | `ggml-ops-ext-bridge-sycl.cpp` |

### 3.2 注入细节

#### ggml-cpu.cpp

```cpp
// 注入 1: include（锚点 #include "ggml-impl.h"）
#include "ggml-impl.h"
#include "ggml-ops-ext-bridge.h"          // ← 新增

// 注入 2: supports_op（锚点 ggml_backend_cpu_device_supports_op 函数体开头）
static bool ggml_backend_cpu_device_supports_op(...) {
    if (ggml_ops_ext_bridge_supports_op(op->op)) return true;  // ← 新增
    // ... 原有逻辑 ...
}

// 注入 3: graph_compute（锚点 cgraph->n_nodes 循环体开头）
// 在 for (int i = 0; i < cgraph->n_nodes; i++) { 之后
    if (ggml_ops_ext_bridge_try_dispatch(backend, node)) {    // ← 新增
        node->op = GGML_OP_NONE;                              // ← 新增
        continue;                                             // ← 新增
    }                                                         // ← 新增
```

#### ggml-cuda.cu / ggml-sycl.cpp 同理

#### ggml/src/CMakeLists.txt

```cmake
# 锚点: ggml-backend-reg.cpp
add_library(ggml
            ggml-backend-dl.cpp
            ggml-backend-reg.cpp
            ggml-ops-ext-bridge.cpp)      # ← 新增
```

### 3.3 为什么 GLOB 让 CUDA/SYCL bridge 零 CMake 注入

```cmake
# ggml/src/ggml-cuda/CMakeLists.txt（现有代码，不修改）
file(GLOB GGML_SOURCES_CUDA "*.cu")
# → ggml-ops-ext-bridge-cuda.cu 拷贝进去后自动被 GLOB 拾取

# ggml/src/ggml-sycl/CMakeLists.txt（现有代码，不修改）
file(GLOB GGML_SOURCES_SYCL "*.cpp")
# → ggml-ops-ext-bridge-sycl.cpp 拷贝进去后自动被 GLOB 拾取
```

## 4. Bridge API 设计

### 4.1 公开头文件 (`ggml-ops-ext-bridge.h`)

```c
#ifndef GGML_OPS_EXT_BRIDGE_H
#define GGML_OPS_EXT_BRIDGE_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// 自定义算子起始编号
#define GGML_OP_EXT_BASE  2000

// ---- 钩子机制 ----

typedef bool (*ggml_ops_ext_handler_t)(ggml_backend_t backend, struct ggml_tensor * node);

// 设置全局钩子（由 src/ops/ops.cpp 在初始化时调用）
GGML_API void ggml_ops_ext_bridge_set_hook(ggml_ops_ext_handler_t hook);

// ---- 计算图调度（由 ggml 后端 graph_compute 调用）----

// supplies_op 放行
GGML_API bool ggml_ops_ext_bridge_supports_op(int op);

// 尝试分发自定义算子，返回 true 表示已处理
GGML_API bool ggml_ops_ext_bridge_try_dispatch(
    ggml_backend_t backend,
    struct ggml_tensor * node
);

// ---- 原生资源访问 getter ----

// CUDA（仅 CUDA 后端可用）
GGML_API int ggml_ops_ext_bridge_cuda_get_device(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_cuda_get_stream(ggml_backend_t backend);   // returns cudaStream_t
GGML_API void* ggml_ops_ext_bridge_cuda_get_cublas(ggml_backend_t backend);   // returns cublasHandle_t

// SYCL（仅 SYCL 后端可用）
GGML_API void* ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend);    // returns sycl::queue*

#ifdef __cplusplus
}
#endif

#endif // GGML_OPS_EXT_BRIDGE_H
```

### 4.2 各 Bridge 实现文件职责

| 文件 | 编译进 | 实现内容 |
|---|---|---|
| `ggml-ops-ext-bridge.cpp` | `ggml` (base target) | 钩子全局变量 + `set_hook` · `supports_op` · `try_dispatch`（调度逻辑，后端无关） |
| `ggml-ops-ext-bridge-cuda.cu` | `ggml-cuda` (GLOB) | `cuda_get_device` / `cuda_get_stream` / `cuda_get_cublas`（include `common.cuh`，合法访问内部 struct） |
| `ggml-ops-ext-bridge-sycl.cpp` | `ggml-sycl` (GLOB) | `sycl_get_queue`（include `common.hpp`，合法访问内部 struct） |

### 4.3 Ops 层调用方式（src/ops/ 侧）

原来的写法：
```cpp
// 旧：include ggml 私有头 + 裸强转
#include "ggml-cuda/common.cuh"
ggml_backend_cuda_context* ctx = (ggml_backend_cuda_context*)backend->context;
int device = ctx->device;
cudaStream_t stream = ctx->streams[device][ctx->curr_stream_no];
```

新的写法：
```cpp
// 新：只依赖公开 bridge API
#include "ggml-ops-ext-bridge.h"
int device = ggml_ops_ext_bridge_cuda_get_device(backend);
cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);
cublasHandle_t cublas = (cublasHandle_t)ggml_ops_ext_bridge_cuda_get_cublas(backend);
```

## 5. apply_ggml_patches.py 工作流

```
python scripts/apply_ggml_patches.py [--revert]
```

| Phase | 操作 |
|---|---|
| **CHECK** | 验证所有 COPY 源文件存在（`scripts/ggml-bridge/`） |
| **COPY** | 拷贝 bridge 文件到 `ggml/src/`、`ggml/src/ggml-cuda/`、`ggml/src/ggml-sycl/` |
| **INJECT** | 对 4 个目标文件执行 10 行注入（全部基于单行锚点精确匹配） |
| **VERIFY** | 每个锚点行必须存在，否则立即报错退出，列出失败文件:行号 |

`--revert` 模式：
- 从 ggml 目录删除拷贝的 bridge 文件
- 从注入文件中移除注入行
- 还原 ggml 到 git-clean 状态

## 6. --revert 实现方式

不采用「反向正则替换」，而是使用**标记行**：

```cpp
// 注入行统一格式：
// @GGML_BRIDGE_INJECT: <injection_id>
#include "ggml-ops-ext-bridge.h"
```

`--revert` 时删除所有带有 `@GGML_BRIDGE_INJECT:` 标记的行。

## 7. 与当前方案的对比

| | 当前 patch 方案 | 桥接方案 |
|---|---|---|
| 修改 ggml 文件数 | 5 个 | 4 个 |
| 注入行数 | ~100 行正则 patch | 10 行精确锚点注入 |
| 锚点类型 | 多行正则匹配函数体内部 | 单行（include 指令 / 函数签名 / `cgraph->n_nodes`） |
| 失败模式 | 正则静默不匹配 | 锚点行找不到 → 立即报错 |
| ops 层对 ggml 依赖 | include `common.cuh` + 裸强转（7 个 kernel 文件） | 只 include `ggml-ops-ext-bridge.h` |
| upstream sync 容错 | 函数体改动 → 静默失败 | 函数签名改动才失败（极少） |
