# SYCL Backend 状态记录

> 最后更新: 2026-06-24

## 现状

SYCL 后端（Intel Iris Xe Graphics）**T2S 推理阶段 GPU hang**，无法生成音频。

## 表现

在任何 SYCL 配置下，T2S Step 0 的图执行都在以下位置卡死：

```
[T2S Debug] Step 0: computing graph...
[GGML Hook] Intercepting op 2003 on backend SYCL0        ← Softmax handler 正常
[GGML Hook] Executed handler for op 2003 with result success
[GGML Hook] Intercepting op 2006 on backend SYCL0        ← DoubleSwish handler 被调用
[GGML Hook] Executing handler for op 2006 on backend SYCL0...
                                                          ← 卡死在此，handler 不返回
```

**关键发现**：即使在 DoubleSwish handler 中：
- 去掉 `q->wait()` → 仍然卡在 `q->submit()`
- 用 `q->memcpy()` 代替 kernel submit → 仍然卡
- 创建**全新的独立 queue** 提交 → 仍然卡

这表明**Intel Iris Xe GPU 本身已 hang**（不只是某个 queue 的问题），根因在 ggml 原生 SYCL op 的执行中。

## 尝试过的修复

### 已确认与问题无关的变更

| 尝试 | 结果 |
|---|---|
| 去掉所有 handler 的 `q->wait()` | 不影响，问题不在 wait |
| 恢复 DoubleSwish 原始代码 | 依然卡，非 handler 引入的 bug |
| `GGML_SYCL_DISABLE_DNN=1` 关闭 oneDNN | 依然卡 |
| `GGML_SYCL_FORCE_MMQ=1` 强制量化矩阵乘路径 | 依然卡 |
| 独立 SYCL queue 提交 | 依然卡 |

### 已确认正确的修复（保留）

**ConvTranspose1D 权重布局修复**（`src/ops/ops-sycl/conv_transpose_1d.cpp`）：

ggml 的 ConvT 权重是 `[kW][C_out][C_in]` row-major 布局，但 `column_major::gemm` 按 `[C_out*kW, C_in]` 列主序读取，两者元素位置对不上。新增 `rearrange_weight_conv_t` 核函数在 GEMM 前重排权重。

```cpp
// ggml 实际位置: w_rearranged[k*C_out*C_in + oc*C_in + ic]
// GEMM 读取方式: w_rearranged[oc + k*C_out + ic*C_out*kW] = w[k][oc][ic]
```

1×1 和通用路径均已修复。

## 工程临时变更（需要回退）

以下 CMakeLists.txt 修改是为绕过 cmake reconfigure 失败而做的，如果后续需要正常 cmake 构建需要还原：

| 文件 | 修改 | 原因 |
|---|---|---|
| `CMakeLists.txt` | `project(gpt_sovits C CXX)` 去掉 ASM | 缺少 ASM compiler |
| `CMakeLists.txt` | 注释掉 `add_subdirectory(tests/...)` | IntelSYCL cmake feature test 崩溃 |
| `src/CMakeLists.txt` | 跳过 `find_package(IntelSYCL)` | 同上 |

## 后续排查方向

1. **检查 ggml-sycl 的 MUL_MAT 实现**：T2S 模型中 softmax 和 double_swish 之间的原生 ggml SYCL op（主要是 MUL_MAT）可能是 GPU hang 的触发点
2. **用 CUDA 后端对比**：同一模型在 CUDA 后端（RTX 4060）上是否正常
3. **尝试不同的 oneAPI 版本**：当前 2025.3，降级或升级可能修复驱动兼容性问题
4. **测试 Iris Xe 的 GPU 驱动**：`Ctrl+Win+Shift+B` 重启驱动后重试
