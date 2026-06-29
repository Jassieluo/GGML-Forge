# CPU Conv1D 优化记录

> 最后更新: 2026-06-24

## 背景

VITS 模型包含大量 Conv1D 层（~422 层/次推理），其中绝大部分是小核卷积（kW=3/5/7，stride=1，C_in/C_out 从 16 到 512）。ggml 原生的 Conv1D 实现（im2col + mul_mat）在 CPU 上 RTF ~1.37，我们希望超越这个 baseline。

## 尝试方案

### 阶段 1：im2col + 自定义矩阵乘（已废弃，RTF 2.51）

将 ggml 的矩阵乘（`ggml_vec_dot_f32` + `mul_mat_one_chunk` + 16×16 分块 + atomic work-stealing）完整拷贝为 `ops_matmul_f32`，配合自定义 im2col 用于 Conv1D。

**文件**：`src/ops/ops-cpu/matmul_f32.cpp`、`src/ops/ops-cpu/conv_1d.cpp`

**结果：RTF 2.51**，比 ggml 原生慢约 85%。根因：

1. **im2col 内存开销**：输入数据展开 kW（3~11）倍，`col` 矩阵写+读消耗大量内存带宽
2. **分线程 fork-join 开销**：没有 ggml 的 threadpool，每层 fork OpenMP 线程
3. **无 L1 blocking**：缺少针对小卷积核的数据复用优化

### 阶段 2：讨论方案 → 暂停

分析了几条 CPU Conv 极致优化路线：

| 方案 | 预期 RTF | 代价 | 状态 |
|---|---|---|---|
| Direct Conv (NHWC + AVX2 oc 展开) | ~1.2-1.5 | 纯手动实现，无 JIT | 未实现 |
| Winograd F(2,3) for kW=3 stride=1 | ~1.0-1.3 | 仅覆盖 ~60% 层，代码复杂 | 未实现 |
| **Intel oneDNN** | ~0.5-1.0 | 运行时代码生成（JIT） | 排除（用户不接受 JIT） |
| **ggml 原生（当前回退）** | 1.37 | 无额外代码 | **已启用** |

**决定**：临时回退到 ggml 原生实现，待有明确的优化方案后再重新启用自定义路径。

## 当前状态

- `src/ops/ops-cpu/ops_cpu.cpp` 中 `GGML_OP_OPS_VIRT_CONV_1D` handler 已注释
- CPU 后端 Conv1D 走 `ggml_conv_1d()`（ggml 原生 im2col + mul_mat）
- `src/ops/ops-cpu/conv_1d.cpp` 和 `matmul_f32.cpp` 保留，供后续参考

## 未来方向（待设计）

如果后续要重新优化 CPU Conv1D，建议优先调研 **Direct Conv (HWC 布局)**：

```
核心思路：
  放弃 im2col，跳过中间内存展开，HWC 布局确保内层 ic 循环可以被 SIMD 向量化
  oc_block 分块让权重常驻 L1 cache
  单次 load src 被 oc_block 次 fmadd 复用 → 从 memory bound 转为 compute bound

关键参数：
  oc_block = 8 (AVX2) / 16 (AVX512)
  oc_block × C_in × kW × 4B < L1 cache (32KB)
```

附加优化可以考虑 Winograd F(2,3) 专门处理 kW=3 stride=1 的层（VITS 主力层型），减少 33% 乘法。
