# Double Swish 算子设计与优化方案

## 1. 数学公式与背景描述

Double Swish 激活函数在 GPT-SoVITS 的自回归（AR）文本转语音 Transformer 模型中被广泛使用。它的数学公式定义为：
$$\text{DoubleSwish}(x) = x \cdot \text{sigmoid}(x - 1.0) = \frac{x}{1.0 + e^{-(x - 1.0)}}$$

### A. 性能瓶颈与提取缘由
在标准的自回归（AR）逐词解码生成中，序列长度为 $1$。该算子如果手动使用最基础的 GGML 节点实现，会产生以下子图：
1. `ggml_new_tensor` 创建大小相同的全 `1.0` 临时常数张量。
2. `ggml_fill` 填充 `1.0`。
3. `ggml_sub(x, 1)`。
4. `ggml_sigmoid(...)`。
5. `ggml_mul(x, sig)`。

这共产生了 5 个 GGML 计算图节点。自回归阶段运行数百步，会导致频繁的分配、清空和 5 次显存读写，极大地降低了解码效率。

### B. 优化提取方案
将其作为公共算子提取为 `GGML_OP_OPS_VIRT_DOUBLE_SWISH`。在所有计算后端上实现 **单通、0 临时空间、元素级（Element-wise）** 的核函数，将 5 次显存操作完全融合成 1 次。

---

## 2. CPU 端优化 (AVX2 / AVX512 / OpenMP)
所有的 CPU 侧实现位于 `src/ops/ops-cpu/double_swish/` 文件夹中：
* **运行时 CPUID 特征分发**：在运行时通过 `cpu_has_avx2()` 或 `cpu_has_avx512()` 进行指令集挑选。
* **AVX2 / AVX512 向量化实现**：
  * 使用向量减法计算 $x - 1$。
  * 利用 `ggml_v_expf` 指令快速矢量化计算指数 $e^{-(x-1)}$（并包含 $[-20.0, 20.0]$ 范围内的 clamp 边界处理，防止 `NaN` 或零下溢）。
  * 向量除法 `_mm256_div_ps` 直接得出最终结果，避免了繁重的标量循环。
* **OpenMP 并行化**：在行不连续或超大张量模式下，外层利用多线程进行行级分块，内层继续 SIMD 连续加载。

---

## 3. CUDA 端优化 (GPU 快速超越单元)
CUDA 端的实现位于 `src/ops/ops-cuda/double_swish.cu`。
* **Warp 内存合并读写**：核函数使用一维连续网格跨步循环（Grid-stride Loop），Warp 内 32 个线程的读取和写入地址对齐，融合成极少的 128 字节显存事务。
* **快速硬件指数指令 (`expf` / SFU)**：
  利用 GPU 特殊功能单元（SFU）单周期硬件指令直接计算 $e^{-(x-1)}$，并使用 `fmaxf` 和 `fminf` 硬件指令防止指数溢出。

---

## 4. SYCL 端优化 (Intel oneAPI)
SYCL 端的实现位于 `src/ops/ops-sycl/double_swish.cpp`。
* 使用 `sycl::exp` 超越函数在编译期被 Intel oneAPI DPC++ 优化器转换为针对 Intel Xe 架构 GPU 的矢量算术指令。
* 内置连续空间与非连续 Strided 分支，对满足行连续的张量采用工作组一维展开映射，确保高吞吐率。
