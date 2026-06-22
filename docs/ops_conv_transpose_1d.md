# 1D Transposed Convolution (反卷积) 算子设计与优化方案

## 1. 传统的 im2col 展开 vs 直接流式累加

在一维转置卷积中，输出像素与其对应的输入像素满足以下对应关系：
$$ow = iw \cdot \text{stride} - \text{padding} + kw \cdot \text{dilation}$$

### A. 传统的 im2col/col2im (GGML 默认实现)
* **做法**：将输入或输出扩展成一个巨大的中间矩阵（col_buf），然后通过矩阵乘法（GEMM）计算，最后再通过 `col2im` 累加回原内存。
* **痛点**：对于一维转置卷积，中间膨胀的 `col_buf` 会占用大量的内存/显存（尤其是 Batch 较大或序列较长时）。同时，巨大的中间矩阵的写入与读取会造成极大的显存带宽浪费。

### B. 直接流式累加 (Stream-Accumulation) 与零显存 (0-Workspace) 优化
* **核心思想**：我们彻底移除了任何中间矩阵展开，通过重组循环结构，以无冲突（CPU）或无冲突线程映射（GPU）的架构，直接向输出缓冲区中计算或累加计算结果。
* **物理公式变换**：
  对于每个输出位置 $ow$，其累加来源为：
  $$y[b, c_{out}, ow] = \sum_{c_{in}} \sum_{kw} w[c_{in}, c_{out}, kw] \cdot x[b, c_{in}, iw]$$

---

## 2. CPU 端优化 (行累加与 L1 Cache 常驻)

为了让 CPU 的推理速度达到极致，我们设计了以下极其高效的行级映射：
* **外层多线程并行**：将 OpenMP 并行应用在 `c_out`（输出通道）和 Batch 维度上。由于每个 CPU 核心独立计算一个输出通道的整行数据 `dst_row`，线程之间在内存上完全隔离，**没有任何写-写冲突**，无需使用 Atomic 锁。
* **L1 缓存常驻**：一维输出序列行长度 $L_{out}$ 通常在几百到两千之间，大小仅为 400B 到 8KB。整个 `dst_row` 在累加计算期间会**完全常驻在当前核心的 L1 缓存**中，避免了频繁往返读写主内存。
* **零值剪枝短路**：在内部像素循环中，增加 `if (val_x == 0.0f) continue;`。若输入像素为零则直接跳过内层 `kw` 的计算，提供了极佳的无损推理加速。

---

## 3. CUDA 端优化 (0-Workspace 线程映射与 cuBLAS 捷径)

CUDA 端的实现位于 `src/ops/ops-cuda/conv_transpose_1d.cu`。

### A. 无冲突线程格映射 (Zero-Conflict Thread-Grid Mapping)
* **线程职责**：每一个 CUDA 线程（Thread）独立负责计算输出张量的**单个元素** `dst[b, c_out, ow]`。
* **零 Workspace 与零 Lock 占用**：
  * 因为输出张量的每个位置都有唯一的一个线程专属写入，**完全消除了多个线程写入同一地址的锁冲突**。无需使用极其缓慢的 `atomicAdd`。
  * 彻底去除了 `im2col`/`col2im` 阶段，显存 Workspace 开销降为常数级 **$O(1)$**。
* **Warp 合并显存写入**：由于同一个 Warp（32个线程）的 `threadIdx.x` 映射到输出张量最内层连续的 `ow` 坐标，GPU 将这 32 个线程的写回指令完美合并为一个单指令显存写交易，写入性能极佳。

### B. 1x1 卷积直接 cuBLAS 捷径 (Shortcut)
* **短路判定**：如果卷积核满足 `kW == 1 && stride == 1 && padding == 0 && dilation == 1`：
  * 算子会直接将该反卷积短路为**标准 GEMM 矩阵乘法**。
  * 内部直接调用 NVIDIA 官方的 **`cublasSgemm`**，将输入 $x$ 与权重 $w$ 直接在 GPU Tensor Core 上高速相乘，结果一步到位写回 `dst`。
  * 此路径没有经过任何反卷积 kernel，实现最高硬件理论浮点计算效率。

---

## 4. SYCL 端优化 (oneMKL 与 Intel GPU)

SYCL 端的实现位于 `src/ops/ops-sycl/conv_transpose_1d.cpp`。

### A. 线程格映射与合并读写
* 与 CUDA 方案对齐，采用 Work-item 1-to-1 映射到 `dst[b, c_out, ow]` 的策略。每个工作项在计算时，利用逆向公式反推对应的输入 `iw` 并进行内积运算。
* 完美适配 Intel Arc 及 Data Center GPU 的显存控制器，确保线程写入地址是一维连续的，实现最大吞吐写入。

### B. 1x1 卷积 Intel MKL / oneDNN 捷径
* 针对 1x1 反卷积短路路径，调用 Intel 的 **oneMKL** 的 `gemm` 接口或 **oneDNN** 直连，充分压榨 Intel GPU 核心内部矩阵引擎（Xe Matrix Extensions, XMX）的计算潜力，大幅缩短语音合成中上采样反卷积的推理时间。
