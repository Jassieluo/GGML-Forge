# Gated Tanh Sigmoid 算子设计与优化方案

## 1. 数学公式与通道切分

Gated Tanh Sigmoid 激活函数常用于音频/语音生成网络（如 WaveNet、VITS）的门控机制。其数学公式定义为：
$$y = \tanh(x_a) \cdot \text{sigmoid}(x_b)$$

其中：
* 输入张量 $x$ 的通道数维度大小为 $2C$。
* $x_a$ 为输入通道的前半部分（前 $C$ 个通道），$x_b$ 为输入通道的后半部分（后 $C$ 个通道）。
* 输出张量 $y$ 的通道数大小为 $C$。

---

## 2. CPU 端优化 (AVX2 / AVX512 / OpenMP)

所有的 CPU 侧实现位于 `src/ops/ops-cpu/gated_tanh_sigmoid/` 文件夹中：
* **数学向量化化简**：将 $\tanh(x_a)$ 和 $\text{sigmoid}(x_b)$ 展开为指数计算形式，直接调用编译期高度优化的 `ggml_v_expf` 指令计算 $e^{2x_a}$ 与 $e^{-x_b}$。
* **物理内存连续寻址**：由于后半段通道相对于前半段通道恰好有固定的 $C$ 字节偏移，且每一行的 $x_a$ 与 $x_b$ 各自物理连续，因此可以完美通过 SIMD 寄存器从 `px_l` 和 `px_r = px_l + C` 处连续抓取。
* **行级 OpenMP 并行**：在最外层 Batch 和时间步维度上实施 `omp parallel for collapse(3)`，使多线程处理无写冲突，内层彻底向量化。

---

## 3. CUDA 端优化 (GPU 快速超越函数与合并访问)

CUDA 端的实现位于 `src/ops/ops-cuda/gated_tanh_sigmoid.cu`。

### A. 硬件级双重超越函数加速 (`__tanhf` 与 `__expf`)
CUDA 设备上内置了极速的硬件单精度双曲正切函数 **`__tanhf`**：
* 门控公式在 GPU 核心中表示为：
  $$y[i] = \text{\_\_tanhf}(x_a[i]) \cdot \frac{1.0\text{f}}{1.0\text{f} + \text{\_\_expf}(-x_b[i])}$$
* `__tanhf` 与 `__expf` 会被翻译为 GPU 核心中的特殊功能单元（SFU）指令，完全不占用普通的浮点数 ALU，效率极高。

### B. Warp 内存合并读取与写入
* 当线程块中的 warp 访问 `x_a[i]` 和 `x_b[i]` 时，因为 $x_a$ 的起始地址与 $x_b$ 的起始地址均为通道对齐的（通常 $C$ 为 32 或 64 的倍数），GPU 会将每个 warp（32 个线程）的读操作合并为两个极速的 128 字节显存加载指令，从而实现近乎 100% 的显存总线带宽利用率。

---

## 4. SYCL 端优化 (Intel 架构优化)

SYCL 端的实现位于 `src/ops/ops-sycl/gated_tanh_sigmoid.cpp`。

### A. DPC++ 硬件优化
* 采用 `sycl::tanh` 和 `sycl::exp`。在 Intel 的 OneAPI DPC++ 编译器中，这两个指令会在后端被优化为针对 Intel GPU 特殊硬件计算管线（Xe Vector Engines, XVE）优化的矢量指令，保证高并发下的卓越吞吐率。

### B. 显存合并读取机制
* 同样严格遵守一维连续读取，将时间步和 Batch 进行维度平铺，确保 Intel GPU 的工作项（Work-item）在对输入缓冲区 $x$ 的 $2C$ 通道读取时能合并成超大显存块读写操作，大幅降低了显存读写延迟。
