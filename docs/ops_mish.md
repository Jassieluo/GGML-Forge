# Mish 算子设计与优化方案

## 1. 数学公式转换 (oneDNN 风格优化)

Mish 激活函数的标准定义为：
$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))$$

其中 $\text{softplus}(x) = \ln(1 + e^x)$。标准的实现方式需要先计算 $\ln$，再计算 $\tanh$，这在 CPU/GPU 上涉及两个昂贵的非线性超越函数调用，极难做向量化。

仿照 oneDNN 的高效数学变换，我们可以将 $\tanh(z)$ 的指数展开式 $\frac{e^{2z} - 1}{e^{2z} + 1}$ 代入 $z = \ln(1 + e^x)$：
$$e^z = 1 + e^x$$
$$e^{2z} = (1 + e^x)^2$$

因此，Mish 的计算可以直接简化为：
$$\text{Mish}(x) = x \cdot \frac{(1 + e^x)^2 - 1}{(1 + e^x)^2 + 1}$$

### 优化效果：
* **去除了 $\ln$ (对数) 和 $\tanh$ (双曲正切) 的函数调用**。
* 整个算子退化为仅需 **1 次指数计算、1 次加法、1 次平方、以及基本的乘除法**。
* 极易通过 SIMD/GPU 硬件指令进行加速。

---

## 2. CPU 端优化 (AVX2 / AVX512 / OpenMP)

为了避免全局向量加速代码变得庞大混乱，Mish 算子的所有 CPU 侧实现被收拢到 `src/ops/ops-cpu/mish/` 文件夹中：
* **系统特征动态分发**：在运行时通过 `ggml_cpu_has_avx512()` 或 `ggml_cpu_has_avx2()` 动态挑选最匹配的指令集。
* **行连续快速路径 (Row Contiguous)**：将 OpenMP 并行提至外侧的 `collapse(3)`，让每个 CPU 线程分得完整的行；在线程内部，利用 AVX2（一次处理 8 个 float）或 AVX512（一次处理 16 个 float）进行 SIMD 连续加载与计算。
* **零值/越界保护**：在计算前对 $x$ 进行 $[-20.0, 20.0]$ 范围内的 clamp 剪裁，有效防止指数计算中发生溢出（`NaN`）和下溢（变为 `0`）。

---

## 3. CUDA 端优化 (GPU 并行与硬件加速)

CUDA 端的实现位于 `src/ops/ops-cuda/mish.cu`。

### A. 硬件级快速超越函数 (`__expf`)
在 GPU 上，标准的 `expf` 相对较慢。我们使用了 CUDA 硬件专用的特殊函数单元（SFU）指令 **`__expf`**：
* `__expf(x)` 映射为底层的硬件单周期单精度指数指令，比标准的 `expf` 快数倍。
* 配套使用 GPU 级的边界剪裁（使用 `fminf` 和 `fmaxf` 编译为硬件原生指令），防止输入值过大导致浮点数溢出。

### B. 网格跨距循环 (Grid-Stride Loops)
使用网格跨距循环设计 CUDA Kernel，能够保证：
1. **任意输入规模兼容**：无论元素总数多大，线程块与网格大小都不受硬件资源硬性限制。
2. **合并内存访问 (Coalesced Memory Access)**：各线程读取 `x` 和写入 `y` 时，物理地址处于同一个对齐的 Warp（32线程）内，能够合并成极少的内存交易（Memory Transactions），达到最大理论带宽。

---

## 4. SYCL 端优化 (Intel GPU 与 oneAPI)

SYCL 端的实现位于 `src/ops/ops-sycl/mish.cpp`。

### A. oneAPI 编译器快速数学库
在 Intel GPU 架构下：
* 显式通过编译命令 `-fsycl` 和启用 fast-math 编译选项。
* 内部使用 `sycl::ext::intel::math::exp` 或编译器高度优化的 `sycl::exp` 进行指数操作，映射到 Intel X^e 架构 GPU 内部的向量计算引擎（Vector Engine），提供亚纳秒级计算。

### B. 工作组并行与向量读取
* 每个工作项（Work-item）计算单个元素，自动映射到硬件硬件流水线中。
* 保证对全局显存的一维连续读写，适配 Intel 显存控制器的合并访问机制，最大程度发挥显卡高带宽的优势。
