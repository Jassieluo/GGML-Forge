# GPT-SoVITS.cpp

GPT-SoVITS.cpp 是一个专为 **GPT-SoVITS** 语音合成系统设计的高性能、轻量化、**纯 C++** 推理引擎。

基于优秀的张量计算库 `ggml`，本项目彻底摆脱了 Python 运行时及 PyTorch 等庞大依赖，旨在为各种硬件平台（包括嵌入式设备、桌面应用及高并发服务器）提供低延迟、高吞吐量的本地化语音合成能力。

---

## 🌟 项目亮点

*   **⚡ 纯 C++ 全管线支持**：无需 Python 依赖。集成了从文本前端分词、音素转换、G2P 拼音字典，到 Hubert、BERT、GPT (T2S) 和 VITS 的完整推理流程。
*   **🚀 多后端硬件加速**：
    *   **CPU**：使用 OpenMP 框架与定制的 AVX2 SIMD 指令集加速。
    *   **NVIDIA CUDA**：利用英伟达显卡实现极速语音合成（支持 cuDNN 卷积加速）。
    *   **Intel SYCL**：集成英特尔 oneAPI（DPC++）支持，在 Intel 核心显卡及 Arc 独立显卡上实现高效推理。
*   **💎 Q4_0 混合精度量化**：
    *   提供了一键式量化脚本，支持将 GPT 和 BERT 权重压缩至 4-bit `Q4_0` 格式。
    *   **关键层保护机制**：强制将声学概率预测层等精度敏感层保护为 FP16，使 GPT 量化后模型体积缩减 **70.3%**（自 148.4 MB 降至 44.1 MB）的同时，在采样对齐、停止符预测和音频质量上与原生 FP16 保持完全一致。
*   **🔥 极致的 CPU SIMD 算子加速**：
    *   **SIMD-8 向量化优化**：重构了 CPU 的 1D 卷积算子，将计算步长优化为 8 字节对齐，全面激活了 VITS 最后一级上采样（16-channel，1.8M 长度）在 AVX2 下的并行计算，解决了原版硬编码导致的部分层退化为标量慢速计算的问题。
    *   **Cache-locality 优化**：调整了输入转置的循环嵌套顺序，实现对内存的完全连续读取，充分激发 CPU 硬件数据预取器性能。
    *   经过优化，CPU 推理耗时大幅缩减，实时率（RTF）突破至 **0.83**。
*   **🛠️ 深度硬件 Bug 修复**：
    *   **转置卷积内核**：修复了 GGML 原生 `GGML_OP_CONV_TRANSPOSE_1D` 在 GPU 上的步长错位与通道循环 Bug。通过自定义 GPU Kernel（CUDA/SYCL）实现了精确的高质量音频重建。
    *   **防下溢 SoftMax**：修复了 Intel 显卡上 `sycl::native::exp()` 处理大负数注意力遮罩（`-10000.0f`）时的 NaN 下溢崩溃，保证了声学对齐的鲁棒性。
*   **🎶 健壮的特征对齐**：
    *   在参考音频输入管线中实现了 **零均值单位方差归一化（Zero-Mean Unit-Variance Normalization）**，解决了解析在线提取参考音频特征时因能量未对齐导致的波形异常与时间拉长问题。

---

## 🏗️ 系统架构

项目的推理管线设计如下：

```
[ 输入文本 ] ---> [ 文本前端 G2P / 拼音字典 ] ---> [ BERT 语义提取 ] 
                                                           |
                                                           v
[ 参考音频 ] ---> [ ZMU 归一化 ] ---> [ Hubert 特征提取 ] ---> [ GPT (T2S) 预测器 ] ---> [ VITS 声音合成 ] ---> [ WAV 音频输出 ]
```

---

## 📦 模型准备与量化

在运行项目前，您需要准备对应的 GGUF 模型文件。

### 1. 默认模型路径
请确保将模型文件放置在以下结构中：
```
models/gpt_sovits/
├── dict/                 # 拼音和字典文件
├── reference_audios/     # 参考音频及预提取 of 特征缓存
└── weights/
    ├── bert/
    │   └── bert_q4_0.gguf
    ├── cnhubert/
    │   └── cnhubert_fp16.gguf
    ├── t2s/
    │   └── t2s_q4_0.gguf
    └── vits/
        └── vits_fp16.gguf
```

### 2. 模型量化步骤
项目在 `scripts/gpt-sovits/` 目录下提供了模型量化脚本 [quantize_gpt_sovits.py](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/scripts/gpt-sovits/quantize_gpt_sovits.py)。

您可以使用该脚本对您的 FP16 原生模型进行 Q4_0 量化：
```bash
# 激活 Python 环境并安装依赖 (gguf 库)
pip install gguf

# 执行 T2S 模型量化 (强制保护 ar_predict_layer 输出层)
python scripts/gpt-sovits/quantize_gpt_sovits.py \
  --src models/gpt_sovits/weights/t2s/t2s_fp16.gguf \
  --dst models/gpt_sovits/weights/t2s/t2s_q4_0.gguf \
  --qtype q4_0
```

---

## 🛠️ 编译说明 (Windows x64)

### 前提条件
1.  **Visual Studio 2022 / 2026**（安装 C++ 桌面开发工作负载）。
2.  **Intel oneAPI Base Toolkit**（若使用 SYCL 后端支持，需安装并在 PATH 中）。
3.  **CUDA Toolkit 12.x**（若使用 CUDA 后端支持）。

### 编译步骤
请打开 **Visual Studio 开发者命令提示符**（VS Developer Command Prompt）运行以下命令：

```bash
# 1. 激活编译环境 (根据您的 oneAPI 安装路径调整)
call "%ONEAPI_ROOT%\setvars.bat"

# 2. 配置 CMake
# 可选项:
#  -DGGML_CUDA=ON      启用 NVIDIA CUDA 加速
#  -DGGML_SYCL=ON      启用 Intel SYCL 加速
#  -DGGML_BLAS=ON      启用 BLAS 矩阵乘法加速 (推荐 MKL)
cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_SYCL=ON -DGGML_BLAS=ON

# 3. 执行编译
cmake --build build-release --config Release --parallel
```
编译成功后，可执行程序和依赖 DLL 将生成在 `build-release/bin/` 文件夹下。

---

## 🚀 快速开始

您可以使用生成的 `gpt-sovits-test-pipeline.exe` 来测试合成效果。

### 命令行常用参数说明
*   `--text "<string>"`：待合成的文本目标。
*   `--lang "<string>"`：待合成文本的语言（例如 `zh`, `en`, `ja`）。
*   `--ref-audio "<path>"`：参考音频 WAV 路径（将进行在线 Hubert 特征提取）。
*   `--cpu`：强制使用 CPU 推理模式。
*   `--device <name>`：指定特定 GPU 后端设备（例如 `CUDA0`, `SYCL0`）。
*   `--t2s <path>`：指向 GPT (T2S) 模型文件。
*   `--vits <path>`：指向 VITS 模型文件。

### 运行示例

*   **使用 CPU 后端运行 (使用 Q4_0 量化 T2S 模型)：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe \
      --cpu \
      --t2s models/gpt_sovits/weights/t2s/t2s_q4_0.gguf \
      --text "你好，欢迎使用纯C加加推理的语音合成系统。" \
      --lang "zh" \
      --out scratch/output_cpu.wav
    ```

*   **使用 NVIDIA CUDA 加速运行：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe \
      --device CUDA0 \
      --text "项目整体架构设计干净优雅！" \
      --lang "zh" \
      --out scratch/output_cuda.wav
    ```

*   **使用 Intel GPU (SYCL) 加速运行：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe \
      --device SYCL0 \
      --text "项目整体架构设计干净优雅！" \
      --lang "zh" \
      --out scratch/output_sycl.wav
    ```

---

## 📂 代码库结构

*   `src/ops/`: 包含了外置的自定义算子拦截与分发逻辑。
    *   `ops-cpu/`: CPU 后端算子，集成定制化的 AVX2/OMP 并行加速实现。
    *   `ops-cuda/`: NVIDIA GPU 端的自定义 CUDA 核函数。
    *   `ops-sycl/`: Intel GPU 端的自定义 DPC++ 算子实现。
*   `src/pipelines/`: GPT-SoVITS 语音合成核心管线与特征归一化逻辑。
*   `examples/`: 包含了 CLI 测试程序及使用示例。
*   `scripts/`: 包含模型转换与 Q4 混合量化相关 Python 脚本。
*   `docs/`: 开发过程记录、算子详细设计文档与 Bug 修复记录。

---

## 📊 性能测试报告
具体的硬件测试数据与各后端（CPU / CUDA / SYCL）的性能实时率 RTF 指标对比，请参阅：
👉 **[性能基准测试报告 (Benchmark)](scratch/benchmark_report.md)**

---

## 🤝 参与贡献
欢迎提交 Issue 和 Pull Request 来共同优化算子计算效率、支持更多硬件平台或扩展前端多语言支持！
