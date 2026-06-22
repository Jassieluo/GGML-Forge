# GPT-SoVITS.cpp

GPT-SoVITS.cpp 是一个使用 **纯 C++** 实现的 GPT-SoVITS 语音合成系统（TTS）的高性能推理引擎。基于优秀的张量计算库 `ggml`，本项目摆脱了庞大的 Python 依赖，旨在为各种硬件平台提供轻量、低延迟、且高吞吐量的语音合成能力。

## 🌟 项目亮点

*   **纯 C++ 实现**：无需 Python 运行时，极易整合部署于嵌入式设备、桌面应用和高并发服务器中。
*   **多后端硬件加速**：
    *   **CPU**：基于 `ggml` 提供向量化加速，适合无独显或边缘端场景。
    *   **NVIDIA CUDA**：利用英伟达显卡实现极速语音合成。
    *   **Intel SYCL**：集成英特尔 oneAPI（DPC++）支持，可在 Intel Iris Xe 核显及 Arc 独显上实现高效能推理。
*   **硬件 Bug 深度修复**：
    *   **自定义转置卷积内核**：修复了 GGML 原生 `GGML_OP_CONV_TRANSPOSE_1D` 在 GPU 上的步长错位与通道循环 Bug，通过自定义 GPU Kernel（CUDA/SYCL）实现了精确的高质量音频重建。
    *   **防下溢 SoftMax**：修复了 Intel 显卡上 `sycl::native::exp()` 处理大负数注意力遮罩（`-10000.0f`）时的 NaN 下溢崩溃，保证了声学对齐的鲁棒性。
*   **完整的流式合成管线**：涵盖文本前端分词音素转换、Hubert（自监督特征提取）、BERT（语义理解）、GPT（文本到声学特征预测）以及 VITS（声码器生成）完整流程。

---

## 🏗️ 架构设计

推理管线如下图所示：

```
[ 输入文本 ] ---> [ 文本前端/拼音字典 ] ---> [ BERT 语义提取 ] 
                                                   |
                                                   v
[ 参考音频 ] ---> [ Hubert 音频特征 ] ---> [ GPT (T2S) 预测器 ] ---> [ VITS 声音合成 ] ---> [ WAV 音频输出 ]
```

---

## 🛠️ 编译说明 (Windows x64)

### 前提条件
1.  **Visual Studio 2022 / 2026**（安装 C++ 桌面开发工作负载）。
2.  **Intel oneAPI Base Toolkit**（若使用 SYCL 后端，需确保已安装并在 PATH 中）。
3.  **CUDA Toolkit 12.x**（若使用 CUDA 后端）。

### 构建步骤
在项目根目录下，使用 Visual Studio 开发者命令提示符（VS Developer Command Prompt）运行以下命令：

```bash
# 创建构建文件夹并生成 CMake 配置 (可选择启用 -DGGML_CUDNN=ON 来加速 1D 卷积和反卷积)
cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_SYCL=ON -DGGML_CUDNN=ON

# 编译项目
cmake --build build-release --config Release
```
编译产物将位于 `build-release/bin/` 文件夹下。

---

## 🚀 快速开始

在运行测试前，请确保已经将 GGUF 模型文件放置在 `models/gpt_sovits/` 目录下。

### 1. 运行环境初始化
SYCL 运行依赖于 Intel 运行时变量。运行程序前，建议在命令行中先初始化环境：
```bash
# 激活 Visual Studio 编译环境
call "D:\Visual Studio\Microsoft Visual Studio\VS2026\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
# 激活 oneAPI 环境变量
call "D:\Tools\Intel\oneAPI\setvars.bat"
```

### 2. 执行语音合成测试
可以使用编译生成的 `gpt-sovits-test-pipeline.exe` 来合成指定的文本。

*   **使用 CPU 后端：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe --text "你好，欢迎使用纯C++推理的语音合成系统。" --lang "zh" --cpu --out scratch/output_cpu.wav
    ```

*   **使用 NVIDIA CUDA 后端：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe --text "你好，欢迎使用纯C++推理的语音合成系统。" --lang "zh" --device CUDA0 --out scratch/output_cuda.wav
    ```

*   **使用 Intel SYCL 后端：**
    ```bash
    .\build-release\bin\gpt-sovits-test-pipeline.exe --text "你好，欢迎使用纯C++推理的语音合成系统。" --lang "zh" --device SYCL0 --out scratch/output_sycl.wav
    ```

---

## 📊 性能基准 (Benchmark)

在本项目中，我们对三种后端在多句长文本下的合成速度（实时率 RTF）和系统资源占用进行了全面测试。

> [!NOTE]
> **实时率 (RTF, Real-Time Factor)** = 语音合成消耗时间 (s) / 生成音频长度 (s)。RTF 越低说明合成速度越快。若 RTF = 0.1，意味着合成 10 秒的音频只需 1 秒。

测试的硬件配置与结果报告详情，请参阅生成的测试报告：
👉 **[后端性能基准报告](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/scratch/benchmark_report.md)**

---

## 💻 核心算子覆盖设计 (`src/ops`)

为避免污染底层的 `ggml` 核心代码库，我们设计了外置的自定义算子注册机制：
*   **[ops.cpp](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/src/ops/ops.cpp)**: 负责在后端推理图计算开始前拦截 `MUL_MAT + IM2COL` (Conv1D), `CONV_TRANSPOSE_1D`, `SOFT_MAX` 等算子，并将计算任务分发给各硬件平台的自定义加速实现。
*   **[ops-sycl/ops_sycl.cpp](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/src/ops/ops-sycl/ops_sycl.cpp)**: 包含 Intel GPU 的具体 DPC++ 算子实现。
*   **[ops-cuda/ops_cuda.cu](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/src/ops/ops-cuda/ops_cuda.cu)**: 包含 NVIDIA GPU 的 CUDA 算子实现。
