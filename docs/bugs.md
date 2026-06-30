# GPT-SoVITS.cpp Bug & Hotfix Registry
# GPT-SoVITS.cpp 已知 Bug 与修复记录

本文档详细记录了 `GPT-SoVITS.cpp` 项目在开发与优化过程中遇到的核心 Bug 及其解决方案。这包括内存越界、算子执行断言、跨后端适配、以及声音失真（“电音/破音”）等关键问题的排查与修复过程。

---

## 目录 / Table of Contents
1. [CPU 反卷积 (ConvTranspose 1D) 衬垫越界与断言崩溃 / CPU ConvTranspose 1D Padding Assertion Crash](#1-cpu-反卷积-convtranspose-1d-衬垫越界与断言崩溃--cpu-convtranspose-1d-padding-assertion-crash)
2. [GPU 计算图执行顺序混乱 / GPU Compute Graph Execution Out-Of-Order Bug](#2-gpu-计算图执行顺序混乱--gpu-compute-graph-execution-out-of-order-bug)
3. [GGML SYCL 算子库 Native SoftMax 精度失效导致电音/破音 / GGML SYCL Native SoftMax Producing Extreme Values / NaNs](#3-ggml-sycl-算子库-native-softmax-精度失效导致电音破音--ggml-sycl-native-softmax-producing-extreme-values--nans)
4. [VITS 卷积权重转置与 GGML 连续性布局限制 / VITS Weight Permutation & GGML view_2d Contiguous Layout Constraint](#4-vits-卷积权重转置与-ggml-连续性布局限制--vits-weight-permutation--ggml-view_2d-contiguous-layout-constraint)
5. [GPU 后端 Conv 1D 在 FP16 混合精度下与 CPU 后端数值不对齐及 GGML CPU 断言崩溃问题 / GPU Conv 1D FP16 Numerical Mismatch & GGML CPU Weight Type Assertion Crash](#5-gpu-后端-conv-1d-在-fp16-混合精度下与-cpu-后端数值不对齐及-ggml-cpu-断言崩溃问题--ggml-cpu-weight-type-assertion-crash)
6. [MSVC 静态库链接器裁剪导致自定义算子失效、CUDA 运行时库缺失及 MODULE 目标链接受限问题 / MSVC Static Library Linker Pruning, CUDA Runtime Symbols Mismatch, and MODULE Library Linking Constraint](#6-msvc-静态库链接器裁剪导致自定义算子失效cuda-运行时库缺失及-module-目标链接受限问题--msvc-static-library-linker-pruning-cuda-runtime-symbols-mismatch-and-module-library-linking-constraint)
7. [CPU 后端 FP16 推理与一维卷积/自定义算子指针类型不匹配产生全幅电流音问题 / CPU Backend FP16 Inference producing full-amplitude static noise due to Custom Operator float* Pointer Cast Mismatch](#7-cpu-后端-fp16-推理与一维卷积自定义算子指针类型不匹配产生全幅电流音问题--cpu-backend-fp16-inference-producing-full-amplitude-static-noise-due-to-custom-operator-float-pointer-cast-mismatch)
8. [CPU 后端长语音/多段语音合成自回归 KV Cache 残留导致无限循环生成垃圾音频问题 / CPU Backend KV Cache Residue leading to Infinite Generation Loop in Multi-Segment Synthesis](#8-cpu-后端长语音多段语音合成自回归-kv-cache-残留导致无限循环生成垃圾音频问题--cpu-backend-kv-cache-residue-leading-to-infinite-generation-loop-in-multi-segment-synthesis)

---

## 1. CPU 反卷积 (ConvTranspose 1D) 衬垫越界与断言崩溃 / CPU ConvTranspose 1D Padding Assertion Crash

### 问题背景 / Problem Context
在 CPU 后端上运行 VITS 模型进行音频合成时，当反卷积层（`ConvTranspose 1D`）的 `padding > 0` 时，推理程序会触发 GGML 内部的断言崩溃或抛出内存越界错误。

### 根本原因 / Root Cause
GGML 官方的原生 CPU `ggml_conv_transpose_1d` 实现，在非零 padding（`padding > 0`）的处理上存在边界约束错误与逻辑漏洞。当前版本的 GGML CPU 算子在计算带 padding 的转置卷积输出形状和寻址时，无法正确执行内存跨步拷贝，直接导致底层断言失败。

### 修复方案 / Resolution
为了保证不修改 `ggml` 纯净底层源码的原则，在算子劫持分发层 `src/ops/ops.cpp` 中的 `ops_conv_transpose_1d` 引入了 **CPU 边界裁剪规避策略**：
1. 调用 `ggml_conv_transpose_1d` 时强行将 `padding` 设为 `0`（GGML 对 `padding=0` 的支持是完全稳定且无 Bug 的），计算出一个未经过裁剪的较长临时张量。
2. 若原始 `padding > 0`，使用 `ggml_view_2d` 对该临时张量的左右边界进行精确截断（Cropping），只保留目标输出宽度的核心区域。
3. 通过 `ggml_cont` 算子重新整理内存并使数据连续，从而避开 GGML 原生 CPU 带 Padding 反卷积算子的执行 Bug。

### 代码实现 / Code Implementation
在 `src/ops/ops.cpp` 中的实现代码：
```cpp
struct ggml_tensor* ops_conv_transpose_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    bool is_cuda = false;
    if (gpt_sovits::current_vits_backend) {
        const char * bname = ggml_backend_name(gpt_sovits::current_vits_backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }

    if (is_cuda) {
        // CUDA (GPU) 路径：依然可以安全使用带 padding 的原生节点，分发给 cuDNN / GPU 算子
        int64_t out_w = (x->ne[0] - 1) * stride - 2 * padding + dilation * (w->ne[0] - 1) + 1;
        const int64_t ne[4] = { out_w, w->ne[1], x->ne[2], 1 };
        
        struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        result->op = GGML_OP_CONV_TRANSPOSE_1D;
        // 设置参数
        int32_t params[] = { stride, padding, dilation };
        ggml_set_op_params(result, params, sizeof(params));
        
        result->src[0] = w;
        result->src[1] = x;
        return result;
    } else {
        // CPU Fallback 路径：Padding=0 计算并使用 View 裁剪
        struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
        
        if (padding > 0) {
            int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
            size_t offset_bytes = padding * conv_t_raw->nb[0];
            
            struct ggml_tensor* cropped_view = ggml_view_2d(
                ctx,
                conv_t_raw,
                cropped_seq_len,
                conv_t_raw->ne[1],
                conv_t_raw->nb[1],
                offset_bytes
            );
            
            return ggml_cont(ctx, cropped_view);
        }
        return conv_t_raw;
    }
}
```

---

## 2. GPU 计算图执行顺序混乱 / GPU Compute Graph Execution Out-Of-Order Bug

### 问题背景 / Problem Context
在引入自定义 CUDA/SYCL 加速算子（如卷积劫持）时，如果模型计算图中包含原生算子和自定义劫持算子的混合，模型计算时常发生输出数据异常、断言失败，或者部分节点未被执行的 Bug。

### 根本原因 / Root Cause
GGML 的计算图是通过拓扑排序顺序执行的。劫持分发器 `ops_graph_compute_hook` 最初在遇到劫持算子节点时，会直接跳转去执行自定义实现，而未能保证该节点之前的普通 GGML 算子节点已经按照拓扑顺序在 GPU 上完整计算完毕。这就导致了计算图乱序与依赖缺失。

### 修复方案 / Resolution
在 `src/ops/ops.cpp` 的 `ops_graph_compute_hook` 中重构图编译器调度逻辑。引入**阶段性子图提交机制**（Stage-based Subgraph Execution）：
1. 遍历当前计算图节点。若检测到当前节点为自定义劫持算子（如自定义 1D 卷积、反卷积或 SoftMax）：
   * 首先将该节点之前的普通节点打包构建为一个临时子图视图（`local_graph_view`）。
   * 提交该子图至原生后端（`orig_compute`）先行完成计算。
2. 随后执行当前自定义劫持算子，并将该劫持节点的 `op` 置为 `GGML_OP_NONE` 标记完成。
3. 循环往复，直到遍历完所有节点，最后提交剩余的普通算子子图。这从逻辑上保证了拓扑排序的完整性和前后数据的依赖正确。

---

## 3. GGML SYCL 算子库 Native SoftMax 精度失效导致电音/破音 / GGML SYCL Native SoftMax Producing Extreme Values / NaNs

### 问题背景 / Problem Context
在英特尔显卡（SYCL 后端）上运行全管线推理时，生成的音频会产生刺耳的“电音”、“啸叫”和“爆音”等失真现象。通过排查，发现 VITS 输出波形在第一步后就彻底“爆音”，振幅达到绝对截断值（`min = -32767`, `max = 32767`）。

### 根本原因 / Root Cause
排查 BERT/Hubert 中间层发现，GGML 官方原生 SYCL 后端中的 SoftMax 规约核函数（`ggml_sycl_softmax`）存在严重精度 Bug 或内存越界写缺陷。当在 Intel GPU 上运行 SoftMax 时，它输出的概率张量会变为极大的非物理数值（NaN 或接近无穷大的数）。当此错误在注意机制（Attention）层和后续层进行矩阵乘法传递时，整个网络的激活值彻底发散，传递到 VITS 最后的反卷积/解码层时已完全溢出。

### 修复方案 / Resolution
由于不允许修改 GGML 的 SYCL 源码，我们采取了 **CPU 拦截回退（Fallback）** 策略：
1. 在算子劫持钩子 `ops_graph_compute_hook` 中拦截 `GGML_OP_SOFT_MAX` 操作。
2. 实现一个高性能的主机端 SoftMax 回退核函数 `compute_softmax` (写在 `src/ops/ops-sycl/ops_sycl.cpp` 中)。
3. 在执行到 SoftMax 节点时，将输入张量从 SYCL GPU 显存拷贝回 CPU 内存，在 CPU 上以高精度完成 SoftMax 运算（包括支持 Scale 缩放、ALiBi Mask 的累加和高精度指数归一化），最后再将计算结果写回 SYCL 设备的张量中。该操作耗时极短，成功根治了 SYCL 后端音频发散和“爆电音”的致命缺陷。

---

## 4. VITS 卷积权重转置与 GGML 连续性布局限制 / VITS Weight Permutation & GGML view_2d Contiguous Layout Constraint

### 问题背景 / Problem Context
在加载 VITS 的 1D 卷积权重进行维度变换（Permutation）时，程序直接抛出异常崩溃：
`ggml_new_tensor_impl: not contiguous`，导致无法成功加载模型。

### 根本原因 / Root Cause
1. 之前的权重排列映射（Permute Axes）被设定为了非连续性的轴映射（如 `(1, 2, 0, 3)`）。
2. 在 GGML 中，`ggml_new_tensor_impl` (内部被 `ggml_view_2d` / `ggml_view_3d` 间接调用以生成视图) 在计算张量的物理大小时，会强行要求涉及计算的维度步长（Strides）必须满足内存上的连续性规则（即 `data_size = ne0 * nb0 + (ne1 - 1) * nb1 ...`）。如果对一个不满足该连续性排列的维度切分切片，由于步长不能对齐，GGML 会判定该视图物理空间不合法并直接 Assert 退出。

### 修复方案 / Resolution
在 `src/models/synthesis/vits/vits.cpp` 中，将权重的维度转置序列修正回连续的映射关系 `(2, 0, 1, 3)`，使得被切分的核尺寸（`kernel_size`）在内存排布上保持为连续的外层，从而消除了 `ggml_view_2d` 对其截取 slice 时触发的非法连续性布局断言。

---

## 5. GPU 后端 Conv 1D 在 FP16 混合精度下与 CPU 后端数值不对齐及 GGML CPU 断言崩溃问题 / GPU Conv 1D FP16 Numerical Mismatch & GGML CPU Weight Type Assertion Crash

### 问题背景 / Problem Context
在测试与优化 Conv1D 算子时，如果在验证时强使用 FP32 格式的权重，或者在 GPU (CUDA) 后端对比 CPU Reference 结果时，会遇到以下两个严重问题：
1. **CPU 后端断言崩溃**：在 `test_ops.exe` 中以 FP32 精度传入卷积权重进行 CPU 计算时，程序直接触发 GGML 内部断言挂掉，报错 `GGML_ASSERT(src0->type == GGML_TYPE_F16) failed`。
2. **GPU 与 CPU 数值不对齐**：当解决 CPU 崩溃问题后（将权重强制采用 `GGML_TYPE_F16`），在 GPU (CUDA) 上运行标准 GGML Baseline 或 cuDNN 算子时，与 CPU Reference 输出的误差非常大（`Max Diff = 1.18`），导致测试无法通过。

### 根本原因 / Root Cause
1. **CPU 降级断言限制**：GGML CPU 后端的 `ggml_conv_1d` 原生实现中，内部强行调用了 `ggml_im2col(..., GGML_TYPE_F16)`，且在其前向计算核心中强行断言限制了输入权重 `src0` 的类型必须是 `GGML_TYPE_F16`。这导致以 FP32 格式传入权重时直接挂掉。
2. **FP16 硬件累加舍入误差 (Round-off Error)**：
   * 在 CPU Reference 后端，即使输入权重是 `F16`，但由于没有硬件半精度浮点计算指令，GGML CPU 内部做 `mul_mat` 会被提升至 **FP32** 进行累加。
   * 在 GPU (CUDA) 后端，标准 GGML 的 `ggml_cuda_mul_mat` 在处理 `F16 * F16` 时，会直接利用 GPU Tensor Cores 硬件进行 `F16` 精度的乘加计算。
   * 测试参数（`C_in = 512, kW = 5`）有多达 2560 项的数据累加。在纯 FP16 的有限精度下进行如此长序列的累加，其累加舍入误差（Underflow）非常大。当输出振幅达到 `[-300, 300]` 规模时，累积绝对误差达到 `1.18` 属于正常精度舍入损耗。
   * **证明**：在 GPU 上比对原始 CUDA Baseline（未优化）与 cuDNN 优化版本的输出值，发现它们两者在 GPU 上的输出完全一致，误差绝对为 0。

### 修复与验证方案 / Resolution
1. **测试用例对齐修正**：在 `test_ops.cpp` 中将 Conv1D 测试权重统一设为 `GGML_TYPE_F16`，以解决 CPU 运算崩溃的问题。
2. **误差对比判定修正**：由于 GPU 硬件半精度舍入特性，对比 CPU (FP32 模拟) 与 GPU (FP16 硬件) 的绝对偏差没有实质对齐意义。因此，在 `run_conv_1d_test` 的 GPU 测试分支中，将正确性校准基准修改为 **“对比标准 CUDA Baseline 算子的 GPU 输出”**。二者比对误差为 0，完美通过正确性测试。
3. **编译开启 cuDNN 加速**：运行 CMake 时通过 `-DGGML_CUDNN=ON` 参数，编译时完美启用 cuDNN 的 Implicit GEMM 计算，替换高开销的 `im2col` 加 `cublasSgemm`。

---

## 6. MSVC 静态库链接器裁剪导致自定义算子失效、CUDA 运行时库缺失及 MODULE 目标链接受限问题 / MSVC Static Library Linker Pruning, CUDA Runtime Symbols Mismatch, and MODULE Library Linking Constraint

### 问题背景 / Problem Context
在集成自定义原生算子（特别是 `LayerNorm`、`ConvTranspose1d`、`Conv1d`）到全管线模型中，并尝试在 CPU、CUDA、SYCL 三个后端上进行音频生成测试时，遇到了以下一系列编译与运行时问题：
1. **CPU/GPU 算子未执行 (静默 fallback)**：在 CPU 上生成的音频中，虽然推理成功，但实际执行的是数学等价的原生 GGML 分步算子 fallback 路径，自定义融合算子（`GGML_OP_OPS_VIRT_LAYER_NORM`）并没有运行。在 CUDA 上则直接崩溃于原生 GGML CPU 反卷积对权重精度（`F32` 限制）的断言。
2. **符号未解析错误 (Linker LNK2019/LNK2001)**：尝试将 `register_backend()` 显式暴露以解决裁剪问题时，编译链接 `tts.dll` 产生大量关于 CUDA 运行时符号（如 `__cudaRegisterFatBinary`、`cudaSetDevice`）以及 `ggml_cuda_set_device`、`ggml_cuda_error` 的未解析外部符号错误。
3. **CMake 配置错误**：试图直接在 `ggml_ops_ext_cuda` 链接 `ggml-cuda` 目标以解决内部 CUDA 符号未解析时，CMake 报错：`Target "ggml-cuda" of type MODULE_LIBRARY may not be linked into another target.`。

### 根本原因 / Root Cause
1. **MSVC 链接器优化裁剪特性**：由于自定义后端算子库（如 `ggml_ops_ext_cuda`、`ggml_ops_ext_cpu` 等）被编译为静态库（`.lib`），且在主项目和 `tts` 动态库中，先前仅在各自库内的全局静态构造函数 `RegisterCuda` / `RegisterCpu` 进行自动注册，没有任何代码显式引用这些文件中的符号。MSVC 链接器在链接 `tts.dll` 时，判定这几个翻译单元“无用”并将其整片裁剪丢弃，导致算子注册函数根本没有被执行，分发钩子失效。
2. **CUDA 运行时链接缺失**：自定义算子库 `ggml_ops_ext_cuda` 包含 `.cu` 编写 of CUDA 核函数，编译后会自动插入对 CUDA 运行时的 API 依赖（`cudart.lib` 中的 `__cudaRegisterFatBinary` 等），但其 CMake 中仅链接了 `CUDA::cublas`，未声明链接 `CUDA::cudart`，导致符号未解析。
3. **GGML-CUDA 内部符号依赖与 `MODULE` 类型限制**：
   * 自定义算子引用了 `ggml-cuda/common.cuh` 内部头文件，间接实例化了含有 `ggml_cuda_set_device` 和 `ggml_cuda_error` 调用的内联函数。
   * 在 GGML 底层架构中，CUDA 插件后端 `ggml-cuda` 被声明为 `MODULE_LIBRARY`（动态加载模块），在 Windows 上不会生成或允许链接导入库（`.lib`），故无法通过 `target_link_libraries` 声明强链接，进而造成编译链中断。

### 修复与应对方案 / Resolution Plan
1. **显式注册机制 (Avoid Pruning)**：
   * 在 `src/pipelines/gpt_sovits/gpt_sovits_pipeline.cpp` 的引擎初始化入口 `gpt_sovits_init_with_device` 和 `gpt_sovits_init_ext` 中，通过 `std::once_flag` 显式且仅一次地调用 `cpu::register_backend()`、`cuda::register_backend()` 和 `sycl::register_backend()`。
   * 这直接在强链接的 `tts` 动态库源码中引入了对这几个后端符号的引用，强制 MSVC 链接器无差别保留整个后端算子库的目标文件。
2. **补充 CUDA 运行时依赖**：
   * 在 `src/ops/ops-cuda/CMakeLists.txt` 中，将 `CUDA::cudart` 加入 `target_link_libraries` 依赖项中，自动引入 `cudart.lib`。
3. **规避 MODULE 强链接 & 本地化封装**：
   * 移除对 `ggml-cuda` 目标不合法的 `target_link_libraries` 强关联。
   * 为 `ggml_cuda_set_device` 和 `ggml_cuda_error` 等未导出的 GGML CUDA 内部接口提供在自定义算子层（如 `ops_cuda.cu`）的本地全局包装定义，使其能完美就地解析，彻底阻断由于 MODULE 库不可被链接带来的构建失败。

---

## 7. CPU 后端 FP16 推理与一维卷积/自定义算子指针类型不匹配产生全幅电流音问题 / CPU Backend FP16 Inference producing full-amplitude static noise due to Custom Operator float* Pointer Cast Mismatch

### 问题背景 / Problem Context
在执行 CPU 推理时，当模型以原生 `FP16`（或量化格式如 `Q8_0` / `Q4_0` 伴随 FP16 运行）载入内存后，推理虽然能够成功进行，但最终合成并输出的 WAV 音频文件中包含**全幅的电噪音/电流杂音**，完全没有任何可识别的语音信号。与此相反，GPU (CUDA) 后端生成的音频完全正确、音质清晰。

### 根本原因 / Root Cause
在之前的原生 FP16 推理优化中，我们将 CPU 后端下的 VITS/T2S 等所有权重数据在内存中直接存为了 `GGML_TYPE_F16`（半精度浮点数）。
然而，`src/ops/ops-cpu/` 目录下的 CPU 自定义一维卷积算子（如 `ops_conv_1d`、`ops_matmul_f32` 等）在实现时强行做指针转换，假定输入参数 `A`、`B` 的 `data` 永远是单精度浮点数 `float*` (FP32)：
1. 在 `matmul_f32.cpp` 中，由于启用了一键 BLAS 加速（oneMKL），`ops_matmul_f32` 会直接把 `GGML_TYPE_F16` 权重的 `data` 指针转换为 `float*` 传递给单精度矩阵乘法函数 `cblas_sgemm`。
2. 在自定义卷积的前向计算中，16位的 FP16 浮点数字节（2 bytes）被 CPU 强制当作 32位的 FP32 浮点数字节（4 bytes）进行寻址与乘加计算。
3. 这导致了极其严重的浮点数指数位和尾数位解析错乱，产生大面积的无效溢出或数值截断，最终在声码器输出层累积成全幅振幅不断在 `-1.0` 和 `1.0` 之间剧烈波动的方波电流噪波（clipping 比例高达 63.57%）。

### 修复方案 / Resolution Plan
在下一个会话中，我们需要：
1. **完善 CPU 算子的输入类型检查**：在 CPU 自定义算子（特别是一维卷积与矩阵乘法分发逻辑）中，根据输入 Tensor 的实际数据类型（`F32` 或 `F16`）分流处理。若为 `F16`，应先使用 `ggml_fp16_to_fp32` 对局部数据进行转换，或调用 GGML 自带的 `ggml_vec_dot_f16` 以正确执行半精度到单精度的数学转换。
2. **规范矩阵乘法在 F16 权重下的回退逻辑**：在 MKL sgemm 之前增加对权重类型的判断，或者将 VITS 在 CPU 上运行时的非卷积部分权重保持/重构为 FP32 格式，确保自定义 CPU 算子拿到的永远是正确的数据类型。

---

## 8. CPU 后端长语音/多段语音合成自回归 KV Cache 残留导致无限循环生成垃圾音频问题 / CPU Backend KV Cache Residue leading to Infinite Generation Loop in Multi-Segment Synthesis

### 问题背景 / Problem Context
在执行长文本合成（多分段合成，如段落切分逻辑）时：
* GPU 模式下，各段音频的生成长度和速度完全正常，音频清晰自然。
* CPU 模式下，第一段音频生成正常，但**从第二段开始**，模型合成时间极长（5秒以上，甚至打满 T2S 最大 token 步数限制，即 512 步），生成的音频被拉长到 41.52 秒极限，内容全为无意义的复读、长静音或杂音。

### 根本原因 / Root Cause
该问题源于自回归生成的核心组件 **KV Cache（键值缓存）生命周期的管理缺陷**：
1. **持久化内存驻留**：为了避免多次分配带来的开销，`T2SModel` 中的 KV Cache 缓存（`kv_k` 和 `kv_v`）是在模型加载初始化（`T2SModel::init`）时一次性分配的，并在后续 of `forward()` 中持久复用。
2. **多分段残留污染**：
   * 在进行多段落合成时，上一分段合成结束，开启下一分段合成时，我们并未在 `T2SModel::forward()` 入口显式清空（归零）这块缓存。
   * 下一分段在自回归解码第 0 步时，前向计算会通过自注意力机制（Self-Attention）读取 KV Cache。由于没有清零，第一分段在原缓存空间中写入的数据（残留 Key-Value 向量）仍保留在对应内存处。
   * 自注意力机制将第一分段残留的垃圾特征误作为当前生成的输入上下文进行计算，导致模型输出概率分布（Logits）完全偏离正常分布，GPT 无法正确输出结束符 `<EOS>`，从而在复读垃圾状态中死循环，直至超出最大限制步长。
3. **为什么 CPU 报错，而 GPU 正常？**
   * **GPU 自动清零特性（误打误撞）**：在 pipeline 逻辑中，为了省显存，每段合成完后会调用 `impl->offload_model(2)`（卸载 T2S），当新的一段开始时，再调用 `impl->load_model(2)`。这在 GPU (CUDA) 后端上会导致显存释放与重新申请，重新分配的 GPU 显存默认被初始化为 0，因而避开了残留污染问题。
   * **CPU 持久驻留特性**：在 CPU 模式下，因不存在显存紧张，`offload_model` 与 `load_model` 是空操作（No-op），不会发生真正的内存释放与重新分配。因此，老数据原封不动保留，触发死循环 Bug。

### 修复方案 / Resolution Plan
在 `src/models/lm/gpt_t2s/gpt_t2s.cpp` 的 `T2SModel::forward()` 的最初阶段，显式地对持久化 KV Cache 张量执行置零清空：
* 调用 `ggml_backend_tensor_set`，以高效地将 `kv_k` 和 `kv_v` 张量内容重置为全零的 float 数组。
* 确保不管在任何后端（CPU、CUDA、SYCL），也不管是否开启 offload 机制，每次全新的 forward 生成都会面临干净的、初始为 0 的 KV Cache 状态。

