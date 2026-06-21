# GPT-SoVITS.cpp Bug & Hotfix Registry
# GPT-SoVITS.cpp 已知 Bug 与修复记录

本文档详细记录了 `GPT-SoVITS.cpp` 项目在开发与优化过程中遇到的核心 Bug 及其解决方案。这包括内存越界、算子执行断言、跨后端适配、以及声音失真（“电音/破音”）等关键问题的排查与修复过程。

---

## 目录 / Table of Contents
1. [CPU 反卷积 (ConvTranspose 1D) 衬垫越界与断言崩溃 / CPU ConvTranspose 1D Padding Assertion Crash](#1-cpu-反卷积-convtranspose-1d-衬垫越界与断言崩溃--cpu-convtranspose-1d-padding-assertion-crash)
2. [GPU 计算图执行顺序混乱 / GPU Compute Graph Execution Out-Of-Order Bug](#2-gpu-计算图执行顺序混乱--gpu-compute-graph-execution-out-of-order-bug)
3. [GGML SYCL 算子库 Native SoftMax 精度失效导致电音/破音 / GGML SYCL Native SoftMax Producing Extreme Values / NaNs](#3-ggml-sycl-算子库-native-softmax-精度失效导致电音破音--ggml-sycl-native-softmax-producing-extreme-values--nans)
4. [VITS 卷积权重转置与 GGML 连续性布局限制 / VITS Weight Permutation & GGML view_2d Contiguous Layout Constraint](#4-vits-卷积权重转置与-ggml-连续性布局限制--vits-weight-permutation--ggml-view_2d-contiguous-layout-constraint)

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
