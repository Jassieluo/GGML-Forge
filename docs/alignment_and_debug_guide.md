# GPT-SoVITS.cpp V1/V2 对齐与调试备忘录

本项目是一个基于 GGML 的高性能 C++ 推理引擎，在设计风格上完全在模仿 PyTorch 推理端的行为，且支持 GGUF 直通加载与硬件平台优化。本备忘录记录了项目的工作区物理路径、环境配置、已解决的关键 Bug 以及开新会话时的续接测试命令。

---

## 1. 物理工作区路径与环境配置

* **C++ 推理工作区**：
  `D:\Projects\CMake Projects\GPT-SoVITS.cpp`
* **Python 原版 GPT-SoVITS 源码工作区**：
  `D:\Projects\PycharmProjects\GPT-SoVITS-main`
* **Python Conda 环境解释器**：
  `D:\anaconda3\envs\gpt-sovits\python.exe`
* **C++ 一键编译 Preset 命令**（英特尔 oneAPI/SYCL/MKL 编译）：
  ```powershell
  cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16
  ```

---

## 2. 核心架构设计与优化

### 2.1 动态 KV Cache 长度与静态加速的平衡
为了支持 **CUDA Graph** 等硬件级静态图推理加速，本项目的注意力计算采用了“**固定计算长度 + 填充屏蔽遮罩（Mask）**”的设计：
* **去硬编码**：固定了在 `attention.cpp` 和 `gpt_t2s.cpp` 里原本硬编码 `512` 最大长度的做法，改为动态通过 `kv_k->ne[1]` 读取当前分配的 KV 缓存物理维度（当前 GGUF 模型默认读取为 512）。
* **静态化加速**：在自回归解码步（`total_decoded > 0`），注意力前向计算的 `kv_len` 维持为 `kv_k->ne[1]`（静态图尺寸）。通过生成一个 `kv_k->ne[1]` 长度 of 1D 遮罩（Mask），在超出当前已解码有效长度 `total_len` 后的区间填充 `-1e4f` 溢出值，从而屏蔽无效缓存。

### 2.2 重复惩罚机制 (Repetition Penalty)
* 模型在自回归采样（`sample_logits`）时，会将生成的 `next_token` 不断推入 `current_audio_ids` 中，以此实现重复性惩罚。
* 注意：在 Step 0 阶段，历史列表里已存有参考音频的 semantic tokens 序列。这使得第一步预测时，参考音频中已包含的词的 logits 会被除以 `1.35`，从而改变首个 Token 的最大概率预测。

---

## 3. V1 调试对齐核心突破（当前会话关键成果）

1. **BERT 通道转置对齐**：
   * **现象**：PyTorch 导出的 `bert_feature` 为 `[1024, seq_len]` (Channel-first)，但在喂给自回归模型前，PyTorch 侧执行了 `.transpose(1, 2)` 为 `[seq, 1024]` 排布。
   * **解决**：在 Python 侧 dump 输入特征时，补上了这一转置；C++ 端加载的数据已完全对位，首步 Logits 的分布达成了极高的一致性。
2. **Causal Mask 行列顺序修正**：
   * **解决**：将之前因调试错置的 causal mask 计算索引还原为行优先 `r * seq_len + c`，彻底解决了 V1 之前死循环生成 2700 多个 Token 或无故早停的 Bug。
   * **当前状态**：在随机种子下，V1 语义 Token 长度在 Segment 1, 2, 3 中分别生成为正常自然的 `95`、`92`、`92` 个，生成的音频时长和音色还原度均已完全达标。

---

## 4. 调试测试与对齐工具指令

### 4.1 使用 Python 黄金特征覆盖（Override）进行 C++ 管道对齐测试
我们在 C++ 侧内置了调试 Override 开关。可以通过加载 Python 侧导出的纯净特征文件来规避 G2P/前端文本的分歧，专攻自回归与 VITS 侧的对齐：

```powershell
# 1. 开启贪心解码环境变量以做物理对齐核查
$env:GPT_SOVITS_DEBUG = 1
$env:T2S_TOP_K = 1
$env:T2S_TEMPERATURE = 0

# 2. 设置 Override 特征文件路径
$env:T2S_OVERRIDE_PHONES_FILE = "D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_phones.bin"
$env:T2S_OVERRIDE_BERT_FILE = "D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_bert.bin"
$env:T2S_OVERRIDE_PROMPT_SEMANTIC = "D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_prompt_semantics.bin"

# 3. 运行测试管道输出音频
.\build-x64-windows-cuda-sycl-cpu-dl-release-f16\bin\gpt-sovits-test-pipeline.exe `
  --cpu `
  --t2s models/gpt_sovits/weights/t2s/t2s_v1_q4_0.gguf `
  --vits models/gpt_sovits/weights/vits/vits_v1_q4_0.gguf `
  --out scratch/test_v1_cpu_override.wav
```

### 4.2 Python 侧提取首步 Logits 与采样测试指令
如果想在 PyTorch 侧验证某一步的推理，可以直接运行以下极简的测试命令（已解决 FP16 混合精度数据类型报错）：

```bash
# 进入 Python 项目目录并启动环境
cd "D:\Projects\PycharmProjects\GPT-SoVITS-main"
D:\anaconda3\envs\gpt-sovits\python.exe -c "
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'GPT_SoVITS'))
from inference_webui import change_gpt_weights, change_sovits_weights, t2s_model
change_gpt_weights(gpt_path='GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')

# 加载覆盖特征
phones = np.fromfile(r'D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_phones.bin', dtype=np.int32)
bert = np.fromfile(r'D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_bert.bin', dtype=np.float32).reshape(-1, 1024)
prompt_semantics = np.fromfile(r'D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\py_prompt_semantics.bin', dtype=np.int32)

all_phoneme_ids = torch.LongTensor(phones).unsqueeze(0).cuda()
prompt = torch.LongTensor(prompt_semantics).unsqueeze(0).cuda()
bert_tensor = torch.FloatTensor(bert).unsqueeze(0).half().cuda()

with torch.no_grad():
    x = t2s_model.model.ar_text_embedding(all_phoneme_ids)
    x = x + t2s_model.model.bert_proj(bert_tensor)
    x = t2s_model.model.ar_text_position(x)
    
    y = prompt
    y_emb = t2s_model.model.ar_audio_embedding(y)
    y_pos = t2s_model.model.ar_audio_position(y_emb)
    xy_pos = torch.concat([x, y_pos], dim=1)
    
    x_len, y_len = x.shape[1], y_emb.shape[1]
    src_len = x_len + y_len
    x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
    x_attn_mask_pad = torch.nn.functional.pad(x_attn_mask, (0, y_len), value=True)
    y_attn_mask = torch.nn.functional.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
    xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).unsqueeze(0).expand(16, -1, -1).view(1, 16, src_len, src_len).cuda()
    
    xy_dec, _, _ = t2s_model.model.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
    logits = t2s_model.model.ar_predict_layer(xy_dec[:, -1])
    logits_sliced = logits[0, :-1].cpu().numpy()
    
    # 打印前五个最大可能性的词和其 logits 值
    top_indices = np.argsort(logits_sliced)[::-1][:10]
    print('=== PyTorch Step 0 Top 10 Logits ===')
    for rank, idx in enumerate(top_indices):
        print(f'  Rank {rank}: index={idx}, value={logits_sliced[idx]}')
"
```
