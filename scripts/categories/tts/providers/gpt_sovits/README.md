# GPT-SoVITS GGUF Export

`process.py` is the only supported conversion and quantization entry point. It
processes one explicit source artifact at a time and requires the exact model
version for T2S and VITS. Attention heads are read from checkpoint config.

```powershell
python scripts/categories/tts/providers/gpt_sovits/process.py --model-type t2s --version v3 --src s1v3.ckpt --output t2s_v3_q4_0.gguf --quantize Q4_0
```

`--quantize` accepts `F16`, `Q4_0`, `Q4_K`, `Q4_K_M`, and `Q8_0`. The common
library includes a GGML-compatible Q4_K encoder. `Q4_K_M` uses the same Q4_K
storage format with Q8/F16 fallbacks for sensitive or incompatible tensors.
`process.py` owns GPT-SoVITS tensor classification and Conv1D packing while
reusing the model-independent executor in `scripts/common`.

An optional precision policy can override candidate combinations by provider
tensor role or sensitivity while retaining GGML block-size validation:

```powershell
python scripts/categories/tts/providers/gpt_sovits/process.py --model-type t2s --version v3 `
  --src s1v3.ckpt --output t2s_v3_custom.gguf --quantize Q4_K `
  --quant-policy scripts/common/policy.example.json
```

Classic VITS versions require an explicit `s2D.pth,s2G.pth` source pair. V3 and
V4 require an explicit `s2G.pth,vocoder` pair. The exporter never selects a
checkpoint from a directory or infers a model version from its filename.
