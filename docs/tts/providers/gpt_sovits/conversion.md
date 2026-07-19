# GPT-SoVITS GGUF Export

`process.py` is the only supported conversion and quantization entry point. It
processes one explicit source artifact at a time and requires the exact model
version for T2S and VITS. Attention heads are read from checkpoint config.

```powershell
python scripts/conversion/categories/tts/providers/gpt_sovits/process.py --model-type t2s --version v3 --src s1v3.ckpt --output t2s_v3_q4_0.gguf --quantize Q4_0
```

`--quantize` accepts `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`,
`Q4_K_M`, `Q8_0`, and `MXFP4`. The common library includes a GGML-compatible
Q4_K encoder. `Q4_K_M` uses the same Q4_K
storage format with Q8/F16 fallbacks for sensitive or incompatible tensors.
`process.py` owns GPT-SoVITS tensor classification and Conv1D packing while
reusing the model-independent executor in `scripts/conversion/common`.

The runtime can also consume Q2_K, Q3_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS
ConvND weights produced by external GGUF tools. They are intentionally not
listed as script targets because the pinned Python GGUF package currently
provides decoders but no encoders for those formats.

An optional precision policy can override candidate combinations by provider
tensor role or sensitivity while retaining GGML block-size validation:

```powershell
python scripts/conversion/categories/tts/providers/gpt_sovits/process.py --model-type t2s --version v3 `
  --src s1v3.ckpt --output t2s_v3_custom.gguf --quantize Q4_K `
  --quant-policy scripts/conversion/common/policy.example.json
```

Classic VITS versions require an explicit `s2D.pth,s2G.pth` source pair. V3 and
V4 require an explicit `s2G.pth,vocoder` pair. The exporter never selects a
checkpoint from a directory or infers a model version from its filename.

V2Pro uses the upstream ERes2NetV2 speaker-verification checkpoint as a
separate provider component. Conversion folds every inference BatchNorm2D into
its preceding Conv2D and emits the 20480-dimensional `forward3` encoder:

```powershell
python scripts/conversion/categories/tts/providers/gpt_sovits/process.py `
  --model-type speaker_encoder `
  --src GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt `
  --output models/tts/gpt_sovits/weights/speaker_encoder/eres2net_v2_q4_0.gguf `
  --quantize Q4_0 `
  --quant-policy scripts/conversion/categories/tts/providers/gpt_sovits/speaker_encoder_q4.json
```

The native frontend matches `Kaldi.fbank` with 16 kHz input, 80 mel bins, and
zero dither. The resulting artifact is shared by V2Pro and V2ProPlus configs.
The quality policy stores robust later-stage Conv2D rows as Q4_0 while retaining
Q8_0/F16 for sensitive or block-incompatible tensors. It is a real mixed-Q4
artifact rather than an F16 model carrying a Q4 filename.
