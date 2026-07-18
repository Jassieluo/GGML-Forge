# Using Conversion Tools

Conversion entry points are grouped by category and provider.

## GPT-SoVITS

Convert one explicit source artifact at a time:

```powershell
python scripts/conversion/categories/tts/providers/gpt_sovits/process.py `
  --model-type t2s --version v3 `
  --src path/to/source.ckpt `
  --output path/to/t2s_v3_q4_0.gguf `
  --quantize Q4_0
```

Supported targets are `F16`, `Q4_0`, `Q4_K`, `Q4_K_M`, and `Q8_0`. Provider
details and multi-file VITS source rules are documented in
[GPT-SoVITS conversion](../tts/providers/gpt_sovits/conversion.md).

## llama.cpp

The mirrored tools are invoked directly and retain upstream arguments:

```powershell
python scripts/conversion/categories/llm/providers/llama_cpp/tools/convert_hf_to_gguf.py `
  path/to/hugging-face-model --outfile path/to/model.gguf

python scripts/conversion/categories/llm/providers/llama_cpp/tools/convert_lora_to_gguf.py `
  path/to/lora --outfile path/to/lora.gguf
```

Install their pinned Python dependencies from the mirrored requirements file
in an isolated environment:

```powershell
python -m pip install -r scripts/conversion/categories/llm/providers/llama_cpp/tools/requirements.txt
```

Do not edit mirrored files; refresh them through the llama.cpp dependency
update workflow. See
[llama.cpp conversion tools](../llm/providers/llama_cpp/conversion.md).

## whisper.cpp

The mirrored whisper.cpp tools convert OpenAI Whisper checkpoints and other
supported upstream formats:

```powershell
python scripts/conversion/categories/asr/providers/whisper_cpp/tools/convert-pt-to-ggml.py `
  path/to/model.pt path/to/openai-whisper path/to/output-directory
```

These scripts retain their upstream arguments and dependencies. See
[whisper.cpp conversion tools](../asr/providers/whisper_cpp/conversion.md).

## stable-diffusion.cpp

Visual-model converters are mirrored from the pinned upstream scripts:

```powershell
python scripts/conversion/categories/visual_generation/providers/stable_diffusion_cpp/tools/convert_qwen3_vl.py `
  path/to/qwen3-vl path/to/output.safetensors
```

See [stable-diffusion.cpp conversion tools](../visual_generation/providers/stable_diffusion_cpp/conversion.md)
for the synchronized tool set and update policy.
