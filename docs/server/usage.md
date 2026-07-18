# Using forge-server

Enable the tool with the project preset, then build it:

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16 -DINFERENCE_BUILD_SERVER=ON
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16 --target forge-server
```

At least one model must be supplied. Multiple categories may be served by the
same process:

```powershell
build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/forge-server.exe `
  --host 127.0.0.1 --port 8080 --api-key local-secret `
  --device CUDA0 --threads 8 --llm-gpu-layers -1 `
  --llm-model models/llm/llama_cpp/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf `
  --llm-mmproj models/llm/llama_cpp/qwen3.5-4b/mmproj-F16.gguf `
  --asr-model models/asr/whisper_cpp/whisper-small-q4_0.bin
```

Authentication is optional. When `--api-key` is present, use either
`Authorization: Bearer local-secret` or `x-api-key: local-secret`. `/health`
remains public. Inspect `/v1/models` for loaded category/provider pairs and
`/forge/v1/capabilities` for registered routes.

`--tts-model` accepts a Forge TTS composition JSON. `--visual-model` accepts a
model supported by the active visual provider, including a stable-diffusion.cpp
safetensors checkpoint. See `examples/server` for compiled C++ clients covering
the OpenAI, Anthropic, ASR, TTS, image, and video routes.
