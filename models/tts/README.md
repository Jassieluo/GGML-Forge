# TTS Models

Each provider owns an isolated runtime asset directory:

- `gpt_sovits/`: GPT-SoVITS configurations, frontend resources, weights, and voices.
- `chat_tts/`: Reserved for ChatTTS integration.
- `qwen3_tts/`: Reserved for Qwen3-TTS integration.

Provider configurations and required text resources are version controlled.
Large weights, reference audio, and generated caches remain local.
