# GPT-SoVITS Runtime Assets

- `configs/`: Version and precision compositions consumed by the TTS runtime.
- `resources/dictionaries/`: Version-controlled frontend dictionaries.
- `weights/`: Local GGUF artifacts, excluded from Git.
- `voices/`: Local reference audio and generated feature caches, excluded from Git.

Paths inside a config are resolved relative to that config file.
