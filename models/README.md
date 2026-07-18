# Runtime Models

Runtime assets are grouped first by model category and then by provider:

```text
models/
└── tts/
    ├── gpt_sovits/
    ├── chat_tts/
    └── qwen3_tts/
```

Future categories should be added alongside `tts` rather than mixed into a
shared weights directory.
