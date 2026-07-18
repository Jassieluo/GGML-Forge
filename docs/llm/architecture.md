# LLM Category Architecture

The LLM category exposes runtime, loaded-model, session, and generation
lifetimes independently of a concrete inference library.

```text
llm_runtime -> LLM provider -> llm_model -> llm_session -> streamed text
                                  |
                                  `-> optional vision/audio projector
```

Runtime configuration owns machine policy such as context size, threads, GPU
layers, and device selection. A loaded model owns provider model state. A
session owns mutable generation context and can be reset without reloading the
model.

Generation input is either a legacy text prompt or an ordered sequence of
content parts. Text, encoded image, and encoded audio parts remain generic at
the category boundary; providers report vision and audio capabilities from the
loaded artifacts and translate supported media into native embeddings.

Providers adapt external or project-owned engines behind the category API.
The initial `llama_cpp` provider uses official llama.cpp APIs while linking to
the project GGML target. Upstream implementation details do not leak into the
public category contract.
