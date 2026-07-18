# Using TTS

The public TTS API is declared in `include/categories/tts/tts.h`. Applications
work with three opaque handles: runtime, loaded model, and session.

## Required inputs

- A portable model-composition JSON, for example one of the configurations
  under `models/tts/gpt_sovits/configs/`.
- The GGUF artifacts and frontend resources referenced by that composition.
- For a voice-cloning provider, mono reference audio and its transcript.

Paths inside a composition are resolved relative to the composition file.

## Basic lifecycle

```cpp
#include "categories/tts/tts.h"

tts_runtime_params runtime_params = tts_runtime_default_params();
runtime_params.device = "auto";
runtime_params.n_threads = 4;
runtime_params.max_concurrency = 1;

tts_runtime_ptr runtime = tts_runtime_create(runtime_params);
tts_model_ptr model = tts_load_model(
    runtime, "models/tts/gpt_sovits/configs/v3-q4.json");
tts_session_ptr session = model ? tts_create_session(model) : nullptr;

bool reference_ok = session && tts_session_set_reference(
    session,
    reference_samples,
    reference_sample_count,
    reference_sample_rate,
    reference_transcript,
    "zh");

int32_t sample_count = 0;
const float* audio = reference_ok
    ? tts_synthesize(session, "要合成的文本", "zh", 1.0f, &sample_count)
    : nullptr;

// Copy or write `audio` before the next synthesis on this session.

tts_free_session(session);
tts_free_model(model);
tts_runtime_free(runtime);
```

The returned audio is mono float32 data owned by the session. It remains valid
until the next synthesis on that session. Obtain its sample rate with
`tts_session_get_output_sample_rate`.

## Streaming

Use `tts_synthesize_streaming` only when
`tts_model_get_capabilities(model).streaming` is true. The chunk callback must
consume or copy each chunk during the callback; it does not own the supplied
buffer.

## Runtime and session policy

Call `tts_runtime_set_component_policy` before loading a model. Component IDs
are provider-defined; GPT-SoVITS currently exposes `hubert`, `bert`, `t2s`, and
`vits`. A null component device inherits the runtime device.

Sessions own reference voice, request options, prompt cache, and mutable
generation state. Do not execute two requests concurrently on the same
session. Create separate sessions for concurrent requests; they may share one
loaded model up to the runtime's `max_concurrency` limit.

Provider-specific options are set on a session. The stable generic float
option is `speed`; GPT-SoVITS CFM models additionally accept `cfm_steps`.

The complete command-line integration is implemented in
[`examples/tts/gpt-sovits/gpt-sovits.cpp`](../../examples/tts/gpt-sovits/gpt-sovits.cpp).
