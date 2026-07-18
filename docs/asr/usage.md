# Using ASR

The public C API is declared in `include/categories/asr/asr.h`. The current
provider loads whisper.cpp GGML model files directly.

```cpp
#include "categories/asr/asr.h"

#include <iostream>

static bool on_event(const asr_event* event, void*) {
    if (event->type == ASR_EVENT_SEGMENT) {
        std::cout.write(event->text, event->text_length);
    }
    return true;
}

asr_runtime_params runtime_params = asr_runtime_default_params();
runtime_params.device = "auto";
runtime_params.n_threads = 8;

asr_runtime_ptr runtime = asr_runtime_create(runtime_params);
asr_model_ptr model = asr_load_model(runtime, "path/to/ggml-model.bin");
asr_session_ptr session = model ? asr_create_session(model) : nullptr;

asr_request_params request = asr_request_default_params();
request.language = "auto";
request.task = ASR_TASK_TRANSCRIBE;
request.token_timestamps = false;

bool ok = session && asr_transcribe(
    session,
    mono_samples,
    sample_count,
    input_sample_rate,
    request,
    on_event,
    nullptr);

asr_free_session(session);
asr_free_model(model);
asr_runtime_free(runtime);
```

Input must be finite mono float32 audio. The category accepts any positive
sample rate and performs linear resampling to the provider rate. Callback text
is a byte span valid only during the callback; use `text_length` and copy it if
it must outlive the callback.

Use `ASR_TASK_TRANSLATE` only when the model capabilities report translation.
Null, empty, and `auto` language values enable automatic language selection.
Set `token_timestamps` only when token events are required.

One session must not process concurrent requests. The whisper.cpp provider
shares model weights between sessions and currently serializes execution on
that shared model context. Its streaming capability is therefore reported as
false.
