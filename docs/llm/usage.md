# Using LLM

The public LLM API is declared in `include/categories/llm/llm.h`. The current
provider loads GGUF models through llama.cpp.

```cpp
#include "categories/llm/llm.h"

#include <iostream>

static bool print_piece(const char* utf8, size_t length, void*) {
    std::cout.write(utf8, static_cast<std::streamsize>(length));
    return true; // Return false to stop generation early.
}

llm_runtime_params runtime_params = llm_runtime_default_params();
runtime_params.n_ctx = 4096;
runtime_params.n_batch = 512;
runtime_params.n_threads = 8;
runtime_params.n_gpu_layers = 0;
runtime_params.device = "cpu"; // or an exact device such as "CUDA0" / "SYCL0"

llm_runtime_ptr runtime = llm_runtime_create(runtime_params);
llm_model_ptr model = llm_load_model(runtime, "path/to/model.gguf");
llm_session_ptr session = model ? llm_create_session(model) : nullptr;

llm_generation_params generation = llm_generation_default_params();
generation.max_tokens = 128;
generation.temperature = 0.8f;
generation.top_k = 40;
generation.top_p = 0.95f;

bool ok = session && llm_generate(
    session, "Write a short greeting:", generation, print_piece, nullptr);

llm_free_session(session);
llm_free_model(model);
llm_runtime_free(runtime);
```

For a multimodal model, load its matching projector and submit ordered content
parts. Media data contains encoded file bytes rather than provider-specific
pixel tensors:

```cpp
llm_model_params model_params = llm_model_default_params();
model_params.model = "path/to/model.gguf";
model_params.mmproj = "path/to/mmproj.gguf";
llm_model_ptr model = llm_load_model_with_params(runtime, &model_params);

llm_content_part parts[] = {
    {LLM_CONTENT_TEXT, "Describe this image: ", 21, "text/plain"},
    {LLM_CONTENT_IMAGE, jpeg_bytes, jpeg_size, "image/jpeg"},
};
bool ok = llm_generate_content(
    session, parts, 2, generation, print_piece, nullptr);
```

Query `llm_model_get_capabilities` before sending image or audio content. Parts
are borrowed for the synchronous call and can be interleaved; the provider
preserves their order. The llama.cpp provider accepts common image formats and
WAV/MP3/FLAC through mtmd.

Callback text is a byte span and is not guaranteed to be null-terminated; use
the supplied length. Returning false requests a successful early stop.

Each call to `llm_generate` resets the session's llama.cpp memory before
processing its prompt. `llm_session_reset` can also clear it explicitly. The
prompt plus requested output must fit in `n_ctx`. A temperature less than or
equal to zero selects greedy decoding; positive temperatures use top-k,
top-p, temperature, and seeded distribution sampling.

Sessions retain their loaded model implementation, but applications should
still release handles in session, model, runtime order. Use one session per
concurrent generation request.

Complete command-line examples are available for both input forms:

- [`examples/llm/generate.cpp`](../../examples/llm/generate.cpp): plain text
  generation with `llm_generate`.
- [`examples/llm/multimodal.cpp`](../../examples/llm/multimodal.cpp): ordered
  text and image parts with `llm_generate_content`.

```text
llm-generate <model.gguf> [prompt]
llm-multimodal <model.gguf> <mmproj.gguf> <image> [question]
```

These examples submit raw prompt text. Supply any chat-template control tokens
required by the selected model as part of `prompt` or `question`.
