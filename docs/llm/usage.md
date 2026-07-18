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
