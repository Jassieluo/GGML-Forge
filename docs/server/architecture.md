# Server architecture

`forge-server` is a protocol adapter over the public category APIs. It is not
an alternative inference runtime and does not include provider-specific model
logic.

The layout follows the useful separation in llama.cpp's server without
embedding llama.cpp's server implementation:

```text
main.cpp                 thin process entry point
server_config.*          command-line configuration
server_context.*         HTTP lifetime, middleware, and route assembly
model_registry.*         loaded category models and per-request sessions
protocols/               OpenAI/Anthropic request normalization
routes/                   HTTP response and streaming adapters
codecs.*                 base64, WAV, and PNG boundaries
```

Requests are normalized into Forge-level messages or media requests. The
registry then calls only headers under `include/categories`. For example, chat
requests use `llm_generate_chat` or `llm_generate_chat_content`; the active LLM
provider owns chat-template application and multimodal encoding. This makes a
new provider available to the server when it implements its category contract,
without adding a provider-specific route.

The executable is built only when `INFERENCE_BUILD_SERVER=ON`. Routes are
compiled and registered only for enabled categories. cpp-httplib and nlohmann
JSON are currently consumed from the llama.cpp vendor snapshot already synced
by the project; no llama.cpp server source is enabled.

Each request creates a category session. Access to a loaded model is currently
serialized per category with a mutex, which is conservative and correct for
the existing providers. A future scheduler can replace this with session lanes
or a bounded job queue without changing protocol routes. Long-running visual
requests are currently synchronous for the same reason.
