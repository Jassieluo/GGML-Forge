# llama.cpp Integration

llama.cpp is maintained as a Git submodule at:

```text
src/categories/llm/providers/llama_cpp/llama.cpp/
```

The provider adapter includes the official `llama.h` API and builds the
standalone `libmtmd` target for multimodal projectors. During CMake
integration, the existing project GGML target causes llama.cpp to use Forge's
root `ggml/` rather than building its bundled copy.

An optional mmproj GGUF is loaded against the text model. Encoded image and
audio content is decoded by mtmd, converted into model-native media chunks,
and evaluated in the same session context as surrounding text. The loaded
model reports vision and audio independently; requests never infer media
support from the model filename.

The root GGML tree is derived from the same pinned llama.cpp commit. Forge's
structural bridge patch is applied to a temporary copy and verified before the
active tree is replaced. The dependency lock records the llama.cpp commit,
upstream GGML tree, patched tree digest, bridge revision, and copied conversion
tool digest.

The single-library updater performs submodule update, GGML synchronization,
bridge application, conversion-tool mirroring, and lock update as one managed
workflow. See [Using dependency updates](../../../maintenance/usage.md).
