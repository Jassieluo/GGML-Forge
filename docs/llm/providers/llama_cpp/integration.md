# llama.cpp Integration

llama.cpp is maintained as a Git submodule at:

```text
src/categories/llm/providers/llama_cpp/llama.cpp/
```

The provider adapter includes the official `llama.h` API. During CMake
integration, the existing project GGML target causes llama.cpp to use Forge's
root `ggml/` rather than building its bundled copy.

The root GGML tree is derived from the same pinned llama.cpp commit. Forge's
structural bridge patch is applied to a temporary copy and verified before the
active tree is replaced. The dependency lock records the llama.cpp commit,
upstream GGML tree, patched tree digest, bridge revision, and copied conversion
tool digest.

The single-library updater performs submodule update, GGML synchronization,
bridge application, conversion-tool mirroring, and lock update as one managed
workflow. See [Using dependency updates](../../../maintenance/usage.md).
