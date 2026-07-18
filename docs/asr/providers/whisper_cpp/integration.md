# whisper.cpp Integration

The official whisper.cpp repository is a Git submodule at:

```text
src/categories/asr/providers/whisper_cpp/whisper.cpp/
```

Forge links the upstream `whisper` target but does not build whisper.cpp's
bundled GGML. The root project creates `ggml` first; whisper.cpp's supported
parent-target path detects that target and skips `add_subdirectory(ggml)`.

Root GGML remains derived from the pinned llama.cpp revision. The whisper.cpp
lock records both upstream revisions, the active root GGML digest, whisper's
bundled GGML tree for diagnostics, the patch revision, and the mirrored
conversion-tool digest.

Updating whisper.cpp validates that its parent-GGML CMake contract still
exists. GGML API compatibility is then verified by the normal ASR target
build; whisper.cpp can never overwrite or downgrade root GGML.
