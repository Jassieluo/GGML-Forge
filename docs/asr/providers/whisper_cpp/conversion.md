# whisper.cpp Conversion Tools

`scripts/conversion/categories/asr/providers/whisper_cpp/tools/` is generated
from the pinned whisper.cpp `models/` conversion scripts. Do not edit the
mirrored files directly.

The main OpenAI Whisper checkpoint converter is invoked as upstream defines:

```powershell
python scripts/conversion/categories/asr/providers/whisper_cpp/tools/convert-pt-to-ggml.py `
  path/to/model.pt path/to/openai-whisper path/to/output-directory
```

Other mirrored converters cover the H5, Core ML, OpenVINO, Parakeet, and Silero
VAD formats present in the pinned upstream revision. Their Python dependencies
remain upstream-defined.

Refresh the submodule, integration lock, and all mirrored tools together with:

```powershell
python scripts/maintenance/updates/whisper_cpp.py
```
