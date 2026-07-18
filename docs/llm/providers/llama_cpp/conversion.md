# llama.cpp Conversion Tools

`tools/` is a generated mirror of llama.cpp's top-level `convert_*.py` scripts
and their local `conversion`, `gguf-py`, and requirements dependencies. Do not
edit generated files directly.

The tools can be invoked directly:

```powershell
python scripts/conversion/categories/llm/providers/llama_cpp/tools/convert_hf_to_gguf.py --help
python scripts/conversion/categories/llm/providers/llama_cpp/tools/convert_lora_to_gguf.py --help
```

`source.json` records the exact llama.cpp commit and content digest. Refresh
the submodule, root GGML, Forge bridge, and this tool mirror together with:

```powershell
python scripts/maintenance/updates/llama_cpp.py
```
