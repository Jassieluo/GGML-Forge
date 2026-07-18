# stable-diffusion.cpp Conversion Tools

Forge mirrors upstream conversion scripts into:

```text
scripts/conversion/categories/visual_generation/providers/stable_diffusion_cpp/tools/
```

The pinned set currently contains converters for FP8 scale folding, Qwen3-VL,
SeFi-Image, and YOLOv8 ADetailer checkpoints. They retain upstream command-line
interfaces and Python dependencies. For example:

```powershell
python scripts/conversion/categories/visual_generation/providers/stable_diffusion_cpp/tools/convert_sefi.py `
  path/to/sefi-diffusers path/to/sefi.safetensors
```

Do not edit mirrored files. Running the stable-diffusion.cpp updater refreshes
the upstream submodule, validates root GGML reuse, copies the current
`convert*.py` files, and updates hashes in the dependency lock.
