# Using Dependency Updates

Update one managed library from any working directory:

```powershell
python scripts/maintenance/updates/llama_cpp.py
python scripts/maintenance/updates/whisper_cpp.py
python scripts/maintenance/updates/stable_diffusion_cpp.py
```

Without `--ref`, the updater uses `default_ref` from the library's dependency
configuration. Pin a particular commit or tag with:

```powershell
python scripts/maintenance/updates/llama_cpp.py --ref <commit-or-tag>
```

For llama.cpp, this one command updates the submodule, synchronizes root GGML,
applies and verifies Forge patches, mirrors conversion tools, and updates lock
metadata. Outputs are staged and validated before installation.

The whisper.cpp updater never replaces root GGML. It validates the upstream
parent-GGML contract, records the active llama.cpp-derived GGML digest, and
mirrors whisper.cpp conversion tools.

The stable-diffusion.cpp updater follows the same root-GGML rule, records its
nested dependency gitlinks and the shared `GGML_MAX_NAME` ABI value, and
mirrors upstream visual-model converters.

Verify reproducibility without fetching or changing files:

```powershell
python scripts/maintenance/updates/llama_cpp.py --check
python scripts/maintenance/updates/all.py --check
```

Update every managed library with:

```powershell
python scripts/maintenance/updates/all.py
```

`--force` allows synchronized generated outputs with local changes to be
replaced. It should only be used when those changes are intentionally
discardable. Never edit copied conversion tools or patched root GGML as the
long-term source of a change; update the corresponding dependency rule or
Forge patch instead.
