# Maintenance Architecture

Project maintenance tooling lives under `scripts/maintenance/` and separates
dependency definitions, Forge patches, and update orchestration.

```text
scripts/maintenance/
|-- dependencies/<library>/       config, lock, and synchronization logic
|-- patches/<library>/             Forge-owned patch implementation and assets
`-- updates/
    |-- <library>.py               update and patch one library
    `-- all.py                     update every managed library
```

Every entry point resolves the repository from its own file location and can
therefore run from any working directory. Library-specific update scripts own
their complete consistency transaction; `all.py` only discovers and invokes
those scripts. Dependency configurations declare `depends_on`; the total
updater resolves that graph so producers such as llama.cpp/root GGML are
updated before consumers such as whisper.cpp and stable-diffusion.cpp.
