# Repository Architecture

GGML-Forge uses responsibility first and category/provider second.

```text
include/, src/
|-- ops/                         operator contracts and backend kernels
|-- nn/                          reusable model composition and loading
`-- categories/
    |-- tts/
    |   `-- providers/
    |-- llm/
    |   `-- providers/
    |-- asr/
    |   `-- providers/
    `-- visual_generation/
        `-- providers/

scripts/
|-- conversion/                  artifact and model conversion
`-- maintenance/                 dependencies, patches, and updates

models/
`-- <category>/<provider>/        runtime configurations and local assets
```

`ops` owns operator semantics and backend execution. `nn` builds reusable
modules on those operations. A category owns its public lifecycle and request
contracts. Concrete architecture knowledge stays inside providers.

Provider boundaries continue in model assets and conversion tools. This keeps
model-specific checkpoint inspection, configuration, frontend resources, and
runtime implementation aligned without teaching shared layers about a
particular model family.
