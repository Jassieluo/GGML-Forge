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
    |-- visual_generation/
    |   `-- providers/
    |-- object_detection/
    |   `-- providers/
    |-- depth_estimation/
    |   `-- providers/
    `-- semantic_segmentation/
        `-- providers/

scripts/
|-- conversion/                  artifact and model conversion
`-- maintenance/                 dependencies, patches, and updates

models/
`-- <category>/<provider>/        runtime configurations and local assets

tools/
`-- server/                       provider-neutral long-running services

ui/
`-- eui_neo/                      managed upstream UI framework

examples/
|-- <category>/                   direct public category API examples
`-- server/                       protocol client examples

apps/                             reserved for future user-facing C++ applications
```

`ops` owns operator semantics and backend execution. `nn` builds reusable
modules on those operations. A category owns its public lifecycle and request
contracts. Concrete architecture knowledge stays inside providers.

Provider boundaries continue in model assets and conversion tools. This keeps
model-specific checkpoint inspection, configuration, frontend resources, and
runtime implementation aligned without teaching shared layers about a
particular model family.

`tools` contains reusable operational programs whose lifecycle is larger than
a single example. `forge-server` therefore lives there, while small HTTP calls
that demonstrate its protocols live under `examples/server`. The `apps`
namespace is intentionally kept separate for future C++ UI applications.
The `ui/eui_neo` tree is synchronized as an untouched upstream; Forge-owned UI
application code belongs under `apps`, not inside the framework submodule.
