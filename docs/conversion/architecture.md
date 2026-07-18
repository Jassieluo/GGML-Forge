# Conversion Architecture

All project conversion tooling lives under `scripts/conversion/`.

```text
scripts/conversion/
|-- common/                       artifact, schema, validation, and quantization
`-- categories/
    `-- <category>/providers/<provider>/
```

`common/` is model-independent. It owns artifact construction, layout
contracts, validation, precision policy, and quantization, but it does not
inspect provider checkpoints or select model components.

Each provider owns source inspection, tensor adaptation, component ordering,
and its command-line entry point. Project-owned converters may reuse
`common/`. Mirrored upstream converters remain intact and are refreshed by the
corresponding dependency updater.
