# Model Export and Quantization Scripts

This directory contains domain/provider exporters and model-independent artifact tooling.

## Directory Structure

- `common/`: Artifact construction, layout contracts, validation, precision
  policy, and quantization. It is independent of model domain and owns no CLI.
- `tts/providers/gpt_sovits/`: GPT-SoVITS checkpoint adaptation and its
  provider-owned `process.py` CLI.

Every provider owns its source inspection and `process.py`. Providers may reuse
`common`, but `common` never selects source files, architectures, components, or
conversion order.
