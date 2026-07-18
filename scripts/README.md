# Model Export and Quantization Scripts

This directory contains domain/provider exporters and model-independent artifact tooling.

## Directory Structure

- `common/`: Artifact construction, layout contracts, validation, precision
  policy, and quantization. It is independent of model domain and owns no CLI.
- `categories/tts/providers/gpt_sovits/`: GPT-SoVITS checkpoint adaptation and its
  provider-owned `process.py` CLI.

Every provider owns its source inspection and `process.py`. Providers may reuse
`common`, but `common` never selects source files, architectures, components, or
conversion order.

## Provider Contract

Providers must implement `common.ModelProvider` and return a
`common.ModelDefinition`. They may inspect, merge, split, fold, or otherwise
adapt source checkpoints, but they must not import `GGUFWriter`, select storage
dtypes, write layout metadata, or bind names used only by the C++ loader.

```python
class Provider(ModelProvider):
    def load_source(self):
        return load_checkpoint(self.path)

    def build(self, source):
        checkpoint = Checkpoint(source)
        model = ModelDefinition("my_architecture")
        model.uint32("my_model.layer_count", detect_layers(source))
        model.parameter(Parameter(
            "encoder.layers.0.projection.weight",
            checkpoint.take("old.layer.0.proj.weight").cpu().numpy(),
        ))
        checkpoint.finish()
        return model

export_model(provider.definition(), output, target_type="Q4_0")
```

Canonical parameter names must exactly match the C++ `nn::Module` tree. The
common exporter owns dtype fallback, quantization, physical layout metadata,
validation, temporary files, and atomic commit. Bias, normalization, scalar,
and constant parameters remain F32 by default; ordinary weights follow the
selected `Q4 -> Q8 -> F16` or `Q8 -> F16` policy.
