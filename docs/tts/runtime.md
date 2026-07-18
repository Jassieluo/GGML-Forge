# Runtime Architecture

The portable model composition describes which artifacts form a TTS model. It
does not contain device names, thread counts, concurrency, or residency rules.
Those settings belong to the machine-specific runtime.

## Ownership

- `tts_runtime` owns mutable policy used to create future model instances.
- `tts_model` snapshots that policy and owns the provider's loaded model instance.
- `tts_session` owns one logical synthesis session and keeps its reference,
  prompt cache, RNG, generation state, and output buffer independent.
- Loaded models own model artifacts and map provider component IDs to
  their concrete models.

Changing a runtime policy never migrates an already loaded model. Load another
model to apply the new policy. This avoids hidden device transfers and races.

`tts_runtime_params::max_concurrency` limits the number of execution lanes for
one loaded model. Busy sessions lease different lanes and execute concurrently;
additional requests wait for a lane instead of racing shared execution state.

## Component Policy

`tts_runtime_set_component_policy` accepts a provider component ID, optional
exact backend device name, and residency:

- `TTS_COMPONENT_RESIDENT`: load with the model and retain it.
- `TTS_COMPONENT_ON_DEMAND`: load for a request and unload when that request
  scope exits.

A null or empty component device inherits the runtime default. An unavailable
non-empty device is rejected; the runtime never silently falls back to CPU.

The GPT-SoVITS provider exposes `hubert`, `bert`, `t2s`, and `vits`. For
example, this keeps the main provider on the default device while unloading
HuBERT between requests:

```cpp
tts_runtime_set_component_policy(
    runtime, "hubert", nullptr, TTS_COMPONENT_ON_DEMAND);
```

Different components may target exact CPU, CUDA, or SYCL device names. The
policy is generic; each provider defines its own stable component IDs.

## Concurrency Boundary

The GPT-SoVITS provider uses a lazily expanded execution-lane pool. Every lane
owns an independent provider engine, backend instances, KV cache, input arenas,
graph allocators, upload queues, and output storage. Prompt cache and RNG state
remain session-owned and are attached only to the leased lane for a request.

Resident HuBERT and BERT artifacts are shared by compatible lanes because their
normal inference paths are immutable after loading. Debug mode and on-demand
residency keep separate instances because their lifetime or diagnostic tensor
state is lane-local. T2S and VITS are still replicated per lane because they
currently own mutable KV caches, input arenas, upload queues, and graph state.

The next memory optimization is to extract those T2S/VITS fields into explicit
per-lane execution state. Their immutable GGUF weights can then use the same
artifact registry without changing the lane scheduling contract.
