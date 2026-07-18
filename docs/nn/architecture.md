# NN Architecture

The `nn` layer provides reusable neural-network composition without owning a
model category or provider.

Its public structure is divided by responsibility:

- `core/` owns modules, parameters, state dictionaries, execution context,
  layouts, and executors.
- `functional/` exposes stateless neural-network operations.
- `layers/` composes reusable stateful layers such as attention, convolution,
  embeddings, feed-forward blocks, and normalization.
- `io/` loads model artifacts and GGUF-backed tensor sources.
- `runtime/` owns reusable mutable execution structures such as KV caches.
- `schema/` defines loader-visible model contracts.

Provider models compose these facilities but retain architecture-specific
tensor names, model versions, and pipeline decisions. Backend-specific kernels
remain below `ops`, not in model layers.
