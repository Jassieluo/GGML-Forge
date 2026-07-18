# GPT-SoVITS Runtime Assets

- `configs/`: Version and precision compositions consumed by the TTS runtime.
- `resources/dictionaries/`: Version-controlled frontend dictionaries.
- `weights/`: Local GGUF artifacts, excluded from Git.
- `voices/`: Local reference audio and generated feature caches, excluded from Git.

Paths inside a config are resolved relative to that config file.

The maintained GPT-SoVITS runtime set uses version-qualified Q4 filenames.
Shared BERT and CNHuBERT weights keep provider-level names; T2S, VITS, and the
ERes2NetV2 speaker encoder include their model family/version in the filename.

V2Pro and V2ProPlus compositions additionally declare `speaker_encoder`, an
ERes2NetV2 GGUF artifact. When a session reference is attached, the provider
resamples it to 16 kHz, computes the Kaldi-compatible 80-bin filterbank, and
caches the resulting 20480-float speaker vector with the other prompt features.
Other GPT-SoVITS versions do not load this component.

Generated caches use two compatibility namespaces under `voices/`:

- `features/<exact-version>/<profile-id>/<backend>/` stores complete prompt
  caches. Exact VITS profiles and CPU/CUDA/SYCL payloads never overwrite one
  another. Compatible legacy flat feature caches are migrated on first use.
- `.cache/speaker_embeddings/eres2net_v2/kaldi_fbank80_v1/<weight-id>/`
  stores content-addressed 20480-float vectors. These are keyed by the exact
  speaker-encoder weights, preprocessing contract, sample rate, and decoded
  reference samples, so V2Pro and V2ProPlus can safely share the expensive
  audio-only encoding while encoder or preprocessing updates invalidate it.
