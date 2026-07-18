# ASR Category Architecture

The ASR category defines provider-independent runtime, loaded-model, session,
request, capability, and result-event contracts.

```text
asr_runtime -> ASR provider -> asr_model -> asr_session -> ASR events
```

The runtime owns machine policy such as device and CPU thread count. A loaded
model owns immutable model resources. A session owns provider decoding state
and is the unit of request isolation.

Input normalization is category infrastructure: the public API accepts mono
float32 audio with an explicit sample rate and resamples it to the provider's
required rate. Model-specific mel extraction, tokenization, encoder/decoder
stages, context windows, and decoding rules remain provider internals.

Results use a generic event stream:

- `ASR_EVENT_LANGUAGE` reports detected language.
- `ASR_EVENT_SEGMENT` reports timestamped transcript segments.
- `ASR_EVENT_TOKEN` optionally reports token text, timestamps, and confidence.

Capabilities explicitly report translation, language detection, segment and
token timestamps, and streaming. A provider must not claim streaming merely
because it can repeatedly transcribe overlapping windows.
