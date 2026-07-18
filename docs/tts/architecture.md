# TTS Category Architecture

The TTS category defines stable request, capability, runtime, loaded-model, and
session contracts. It does not prescribe a fixed sequence of text encoder,
semantic generator, acoustic model, or vocoder stages.

## Ownership

```text
TTSProviderRegistry
  -> ITTSProvider       stateless composition validator and loaded-model factory
     -> ITTSModel       shared artifacts, execution lanes, and capabilities
        -> ITTSSession  voice, prompt cache, RNG, and logical request state
```

A provider may implement one or more internal pipelines. Pipeline stages are an
implementation detail unless two independent providers demonstrate a stable,
identical component contract.

Runtime policy remains separate from portable model composition. A loaded model
snapshots device, thread, concurrency, and component-residency settings. A
session keeps the loaded model alive after the public model handle is released.

## Model placement

Concrete model implementations are provider-owned:

```text
providers/gpt_sovits/models/{bert,hubert,speaker_encoder,t2s,vits}
providers/gpt_sovits/frontend
providers/chat_tts/models/...
providers/qwen3_tts/models/...
```

Directories do not claim that every TTS architecture has SSL, text, language
model, acoustic, or synthesis stages. Reusable tensor operations and neural
network layers belong to `ops` and `nn`. A provider implementation is promoted
to a shared TTS component only after its artifact and execution contracts are
proven reusable by multiple providers.

Text normalization, phonemization, symbol inventories, and their vendored
dependencies follow the same ownership rule. For example, cppjieba and the
GPT-SoVITS phone-symbol tables live under `providers/gpt_sovits/frontend`; they
are not part of the generic TTS frontend or public category API.

## Compatibility

The public `tts_*` API is the category entry point. The `gpt_sovits_*` API is a
provider compatibility layer and may call provider internals, but new category
code must not depend on that compatibility surface.
