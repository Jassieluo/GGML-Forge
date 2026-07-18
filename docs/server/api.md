# Server API

## Compatibility routes

| Protocol | Method and path | Current behavior |
| --- | --- | --- |
| OpenAI | `GET /v1/models` | Loaded Forge models with category/provider extensions |
| OpenAI | `POST /v1/chat/completions` | Text and base64 data-URI image/audio content; SSE supported |
| OpenAI | `POST /v1/completions` | Text completion; SSE supported |
| OpenAI | `POST /v1/responses` | Text or message-array input; non-streaming |
| OpenAI | `POST /v1/audio/transcriptions` | Multipart PCM16/float32 WAV; JSON or text response |
| OpenAI | `POST /v1/audio/speech` | PCM16 WAV response; Forge reference-voice extensions |
| OpenAI | `POST /v1/images/generations` | One PNG as `b64_json` or data URI |
| Anthropic | `POST /v1/messages` | Text and base64 image content; SSE supported |
| Forge | `GET /forge/v1/capabilities` | Active categories and endpoint discovery |
| Forge | `POST /forge/v1/videos/generations` | Synchronous PNG frame array |

These are compatibility surfaces, not claims of complete vendor API parity.
Unknown vendor fields are generally ignored. Structured tool/function calls,
logprobs, embeddings, remote image URLs, prompt caching, batch APIs, and
streaming Responses are not implemented. Token usage fields are currently zero
because the public category contract does not yet expose tokenizer accounting.

The video API intentionally uses the Forge namespace: no stable cross-vendor
video schema exists, and the current response is a synchronous frame bundle,
not an encoded WebM/MP4 or asynchronous job. A model without video capability
returns a generation error.

OpenAI-style errors use an `error` object. Authentication accepts Bearer or
`x-api-key`; the Anthropic version header is accepted but is not currently used
for schema negotiation.
