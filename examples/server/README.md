# forge-server C++ examples

These executables call the protocol-facing API exposed by `tools/server`. They
use cpp-httplib and nlohmann JSON from the dependency snapshot already used by
the server; no PowerShell or curl process is involved.

Build the server and clients together:

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16 -DINFERENCE_BUILD_SERVER=ON
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16 --target `
  forge-server server-openai-chat server-anthropic-message server-transcribe `
  server-speech server-image server-video
```

Start the server separately, then run a client. All clients accept
`--base-url`; protected servers additionally accept `--api-key`.

```powershell
build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-openai-chat.exe `
  --prompt "Describe this image." --image input.png

build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-anthropic-message.exe `
  --prompt "Introduce yourself briefly."

build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-transcribe.exe `
  --audio input.wav --language zh

build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-speech.exe `
  --text "This is a server test." --language en --output speech.wav

build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-image.exe `
  --prompt "a small red fox" --size 256x256 --steps 8 --output fox.png

build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/server-video.exe `
  --prompt "ocean waves at sunset" --frames 8 --output video-frames
```

`server-speech` also accepts `--reference-audio`, `--reference-text`, and
`--reference-language`. `server-openai-chat` turns `--image` into a base64 data
URL, while the media clients write binary WAV/PNG responses directly.
