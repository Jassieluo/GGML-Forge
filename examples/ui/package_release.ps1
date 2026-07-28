param(
    [string]$Version = "0.1.0-preview.1",
    [string]$Preset = "x64-windows-cuda-sycl-cpu-dl-release-f16",
    [string]$CudaRoot = "D:\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildBin = Join-Path $projectRoot "build-$Preset\bin"
$distRoot = Join-Path $projectRoot "dist"
$releaseName = "GGML-Forge-$Version-windows-x64"
$releaseRoot = Join-Path $distRoot $releaseName
$releaseBin = Join-Path $releaseRoot "bin"
$archive = Join-Path $distRoot "$releaseName.zip"

if (-not (Test-Path (Join-Path $buildBin "ui-demo.exe"))) {
    throw "ui-demo.exe was not found in $buildBin. Build the preset first."
}
if (Test-Path $releaseRoot) { throw "Release directory already exists: $releaseRoot" }
if (Test-Path $archive) { throw "Release archive already exists: $archive" }

New-Item -ItemType Directory -Force $releaseBin | Out-Null
New-Item -ItemType Directory -Force (Join-Path $releaseRoot "models") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $releaseRoot "outputs") | Out-Null

Copy-Item (Join-Path $buildBin "ui-demo.exe") $releaseBin
Copy-Item (Join-Path $buildBin "*.dll") $releaseBin
Copy-Item (Join-Path $buildBin "assets") $releaseBin -Recurse

$cudaBin = Join-Path $CudaRoot "bin\x64"
foreach ($name in @("cublas64_13.dll", "cublasLt64_13.dll")) {
    $source = Join-Path $cudaBin $name
    if (-not (Test-Path $source)) { throw "Required CUDA redistributable not found: $source" }
    Copy-Item $source $releaseBin
}

Copy-Item (Join-Path $projectRoot "LICENSE") $releaseRoot
Copy-Item (Join-Path $PSScriptRoot "release_assets\README.txt") $releaseRoot
Copy-Item (Join-Path $PSScriptRoot "release_assets\THIRD_PARTY_NOTICES.txt") $releaseRoot

function Copy-ReleaseFile([string]$RelativePath) {
    $source = Join-Path $projectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) { throw "Release input not found: $RelativePath" }
    $target = Join-Path $releaseRoot $RelativePath
    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
}

$modelFiles = @(
    "models\llm\llama_cpp\qwen3.5-2b\Qwen3.5-2B-Q4_K_M.gguf",
    "models\llm\llama_cpp\qwen3.5-2b\mmproj-F16.gguf",
    "models\asr\whisper_cpp\whisper-small-q4_0.bin",
    "models\visual_generation\stable_diffusion_cpp\sdxs-512\sdxs-512-q4_k.gguf",
    "models\tts\gpt_sovits\configs\v2-q4.json",
    "models\tts\gpt_sovits\weights\bert\bert_q4_0.gguf",
    "models\tts\gpt_sovits\weights\cnhubert\cnhubert_q4_0.gguf",
    "models\tts\gpt_sovits\weights\t2s\t2s_v2_q4_0.gguf",
    "models\tts\gpt_sovits\weights\vits\vits_v2_q4_0.gguf",
    "models\visual_perception\depth_estimation\stereonet-q4_0.gguf",
    "models\visual_perception\image_classification\yolov8n-cls-q4_0.gguf",
    "models\visual_perception\instance_perception\yolov8n-q4_0.gguf",
    "models\visual_perception\instance_perception\yolov8n-seg-q4_0.gguf",
    "models\visual_perception\instance_perception\yolov8n-pose-q4_0.gguf",
    "models\visual_perception\instance_perception\yolov8n-obb-q4_0.gguf",
    "models\visual_perception\semantic_segmentation\lraspp-mobilenet-v3-large-q4_0.gguf"
)
foreach ($file in $modelFiles) { Copy-ReleaseFile $file }

$dictionarySource = Join-Path $projectRoot "models\tts\gpt_sovits\resources\dictionaries"
$dictionaryTarget = Join-Path $releaseRoot "models\tts\gpt_sovits\resources\dictionaries"
New-Item -ItemType Directory -Force (Split-Path -Parent $dictionaryTarget) | Out-Null
Copy-Item $dictionarySource $dictionaryTarget -Recurse

Copy-ReleaseFile "models\tts\gpt_sovits\voices\doubao\config.json"
$voiceSource = Join-Path $projectRoot "models\tts\gpt_sovits\voices\doubao\audios"
$voiceTarget = Join-Path $releaseRoot "models\tts\gpt_sovits\voices\doubao\audios"
New-Item -ItemType Directory -Force $voiceTarget | Out-Null
Get-ChildItem $voiceSource -File | Where-Object {
    $_.Extension -eq ".wav" -or $_.Name.EndsWith(".sv.bin")
} | Copy-Item -Destination $voiceTarget

tar.exe -a -cf $archive -C $distRoot $releaseName
if ($LASTEXITCODE -ne 0) { throw "tar.exe failed to create $archive" }

$hash = Get-FileHash $archive -Algorithm SHA256
"$($hash.Hash) *$([System.IO.Path]::GetFileName($archive))" |
    Out-File "$archive.sha256.txt" -Encoding ascii

$bytes = (Get-ChildItem $releaseRoot -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Release directory: $releaseRoot"
Write-Host "Archive: $archive"
Write-Host ("Uncompressed size: {0:N2} GiB" -f ($bytes / 1GB))
Write-Host "SHA256: $($hash.Hash)"
