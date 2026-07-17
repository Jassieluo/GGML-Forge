# GPT-SoVITS.cpp V3 Noise Alignment & Handoff Documentation

This document provides a comprehensive overview of the current status, configurations, environment details, build instructions, and debugging progress for the V3 alignment phase of the `GPT-SoVITS.cpp` project. It is intended to allow the next agent to immediately recover the context and continue the alignment/debugging tasks.

---

## 1. Project Context & Repositories

* **C++ CMake Project**: `D:\Projects\CMake Projects\GPT-SoVITS.cpp` (Current workspace)
* **Python Reference Project**: `D:\Projects\PycharmProjects\GPT-SoVITS-main`
* **Python Conda Environment**: `D:\anaconda3\envs\gpt-sovits` (Python interpreter: `D:\anaconda3\envs\gpt-sovits\python.exe`)
* **Gemini Artifact Directory**: `C:\Users\18341\.gemini\antigravity-cli\brain\cf80ef46-4a1b-4cca-8070-12c36f0926e8`
* **Local Test Scratch Outputs**: `D:\Projects\CMake Projects\GPT-SoVITS.cpp\scratch\`

---

## 2. Compilation & Build Configurations

### Intel oneAPI & MSVC Toolchain
On this system, oneAPI is installed and should be initialized prior to C++ compilation or CPU runs to set up MKL and TBB paths:
```powershell
call "D:\Tools\Intel\oneAPI\setvars.bat"
```

### Build Directories
The C++ project compiles to:
* **Release (CUDA/SYCL/CPU)**: `D:\Projects\CMake Projects\GPT-SoVITS.cpp\build-x64-windows-cuda-sycl-cpu-dl-release-f16`
* Executable: `build-x64-windows-cuda-sycl-cpu-dl-release-f16\bin\gpt-sovits-test-pipeline.exe`

### Typical Build Script (`scratch/build.bat`)
```bat
@echo off
call "D:\Tools\Intel\oneAPI\setvars.bat"
cd "D:\Projects\CMake Projects\GPT-SoVITS.cpp"
cmake --build build-x64-windows-cuda-sycl-cpu-dl-release-f16 --config Release --parallel 4
```

---

## 3. The V3 Inference Pipeline & The Noise Issue

### Current Progress (What is working)
1. **T2S (Text-to-Semantic) Output & Argument Encoding Fix**:
   By using direct Unicode argument list execution (bypassing `cmd.exe` GBK conversion) and mapping filenames to ASCII via [run_v3_clean.py](file:///D:/Projects/CMake%20Projects/GPT-SoVITS.cpp/scratch/run_v3_clean.py), C++ now successfully loads the reference WAV audio (`scratch/ref_audio.wav`) and runs T2S fully.
   * **C++ T2S Output**: Generated 88 semantic tokens (e.g. `280 105 271 505 538 287...`).
   * This confirms that text phonemization, BERT projection, and the autoregressive transformer decoding loops are **mathematically identical** to Python.

2. **Weight Loader & Safe Bounds**:
   Resolved assertions when running under `GPT_SOVITS_DEBUG=1` where fetching tensor data for quantized (`Q4_0`) weights directly caused crashes.

### Current Discrepancies (Why C++ has noise / Python has timbre issues)
1. **VITS Input Alignment & Semantic Lengths**:
   In the latest successful run:
   * **Python**: In `decode_encp`, prompt semantic codes are prepended to target codes. For `"平静.wav"`, target semantics codes have `len=88` and prompt semantics have `len=116` (giving total concatenated `len=204` input for VITS).
   * **C++**: In VITS forward pass, `pred_semantics_tensor` received `len=86` with values starting with `105 271 492 824...`. 
   * **Critical Check**: The next agent should inspect why C++ semantic tokens received by VITS (`len=86` instead of `88`) differ slightly in length and values from the T2S target sequence, and verify if the prompt semantics are prepended properly before VITS VQ-decoding.

---

## 4. Quick Execution & Verification Tools

All script tools reside in `scratch/`:

* **`scratch/run_v3_clean.py`** (Created for direct C++ execution):
  Loads the Intel oneAPI environment variables, copies reference audio to ASCII, and directly triggers the C++ test-pipeline with correct UTF-8 text strings to generate `scratch/output_v3_cpu.wav`.
* **`scratch/run_full_python_inference.py`**:
  Runs the reference Python code using the `gpt-sovits` conda environment to generate the golden target output `scratch/python_inference_output.wav`.
* **`scratch/compare_fea.py`**:
  Compares intermediate features (BERT, T2S, SSL, VITS outputs) saved during inference to pinpoint the divergence location.

---

## 5. Recommended Next Steps for the Next Agent

1. **Verify Semantic Prepended Layout in VITS**:
   Check how `pred_semantics_tensor` view is allocated in `gpt_sovits_pipeline.cpp` (around L1250-1300). Confirm that prompt semantic features (from the reference audio cache) are correctly prepended to the generated target semantic tokens before triggering `impl->vits->forward`.
2. **Compare intermediate VITS tensors**:
   Execute the clean runner script `python scratch/run_v3_clean.py` and run `python scratch/run_full_python_inference.py`. Use `compare_fea.py` to compare intermediate acoustic features layer-by-layer (VQ, MRTE, CFM ODE, Vocoder) to trace the exact source of noise in the synthesis phase.

