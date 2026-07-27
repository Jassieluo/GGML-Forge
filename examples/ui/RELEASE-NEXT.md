# UI Release follow-up

The initial release may keep the current GGML-Forge Studio feature set. The
next UI iteration should focus on release portability and the new visual
perception categories.

## Next steps

1. Replace hard-coded model constants with a small model catalog that scans
   `models/` beside the release package and lets the user select a model.
2. Only show Q4 release artifacts. Exclude F16 checkpoints, original `.pt`
   files, conversion inputs, caches, and other development-only weights.
3. Add one visual-perception page covering image classification, YOLO
   detection/segmentation/pose/OBB, semantic segmentation, and monocular or
   stereo depth estimation.
4. Resolve resources relative to the executable/package directory. Do not
   require a source checkout, `CMakeLists.txt`, or machine-specific absolute
   paths at runtime.
5. Reload an engine when its selected model changes, while continuing to cache
   the active model and backend between requests.
6. Validate the packaged app from a clean directory with CPU, CUDA, and SYCL,
   including missing-model and unavailable-backend error states.

Suggested package layout:

```text
GGML-Forge/
  bin/
    ui-demo.exe
    *.dll
  assets/
  models/
    llm/
    asr/
    tts/
    visual_generation/
    visual_perception/
  outputs/
```
