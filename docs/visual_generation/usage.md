# Using Visual Generation

Include `categories/visual_generation/visual_generation.h` and link the
`visual_generation` target. Create a runtime, load model artifacts, and create
a session before issuing image or video requests.

```cpp
visual_runtime_params runtime_params = visual_runtime_default_params();
visual_runtime_ptr runtime = visual_runtime_create(runtime_params);

visual_model_params model_params = visual_model_default_params();
model_params.model = "models/visual_generation/model.safetensors";
visual_model_ptr model = visual_load_model(runtime, &model_params);
visual_session_ptr session = visual_create_session(model);

visual_image_request request = visual_image_request_default_params();
request.prompt = "a small observatory above a sea of clouds";

visual_image* images = nullptr;
size_t image_count = 0;
if (visual_generate_images(session, &request, &images, &image_count)) {
    // Consume width * height * channels bytes from every image.
    visual_free_images(images, image_count);
}

visual_free_session(session);
visual_free_model(model);
visual_runtime_free(runtime);
```

Input `visual_image` buffers are borrowed for the duration of the call. Output
images must be released with `visual_free_images`; video frames and interleaved
float audio must be released together with `visual_free_video`. Preview frames
are borrowed only during their callback.

Use `visual_model_get_capabilities` before optional operations. Cancellation
may be called from another thread with `VISUAL_CANCEL_ALL`,
`VISUAL_CANCEL_NEW_BATCH_ITEMS`, or `VISUAL_CANCEL_RESET`.

Video uses `visual_video_request`; init/end images and control-frame arrays are
optional. `visual_audio.sample_count` is the number of samples per channel and
`data` is interleaved when `channels` is greater than one.

The complete text-to-image example is
[`examples/visual_generation/text-to-image.cpp`](../../examples/visual_generation/text-to-image.cpp).
It writes an RGB PPM directly so the example does not depend on a particular
PNG or JPEG library:

```text
visual-text-to-image <model> [output.ppm] [prompt]
```
