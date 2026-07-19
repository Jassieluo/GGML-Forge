# Quantization Runtime

The runtime separates artifact storage from backend compute representation.

Runtime ConvND storage includes Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1,
Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS, and MXFP4. The common Python exporter can
currently encode Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q8_0, and MXFP4. Q4_K_M is an
export policy that uses Q4_K with Q8_0/F16 fallbacks; it is not a separate GGML
tensor type.

## Weight contract

GGUF tensors keep their artifact storage type and shape in `WeightDescriptor`.
Quantized convolution tensors use:

- `tts.weights.conv1d_quant_layout = "channel_rows"`
- `tts.weights.packed_convolution_tensors = [...]`

The tensor list is mandatory when the layout key is present. This prevents the
runtime from guessing a tensor layout from its dtype or name.

For packed Conv1D and ConvTranspose1D weights, GGUF stores
`[channel, kernel, outer]` so each quantization row is block aligned. The generic
loader preserves this shape and the original packed storage type. CPU,
CUDA, and SYCL Conv1D/ConvTranspose1D kernels decode only the blocks needed by
the current calculation and accumulate in F32. They never create a complete
F16/F32 copy of a compressed convolution tensor.

Conv2D accepts `flattened_rows` or channel-row storage according to its schema.
Flattened quantized kernels are stored as
`[kernel_width * kernel_height * input_channels, output_channels]`, while
`nn.logical_shape.*` metadata preserves the original four-dimensional kernel
shape. CPU, CUDA, and SYCL consume either layout directly without constructing
an im2col activation tensor. Kernels whose selected row is not block aligned
fall back through compatible storage types and finally F16.

## Compute policy

- Q4/Q8 linear weights remain quantized and are consumed by backend matmul.
- CUDA and SYCL consume F16 linear weights directly.
- CPU promotes F16 linear weights and F16 linear inputs to F32 when required by
  its matmul kernels.
- Conv1D and ConvTranspose1D consume Q4_0/Q4_K/Q8_0/F16 weights directly and
  always execute through the project-owned CPU, CUDA, or SYCL implementation.
- CPU low-bit and F16 convolution paths use AVX2/FMA block dot or AXPY loops;
  CUDA and SYCL decode blocks inside their device kernels.
- Activations are F32 by default. Operators may preserve F16 activations where
  supported, but low-bit activation quantization is not used.
- Attention Q/K scores, normalization statistics, softmax, and matmul
  accumulation remain F32. KV values may remain F16.

`GGUFModel` owns descriptors and backend-resident weights. Sessions only share
these immutable model-replica resources; no process-global cache is keyed by a
tensor data pointer.

Model-specific immutable transforms use the same `ModelTensorArena` through
`prepared_weights`. This includes BERT embedding dequantization, HuBERT
WeightNorm folding, and repeated BigVGAN anti-alias filters. Staged host data is
released immediately after the arena is uploaded.

Mutable inputs are not allowed in `prepared_weights`. BERT and HuBERT create
their inputs in each execution graph. VITS currently uses a separate
`runtime_tensors` arena while the pipeline serializes inference; moving that
arena into `ITTSSession` is required before removing the pipeline inference
mutex.
