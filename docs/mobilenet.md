# Supported model and bundle

MobileNetV1-0.25 uses a 3x3 stride-two convolution (8 channels), followed by
13 depthwise-separable blocks. Pointwise output channels are
`16,32,32,64,64,128,128,128,128,128,128,256,256`. Depthwise strides are
`1,2,1,2,1,2,1,1,1,1,1,2,1`. Global average pooling reduces the final 2x2x256
activation to 256 features, followed by four logits. Activations use ReLU6 in
the convolution blocks. BatchNorm is folded into convolution constants at export.
Keras has 219,572 parameters including BatchNorm state.

The supplied integer graph contains 30 operators: one input QUANTIZE,
14 CONV_2D (13 pointwise), 13 DEPTHWISE_CONV_2D, MEAN, and FULLY_CONNECTED.
It requires **3,329,536 MACs per image**; its largest activation is 16,384 bytes.
The external tensor is uint8; internal convolution activations and final scores
are int8. Multiply-accumulate operations use int32 accumulation.

| File under `integer/` | Purpose |
| --- | --- |
| `model.tflite` | 303,208-byte complete integer software reference |
| `constants.bin` | 219,592-byte aligned constant payload, little endian |
| `constants.npz` | Named copy of constants for inspection/tests |
| `manifest.json` | Tensor layouts, offsets, quantization scales/zero points, operator options, hashes |
| `trace.npz` | One raw input and every operator output for RTL comparison |
| `input.mem`, `expected.json` | Quantized external input bytes and expected output/class |

`input.mem` contains exported input codes, not necessarily original raw RGB.
Use `trace.npz`'s `raw_rgb` for an end-to-end firmware input test. Apply the
manifest pixel LUT when entering the software graph. Array layout is NHWC,
standard kernels OHWI, depthwise kernels 1HWC, and dense weights OI.

The payload contains 208,624 int8 weight bytes and 10,968 bytes of int32 constants
(biases and MEAN axis values). It does **not** include a ready-to-load hardware
instruction/quantization configuration: the compiler planned in
[hardware-plan.md](hardware-plan.md) must generate integer multipliers, shifts,
clamps, buffer assignments, and descriptors from the manifest.

Pinned deployment SHA256 values:

```text
model.tflite  22620a406aaa14f9e0b21e45c1e60fcc2c8da9ddad1930b2399ab6166a2571d0
constants.bin 075bb3829951b62b16d36c6a5cada2560790db689a1105773c361aa9d9798597
```

These hashes identify the supplied release; a retrained run has its own manifest
hashes. Training initialization uses the official Keras MobileNet checkpoint,
SHA256 `7d8b7f3c9555cb4579f9659ee233a540e66fd7ad514c0a8b14493b667447c5d6`.
The `.keras` file beside `integer/` exists exclusively for reproducing and
continuing training. See [training-guide.md](training-guide.md).
