# Integer arithmetic contract

Profile: `mobilenet025-tflite-int8-v1`, input `[1,64,64,3]`, classes
bird/cat/dog/horse. All inference tensors are integer. Floating scale values in
the manifest describe quantization; they do not require floating-point hardware.
The software oracle is the pinned TensorFlow 2.20.0 native TFLite interpreter,
without default delegates. RTL must match its exported intermediate vectors.

## Inputs and accumulation

Firmware writes 12,288 raw RGB bytes, row-major HWC, after EXIF orientation,
RGB conversion, and PIL bilinear resize. Training normalizes `pixel/127.5-1`.
Deployment implements that normalization/quantization with the exported 256-entry
`input_pixel_lut`, then the graph's uint8-to-int8 QUANTIZE operation. Do not
assume that raw RGB values equal external quantized codes or simply subtract 128.

For each convolution/dense output channel:

```text
acc = bias_int32[channel] + sum((activation - input_zero_point) * weight_int8)
```

Weights have symmetric zero point zero and per-output-channel scales. Biases
have input_scale * weight_scale[channel] units. A signed int8 product normally
needs 16 bits, and every product is added to a signed **32-bit** accumulator;
it is never truncated back to int8 before summation. Subtracting a nonzero
activation zero point can require a **9-bit** centered activation. Use a widened
9x8 multiply, or fold the offset correction into a checked bias. Export/compiler
analysis must bound every partial sum, using int64 analysis, before allowing
int32 accumulation; never rely on wraparound or silent saturation of the sum.

## Requantization and output

Rescale int32 accumulators with per-channel
`input_scale * weight_scale[channel] / output_scale`, represented as an integer
Q31 multiplier and shift. Implement TFLite-compatible saturating, rounded
multiplication/shift, add the output zero point, and clamp to the fused activation
bounds (including ReLU6) and int8 limits. A simple arithmetic right shift does
not reproduce these semantics. The requantizer may need 64-bit intermediate
products; that does not change the int32 accumulation contract. Exact rounding,
negative values, shift limits, and overflow cases need directed oracle tests.

Convolution uses the manifest strides and padding. For stride-two SAME with an
even input size, padding can be asymmetric (bottom/right), so never hardcode
symmetric padding. Depthwise convolution accumulates only the corresponding
input channel. MEAN averages the final 2x2 spatial samples with the oracle's
rounding/rescaling. Do not substitute an unverified floor shift by two.

Dense accumulators may use different channel scales. Requantize all four outputs
to the common final int8 scale **before argmax**. Ties select the lowest class
index. Software and the planned LOGITS registers expose these final signed int8
scores (sign extended to 32 bits in MMIO), not raw int32 accumulators. Softmax is
unnecessary for classification and scores are not probabilities.

The manifest, constant layout, graph, and golden vectors are authoritative.
Changing architecture, preprocessing, conversion/runtime version, or compiler
rounding requires revalidation. The hardware coefficient/descriptor generator
and SystemVerilog implementation remain planned work.
