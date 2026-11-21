# Int8 MobileNetV1-0.25: four CIFAR animals

The sole deployment model takes 64x64 RGB and returns four signed int8 scores:
`bird=0, cat=1, dog=2, horse=3`. Convolutions and dense layers accumulate in
int32. The external TFLite input is uint8; use the exported pixel LUT rather than
assuming raw RGB equals quantized input codes.

| Split | Integer accuracy |
| --- | ---: |
| Validation (2,000 images) | 81.65% |
| Test (4,000 images) | 82.05% |

`integer/` contains the full-integer graph, constants, quantization manifest and
golden intermediate vectors. `model.keras` is retained exclusively for training
and re-export; integer evaluation/prediction do not load it. Run metadata and
phase histories preserve the selected training run's provenance. Original output
paths in `training.json` refer to that run before its selected files were copied
here. Native CIFAR images are 32x32 and resized with PIL bilinear interpolation.

```bash
python -m animal_cnn predict --run models/mobilenet025-cifar4 --image ./animal.jpg
python -m animal_cnn evaluate --run models/mobilenet025-cifar4
```

See [training](../../docs/training-guide.md), [bundle details and hashes](../../docs/mobilenet.md),
and the [hardware plan](../../docs/hardware-plan.md). The 219,592-byte constants
payload still needs a hardware configuration compiler and SystemVerilog engine.
No FPGA execution or fit is claimed.
