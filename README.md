# FPGA animal CNN

This repository supports one deployment profile: **MobileNetV1-0.25**, 64x64 RGB,
with **int8 weights/activations and int32 biases/accumulators**. Class IDs are
`0=bird, 1=cat, 2=dog, 3=horse`. It is a 2D CNN using standard, depthwise, and
pointwise convolutions, global average pooling, and a four-output classifier.

The supplied integer model achieved **82.05% test accuracy** on 4,000 held-out
CIFAR-10 animal images and **81.65% validation accuracy** on 2,000 images.
CIFAR's native 32x32 photos are resized to 64x64 with PIL bilinear interpolation;
upsampling does not add detail. The training split contains 18,000 images.

## Run it

Bash on Linux/macOS (Python 3.11-3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
python -m animal_cnn evaluate --run models/mobilenet025-cifar4
python -m animal_cnn predict --run models/mobilenet025-cifar4 --image /path/to/animal.jpg
```

In Windows Git Bash, create the environment with `py -3.11 -m venv .venv` and
activate it with `source .venv/Scripts/activate` instead. Evaluation downloads
CIFAR-10 on first use; prediction only needs the supplied bundle and your image.
For inference without training dependencies, install `'.[runtime]'`.

```bash
# Fresh pretrained run; select the best validation checkpoint, then export int8.
python -m animal_cnn train --output artifacts/my-run --evaluate-test
# Continue fine-tuning until validation plateaus or you press Ctrl+C.
python -m animal_cnn train --resume models/mobilenet025-cifar4 \
  --output artifacts/continued --finetune-epochs 0 --patience 8
python -m pytest -q
```

## Repository contents

- `software/animal_cnn/`: Keras training, integer conversion, evaluation, prediction.
- `models/mobilenet025-cifar4/integer/`: full-integer TFLite reference, binary
  constants, quantization manifest, and intermediate golden vectors.
- `models/mobilenet025-cifar4/model.keras`: **training checkpoint only**; retained
  so you can resume training and regenerate integer weights. Keras training uses
  floating point. Deployment inference uses the integer bundle exclusively.
- `docs/`: [training](docs/training-guide.md), [model and artifacts](docs/mobilenet.md),
  [numeric contract](docs/numeric-contract.md), [hardware plan](docs/hardware-plan.md),
  and [validation](docs/validation.md).

The [software development summary](docs/software-development-summary.md)
records the requests and decisions from this chat that led to the current model.

The intended parent is [ECE544_Final_Project](https://github.com/CaseyRasor/ECE544_Final_Project).
**SystemVerilog CNN RTL is not implemented yet.** The hardware plan targets its
100 MHz Wishbone bus and Nexys A7-100T. FPGA utilization numbers are planning
estimates until synthesis and integration establish the actual fit.
