# Training the integer deployment model

The only supported architecture is MobileNetV1-0.25, 64x64 RGB, four classes in
bird/cat/dog/horse order. Training uses Keras floating point; conversion folds
batch normalization and produces an integer-only inference graph. A floating
checkpoint is required for further gradient training, not for deployment.

## Bash setup and commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
python -m animal_cnn --help
python -m animal_cnn train --output artifacts/my-run --evaluate-test
python -m animal_cnn evaluate --run artifacts/my-run
python -m animal_cnn predict --run artifacts/my-run --image ./animal.jpg
```

For Windows Git Bash use `py -3.11 -m venv .venv` and
`source .venv/Scripts/activate`. All subsequent commands are identical.
`python -m animal_cnn.mobilenet` remains an alias for the same CLI.
The first training run downloads CIFAR-10 and the official ImageNet
MobileNetV1 alpha=0.25, 128px, no-top checkpoint into `.keras/`. Its convolution
weights initialize the 64px network. There is no second deployment architecture.

Default training warms up the classifier for five epochs at learning rate 0.001,
then fine-tunes convolutions for up to 40 epochs at 0.00003. Batch size is 64;
BatchNorm statistics stay frozen and training images receive random horizontal
flips. Validation chooses the checkpoint; test images never calibrate the export.
Use new output directories: existing runs are not overwritten.

## Stop on a plateau or Ctrl+C

```bash
python -m animal_cnn train --resume models/mobilenet025-cifar4 \
  --output artifacts/continued --finetune-epochs 0 \
  --finetune-lr 0.00001 --patience 12 --lr-patience 4
```

An epoch count of zero removes the practical epoch limit for that phase.
Validation-accuracy early stopping remains active: `--patience` controls how many
epochs without sufficient improvement are allowed (`min_delta=0.001`).
`--lr-patience` controls when the learning rate is reduced on a plateau.
The first Ctrl+C requests a graceful stop after the current batch and validation,
then selects a validated checkpoint and exports it. Allow export to finish;
a second Ctrl+C interrupts immediately. A hard kill cannot save/export cleanly.
Resume skips classifier warmup and creates a fresh optimizer. It does not alter
the source run. Head/fine-tune histories and metadata record selection decisions.

More epochs can help while validation improves. They do not guarantee improvement
and can overfit. Compare **integer validation accuracy** after export: calibration
can change the ranking of floating checkpoints. Tune learning rate, patience and
calibration sample count using validation; reserve the test set for final checks.
`--calibration-samples 512` or `1024` can be compared with the default 256 using
new runs. No improved accuracy is promised by a larger calibration subset.

Changing kernels or channels changes this hardware profile and invalidates the
current hardware plan. The fixed profile uses 3x3 spatial convolutions and 1x1
pointwise convolutions. Prefer training adjustments before architectural changes.
The source dataset remains 32x32; making resized images larger creates more
computation without recovering missing source detail.

Outputs include `model.keras` (training only), `training.json`, phase checkpoints
and logs, `validation.json`, and `integer/`. `--evaluate-test` writes
`evaluation.json` after selection; otherwise run `evaluate` separately. Prediction
and evaluation need no `.keras` checkpoint. Predicted integer logits are scores,
not probability percentages. See [the numeric contract](numeric-contract.md).
