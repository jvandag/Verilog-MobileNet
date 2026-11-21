# Validation record

## Supplied integer model

Dataset: CIFAR-10 bird/cat/dog/horse, seed 42. Train/validation/test counts are
18,000/2,000/4,000; splits are disjoint. The loader preserves the selected
checkpoint's original split membership by splitting its historical intermediate
CIFAR labels before selecting the four deployed classes. Native 32x32 RGB images
are resized with PIL bilinear interpolation to 64x64.

| Metric | Integer result |
| --- | ---: |
| Validation accuracy | 81.65% (2,000 images) |
| Test accuracy | 82.05% (4,000 images) |
| Bird test recall | 89.5% |
| Cat test recall | 75.9% |
| Dog test recall | 73.8% |
| Horse test recall | 89.0% |

Test confusion matrix (rows true, columns predicted; bird/cat/dog/horse):

```text
895   55   30   20
 78  759  134   29
 38  193  738   31
 24   40   46  890
```

The checkpoint used five classifier-warmup epochs and 26 fine-tuning epochs,
selecting fine-tune epoch 18. The release binary hashes appear in
[mobilenet.md](mobilenet.md). The cleanup preserves those binaries; it does not
retrain the selected model or establish performance for new runs.

## Reproduce checks

```bash
python -m pip install -e '.[train,test]'
python -m pytest -q
python -m animal_cnn evaluate --run models/mobilenet025-cifar4
```

The cleanup verification passed all **9 tests** and reran the complete integer
test evaluation, reproducing **82.05%** and the confusion matrix above.

Tests cover model topology and frozen BatchNorm, batch-equivalent integer
inference, full-integer export and folded weights, training/resume/evaluation/
prediction, plateau and interruption checkpoint handling, release bundle
integrity, and preserved CIFAR split membership. Smoke-training tests use tiny
synthetic data and random initialization; they verify execution, not accuracy.
Evaluation and prediction use the integer model without loading a floating
checkpoint. Hardware synthesis, simulation and board checks are future work
listed in [hardware-plan.md](hardware-plan.md).
