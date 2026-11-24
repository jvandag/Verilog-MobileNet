# Real-image regression inputs

These are the first CIFAR-10 **test** images for bird, cat, dog and horse,
losslessly extracted as native 32x32 RGB PNG files. Zero-based indices within
the official Python distribution's `test_batch`: bird 25, cat 0, dog 12, horse 13.
Source dataset: https://www.cs.toronto.edu/~kriz/cifar.html (Krizhevsky, 2009).
They are regression inputs, not a new accuracy evaluation.

`tools/compile_hardware.py` resizes them with the deployment preprocessing and
creates fresh reference tensors with the pinned TFLite interpreter. Default
compilation includes the supplied `trace.npz` image and these four images.
