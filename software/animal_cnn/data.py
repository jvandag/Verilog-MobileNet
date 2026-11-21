"""CIFAR-10 bird/cat/dog/horse data with reproducible held-out splits."""

import numpy as np
from . import DEFAULT_CLASSES


def stratified_split(images, labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = max(1, int(round(len(indices) * fraction)))
        if count >= len(indices):
            raise ValueError("Each class needs at least two training images")
        validation.extend(indices[:count])
        train.extend(indices[count:])
    train, validation = rng.permutation(train), np.asarray(validation)
    return (images[train], labels[train]), (images[validation], labels[validation])


def sample(images, labels, count, seed):
    """Balanced subset, shuffled for representative calibration."""
    if count is None or count >= len(images):
        return images, labels
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in np.unique(labels)]
    chosen = []
    while len(chosen) < count:
        for bucket in buckets:
            if bucket and len(chosen) < count:
                chosen.append(bucket.pop())
    chosen = rng.permutation(chosen)
    return images[chosen], labels[chosen]


def load_data(seed=42, classes=DEFAULT_CLASSES):
    """18,000 train / 2,000 validation / 4,000 test images at the default seed.

    Split before filtering to retain the published checkpoint's exact membership.
    The intermediate six CIFAR labels are a reproducibility detail, not a model.
    """
    if tuple(classes) != DEFAULT_CLASSES:
        raise ValueError(f"This model requires class order {DEFAULT_CLASSES}")
    from tensorflow import keras
    (x, y), (xt, yt) = keras.datasets.cifar10.load_data()
    split_ids = np.array([2, 3, 4, 5, 6, 7])

    def split_animals(images, labels):
        labels = labels.reshape(-1)
        mask = np.isin(labels, split_ids)
        return images[mask], np.searchsorted(split_ids, labels[mask])

    def selected(pair):
        images, labels = pair
        mapped = np.array([0, 1, -1, 2, -1, 3])[labels]
        keep = mapped >= 0
        return images[keep], mapped[keep]

    train, validation = stratified_split(*split_animals(x, y), 0.1, seed)
    return selected(train), selected(validation), selected(split_animals(xt, yt)), DEFAULT_CLASSES
