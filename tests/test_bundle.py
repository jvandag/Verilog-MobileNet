"""Release compatibility and the dataset memberships behind its metrics."""

import json
from pathlib import Path
import shutil
import numpy as np
import pytest


def test_release_integrity_golden_scores_and_corruption(tmp_path):
    pytest.importorskip("tensorflow")
    from animal_cnn.mobilenet_export import load_bundle, integer_predictions
    root = Path(__file__).resolve().parents[1]/"models/mobilenet025-cifar4/integer"
    manifest = load_bundle(root)
    assert manifest["model_sha256"] == "22620a406aaa14f9e0b21e45c1e60fcc2c8da9ddad1930b2399ab6166a2571d0"
    with np.load(root/"trace.npz") as trace:
        scores = integer_predictions(trace["raw_rgb"], root/"model.tflite")
    assert scores.tolist() == json.loads((root/"expected.json").read_text())["integer_logits"]
    for filename in ("manifest.json", "model.tflite", "constants.bin"):
        shutil.copyfile(root/filename, tmp_path/filename)
    data = bytearray((tmp_path/"constants.bin").read_bytes())
    data[0] ^= 1
    (tmp_path/"constants.bin").write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_cifar_class_mapping_and_published_split_membership(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from animal_cnn.data import load_data
    # Unique IDs expose membership, including the RNG draws of filtered classes.
    train = (np.arange(200), np.repeat(np.arange(10), 20)[:,None])
    test = (np.arange(20), np.repeat(np.arange(10), 2)[:,None])
    monkeypatch.setattr(tf.keras.datasets.cifar10, "load_data", lambda: (train, test))
    training, validation, testing, classes = load_data()
    assert classes == ("bird", "cat", "dog", "horse")
    assert [len(pair[0]) for pair in (training, validation, testing)] == [72, 8, 8]
    assert not set(training[0]) & set(validation[0])
    np.testing.assert_array_equal(testing[0], [4,5,6,7,10,11,14,15])
    np.testing.assert_array_equal(testing[1], [0,0,1,1,2,2,3,3])
    # Frozen expected IDs for seed 42, not a second call to the split function.
    np.testing.assert_array_equal(validation[0], [55,49,78,68,113,103,142,149])
