import signal
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from animal_cnn.training import fit_model


def fixture_model():
    model = tf.keras.Sequential([tf.keras.Input((1,)),
        tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0), loss="mse")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    data = tf.data.Dataset.from_tensor_slices((np.ones((8, 1)), np.ones((8, 1))))
    return model, data.with_options(options).batch(2)


def test_unlimited_training_stops_on_validation_plateau(tmp_path):
    model, data = fixture_model()
    result = fit_model(model, data, data, tmp_path, epochs=0, patience=2,
                       lr_patience=1, min_delta=0.001)
    assert result["stop_reason"] == "plateau"
    assert len(result["history"]["loss"]) == 3
    assert result["best_epoch"] == 1
    for name in ("initial.keras", "last.keras", "best.keras", "training.csv", "history.json"):
        assert (tmp_path / name).exists()


def test_unlimited_training_can_be_cancelled_without_early_stopping(tmp_path):
    model, data = fixture_model()
    previous = signal.getsignal(signal.SIGINT)

    class Cancel(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch == 0:
                signal.raise_signal(signal.SIGINT)

    result = fit_model(model, data, data, tmp_path, epochs=0,
                       early_stopping=False, extra_callbacks=[Cancel()])
    assert result["stop_reason"] == "cancelled"
    assert result["training_batches_completed"] < 4
    assert signal.getsignal(signal.SIGINT) == previous
    assert result["selection"] == "best_training_epoch"
    restored = tf.keras.models.load_model(tmp_path / "best.keras", compile=False)
    np.testing.assert_array_equal(model(np.ones((1, 1))), restored(np.ones((1, 1))))


def test_interruption_before_first_validation_recovers_initial_model(tmp_path):
    model, data = fixture_model()
    before = model.get_weights()

    class Interrupt(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            raise KeyboardInterrupt

    result = fit_model(model, data, data, tmp_path, epochs=0, extra_callbacks=[Interrupt()])
    assert result["stop_reason"] == "cancelled"
    assert result["best_epoch"] is None
    assert result["history"] == {}
    assert result["selection"] == "initial_checkpoint"
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)
