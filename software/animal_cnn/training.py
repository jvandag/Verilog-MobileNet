"""Training termination and checkpoint handling, independent of CNN topology."""

import signal
from pathlib import Path
from .io import write_json


def fit_model(model, training_data, validation_data, output, *, epochs=80,
              monitor="val_loss", patience=12, lr_patience=5, min_delta=0.0,
              early_stopping=True, extra_callbacks=()):
    """First Ctrl+C requests a stop after a batch; always select a validated best.

    epochs=0 removes the practical epoch limit. Must run in the main thread, as
    the CLI does. A hard process kill cannot execute Python cleanup.
    """
    from tensorflow import keras

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mode = "min" if monitor == "val_loss" else "max"
    limit = epochs if epochs else 2**31 - 1
    requested = False
    completed_batches = 0
    history = {}

    class Progress(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            nonlocal completed_batches
            completed_batches += 1
            if requested:
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            for key, value in (logs or {}).items():
                history.setdefault(key, []).append(float(value))
            write_json(output / "history.json", history)
            if requested:
                self.model.stop_training = True

    def request_stop(signum, frame):
        nonlocal requested
        if requested:
            raise KeyboardInterrupt
        requested = True
        print("\nStop requested. Finishing the current batch/validation and saving the best model.", flush=True)

    stopper = keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                           patience=patience, min_delta=min_delta)
    callbacks = [
        keras.callbacks.ModelCheckpoint(output / "best.keras", monitor=monitor,
                                        mode=mode, save_best_only=True),
        Progress(),
        keras.callbacks.CSVLogger(str(output / "training.csv")),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode,
                                         patience=lr_patience, factor=0.5, min_lr=1e-6),
    ]
    if early_stopping:
        callbacks.append(stopper)
    callbacks.extend(extra_callbacks)
    # Usable even when interruption happens before the first validation result.
    model.save(output / "initial.keras")
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        try:
            model.fit(training_data, validation_data=validation_data, epochs=limit,
                      callbacks=callbacks, verbose=2, shuffle=False)
        except KeyboardInterrupt:
            requested = True
            logger = callbacks[2]
            if getattr(logger, "csv_file", None) is not None:
                logger.on_train_end()
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    model.save(output / "last.keras")
    checkpoint = output / "best.keras"
    selected = "best_training_epoch" if checkpoint.exists() else "initial_checkpoint"
    checkpoint = checkpoint if checkpoint.exists() else output / "initial.keras"
    model.set_weights(keras.models.load_model(checkpoint, compile=False).get_weights())
    write_json(output / "history.json", history)
    values = history.get(monitor, [])
    best_epoch = (min if mode == "min" else max)(range(len(values)), key=values.__getitem__) + 1 if values else None
    reason = "cancelled" if requested else "plateau" if early_stopping and stopper.stopped_epoch > 0 else "epoch_limit"
    return dict(history=history, stop_reason=reason, selection=selected,
                best_epoch=best_epoch, training_batches_completed=completed_batches)
