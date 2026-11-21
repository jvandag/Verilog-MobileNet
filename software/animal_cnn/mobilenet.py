"""MobileNetV1-0.25 training and full-integer deployment.

Run with python -m animal_cnn.mobilenet train/evaluate/predict.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from . import DEFAULT_CLASSES
from .data import load_data, sample
from .io import write_json
from .training import fit_model


def tensorflow(seed=42):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_HOME", str(Path(".keras").resolve()))
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def resize_images(images, size):
    """Explicit PIL resize; used identically for training and inference."""
    return np.stack([np.asarray(Image.fromarray(image).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.uint8) for image in images])


def normalize(images):
    # Training normalization; deployment uses the exported integer pixel LUT.
    return np.asarray(images, dtype=np.float32) / 127.5 - 1.0


def build_model(size=64, num_classes=4, pretrained=True):
    from tensorflow import keras
    if size != 64 or num_classes != 4:
        raise ValueError("The supported profile is 64x64 RGB with four classes")
    # Load the official 128px checkpoint explicitly. Convolution/BN weights do
    # not depend on spatial shape; changing resolution does not add image detail.
    base = keras.applications.MobileNet(input_shape=(size, size, 3), alpha=0.25,
        include_top=False, weights=None, pooling="avg")
    if pretrained:
        source = keras.applications.MobileNet(input_shape=(128,128,3), alpha=0.25,
            include_top=False, weights="imagenet", pooling="avg")
        base.set_weights(source.get_weights())
    output = keras.layers.Dense(num_classes, name="logits")(base.output)
    return keras.Model(base.input, output, name="mobilenet025_animals")


def configure_phase(model, fine_tune):
    from tensorflow import keras
    for layer in model.layers:
        # Freeze moving statistics throughout: small-data fine-tuning should not
        # overwrite pretrained BN estimates. Convolutions still receive gradients.
        layer.trainable = (fine_tune and not isinstance(layer, keras.layers.BatchNormalization)) or layer.name == "logits"


def float_evaluate(model, images, labels, classes, batch_size=64):
    predictions = []
    for start in range(0, len(images), batch_size):
        logits = np.asarray(model(normalize(images[start:start+batch_size]), training=False))
        predictions.extend(logits.argmax(axis=1))
    confusion = np.zeros((len(classes),len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return dict(samples=len(images), classes=list(classes),
                float_accuracy=float(np.mean(np.asarray(predictions)==labels)),
                float_confusion_matrix=confusion.tolist(), confusion_matrix_axes="rows=true, columns=predicted")


def train(args):
    tf = tensorflow(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new output directory")
    train_set, val_set, test_set, classes = load_data(seed=args.seed, classes=DEFAULT_CLASSES)
    train_set = sample(*train_set, args.train_limit, args.seed)
    val_set = sample(*val_set, args.val_limit, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "classes.json", list(classes))
    source_metadata = None
    source = Path(args.resume) if args.resume else None
    if source:
        source_metadata = json.loads((source/"training.json").read_text())
        if source_metadata.get("architecture")!="MobileNetV1" or source_metadata.get("alpha")!=0.25:
            raise ValueError("Resume requires a MobileNetV1-0.25 run")
        if json.loads((source/"classes.json").read_text())!=list(classes):
            raise ValueError("Resume class order mismatch")
        model = tf.keras.models.load_model(source/"model.keras",compile=False)
        if tuple(model.input_shape[1:])!=(args.image_size,args.image_size,3) or model.output_shape[-1]!=len(classes):
            raise ValueError("Resume dimensions mismatch")
    else:
        model = build_model(args.image_size, len(classes), pretrained=not args.random_init)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    flip = tf.keras.layers.RandomFlip("horizontal", seed=args.seed)
    # Resize once in uint8 to keep memory bounded and match PIL deployment.
    train_x = resize_images(train_set[0], args.image_size)
    val_x = resize_images(val_set[0], args.image_size)

    def dataset(x, labels, training=False):
        data = tf.data.Dataset.from_tensor_slices((x,labels)).with_options(options)
        if training:
            data = data.shuffle(len(x), seed=args.seed)
        data = data.batch(args.batch_size)
        def prepare(a, b):
            a = tf.cast(a,tf.float32)/127.5-1
            return (flip(a, training=True) if training else a), b
        return data.map(prepare, num_parallel_calls=1).prefetch(1)

    training_data = dataset(train_x, train_set[1], True)
    validation_data = dataset(val_x, val_set[1])
    phases = {}
    plan = [("finetune",args.finetune_epochs,args.finetune_lr)] if source else [
        ("head",args.head_epochs,args.head_lr),("finetune",args.finetune_epochs,args.finetune_lr)]
    for phase, epochs, learning_rate in plan:
        if phase == "finetune" and phases.get("head",{}).get("stop_reason") == "cancelled":
            break
        initial = float_evaluate(model,val_x,val_set[1],classes) if phase == "finetune" else None
        before = model.get_weights() if phase == "finetune" else None
        configure_phase(model, phase == "finetune")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        result = fit_model(model,training_data,validation_data,out/phase,epochs=epochs,
            monitor="val_accuracy",patience=args.patience,lr_patience=args.lr_patience,min_delta=0.001)
        phases[phase] = {key:value for key,value in result.items() if key != "history"}
        if initial:
            after = float_evaluate(model,val_x,val_set[1],classes)
            if after["float_accuracy"] < initial["float_accuracy"]:
                model.set_weights(before)
                phases[phase]["selection"] = "incoming_checkpoint"
        model.save(out/"model.keras")
    model.save(out/"model.keras")
    write_json(out/"float_validation.json",float_evaluate(model,val_x,val_set[1],classes))
    weights_file = Path(os.environ["KERAS_HOME"])/"models/mobilenet_2_5_128_tf_no_top.h5"
    metadata = dict(architecture="MobileNetV1",alpha=0.25,input_shape=[args.image_size,args.image_size,3],
        source_native_shape=[32,32,3],preprocessing="PIL bilinear resize, float=pixel/127.5-1",
        pretrained_source=source_metadata.get("pretrained_source") if source else "ImageNet MobileNetV1 0.25, 128px, no top" if not args.random_init else None,
        pretrained_sha256=source_metadata.get("pretrained_sha256") if source else hashlib.sha256(weights_file.read_bytes()).hexdigest() if not args.random_init else None,
        source_model_sha256=hashlib.sha256((source/"model.keras").read_bytes()).hexdigest() if source else None,
        train_samples=len(train_x),validation_samples=len(val_x),seed=args.seed,
        tensorflow=tf.__version__,keras=tf.keras.__version__,classes=list(classes),
        parameters=model.count_params(),phases=phases,arguments=vars(args))
    write_json(out/"training.json",metadata)
    calibration,_ = sample(*train_set,args.calibration_samples,args.seed+1)
    from .mobilenet_export import export_model, evaluate_integer
    export_model(model,resize_images(calibration,args.image_size),classes,out/"integer")
    report = evaluate_integer(val_x,val_set[1],classes,out/"integer/model.tflite")
    write_json(out/"validation.json",report)
    print(json.dumps({"validation":report},indent=2),flush=True)
    if args.evaluate_test and not any(p["stop_reason"]=="cancelled" for p in phases.values()):
        test_x = resize_images(test_set[0],args.image_size)
        write_json(out/"evaluation.json",evaluate_integer(test_x,test_set[1],classes,out/"integer/model.tflite"))


def evaluate(args):
    tensorflow()
    root = Path(args.run)
    from .mobilenet_export import load_bundle
    manifest = load_bundle(root/"integer")
    classes = manifest["classes"]
    _,_,test,available = load_data(seed=args.seed,classes=classes)
    if list(available)!=classes:
        raise ValueError("Class order mismatch")
    from .mobilenet_export import evaluate_integer
    images = resize_images(test[0],manifest["input_shape"][1])
    report = evaluate_integer(images,test[1],classes,root/"integer/model.tflite")
    write_json(Path(args.output) if args.output else root/"evaluation.json",report)
    print(json.dumps(report,indent=2))


def predict(args):
    tensorflow()
    root=Path(args.run)
    from .mobilenet_export import load_bundle
    manifest=load_bundle(root/"integer")
    size=manifest["input_shape"][1]
    with Image.open(args.image) as image:
        images=np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((size,size),Image.Resampling.BILINEAR))[None]
    from .mobilenet_export import integer_predictions
    scores=integer_predictions(images,root/"integer/model.tflite")[0]
    index=int(scores.argmax())
    print(json.dumps(dict(class_id=index,animal=manifest["classes"][index],integer_logits=scores.tolist()),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    training=commands.add_parser("train")
    training.add_argument("--output",required=True)
    training.add_argument("--resume",help="Continue a MobileNet run with a fresh optimizer, skipping classifier warmup")
    training.add_argument("--image-size",type=int,choices=(64,),default=64)
    training.add_argument("--head-epochs",type=int,default=5)
    training.add_argument("--finetune-epochs",type=int,default=40)
    training.add_argument("--head-lr",type=float,default=0.001)
    training.add_argument("--finetune-lr",type=float,default=0.00003)
    training.add_argument("--batch-size",type=int,default=64)
    training.add_argument("--patience",type=int,default=8)
    training.add_argument("--lr-patience",type=int,default=3)
    training.add_argument("--seed",type=int,default=42)
    training.add_argument("--calibration-samples",type=int,default=256)
    training.add_argument("--train-limit",type=int)
    training.add_argument("--val-limit",type=int)
    training.add_argument("--random-init",action="store_true",help="Pipeline testing only; omit to use pretrained weights")
    training.add_argument("--evaluate-test",action="store_true",help="Evaluate test data after selection; omit during experiments")
    evaluation=commands.add_parser("evaluate")
    evaluation.add_argument("--run",required=True)
    evaluation.add_argument("--seed",type=int,default=42)
    evaluation.add_argument("--output")
    prediction=commands.add_parser("predict")
    prediction.add_argument("--run",required=True)
    prediction.add_argument("--image",required=True)
    args=parser.parse_args()
    if args.command=="train":
        for field in ("head_epochs","finetune_epochs"):
            if getattr(args,field)<0:
                parser.error(f"--{field.replace('_','-')} must be nonnegative; 0 stops on plateau or Ctrl+C")
        if args.resume and args.random_init:
            parser.error("--resume and --random-init cannot be combined")
        for field in ("batch_size","patience","lr_patience","calibration_samples"):
            if getattr(args,field)<=0:
                parser.error(f"--{field.replace('_','-')} must be positive")
        for field in ("head_lr","finetune_lr"):
            if not np.isfinite(getattr(args,field)) or getattr(args,field)<=0:
                parser.error(f"--{field.replace('_','-')} must be finite and positive")
        for field in ("train_limit","val_limit"):
            if getattr(args,field) is not None and getattr(args,field)<=0:
                parser.error(f"--{field.replace('_','-')} must be positive")
    {"train":train,"evaluate":evaluate,"predict":predict}[args.command](args)


if __name__=="__main__":
    main()
