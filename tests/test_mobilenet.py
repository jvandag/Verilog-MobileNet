import hashlib
import json
import numpy as np
import pytest

tf=pytest.importorskip("tensorflow")
from animal_cnn.mobilenet import build_model,configure_phase,normalize,resize_images
from animal_cnn.mobilenet_export import export_model,integer_predictions


def test_mobilenet_topology_and_transfer_phases():
    model=build_model(64,4,pretrained=False)
    assert model.count_params()==219572
    assert sum(isinstance(layer,tf.keras.layers.DepthwiseConv2D) for layer in model.layers)==13
    assert sum(isinstance(layer,tf.keras.layers.Conv2D) for layer in model.layers)==14
    configure_phase(model,False)
    assert len(model.trainable_weights)==2
    configure_phase(model,True)
    assert len(model.trainable_weights)>2
    batch_norm=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization)]
    assert len(batch_norm)==27 and all(not layer.trainable for layer in batch_norm)
    before=[layer.moving_mean.numpy().copy() for layer in batch_norm]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.train_on_batch(normalize(np.zeros((2,64,64,3),dtype=np.uint8)),np.array([0,1]))
    for layer,expected in zip(batch_norm,before):
        np.testing.assert_array_equal(layer.moving_mean,expected)


def test_full_integer_export_bn_folding_and_batched_inference(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model=build_model(64,4,pretrained=False)
    configure_phase(model,True)
    images=np.random.default_rng(4).integers(0,256,(4,64,64,3),dtype=np.uint8)
    manifest=export_model(model,images,("bird","cat","dog","horse"),tmp_path)
    assert manifest["format"]=="mobilenet025-tflite-int8-v1"
    assert manifest["input_shape"]==[1,64,64,3]
    assert manifest["macs_per_image"]>3000000
    assert manifest["max_activation_elements"]==16384
    assert 200000<manifest["constant_bytes"]<230000
    assert hashlib.sha256((tmp_path/"model.tflite").read_bytes()).hexdigest()==manifest["model_sha256"]
    assert hashlib.sha256((tmp_path/"constants.bin").read_bytes()).hexdigest()==manifest["constants_sha256"]
    assert all(t["dtype"] in ("int8","uint8","int32","int64","bool") for t in manifest["tensors"])
    assert sum(op["kind"]=="DEPTHWISE_CONV_2D" for op in manifest["operators"])==13
    scores=integer_predictions(images,tmp_path/"model.tflite",batch_size=3)
    np.testing.assert_array_equal(scores,integer_predictions(images,tmp_path/"model.tflite",batch_size=1))
    assert scores.dtype==np.int8 and scores.shape==(4,4)
    assert json.loads((tmp_path/"expected.json").read_text())["integer_logits"]==scores[:1].tolist()
    first=next(op for op in manifest["operators"] if op["kind"]=="CONV_2D")
    tensor_index=first["inputs"][1]
    detail=next(t for t in manifest["tensors"] if t["index"]==tensor_index)
    with np.load(tmp_path/"constants.npz") as tensors:
        codes=tensors[f"tensor_{tensor_index}"]
    scale=np.array(detail["scales"])[:,None,None,None]
    dequantized=(codes-np.array(detail["zero_points"])[:,None,None,None])*scale
    conv=model.get_layer("conv1")
    bn=model.get_layer("conv1_bn")
    expected=conv.kernel.numpy()*(bn.gamma.numpy()/np.sqrt(bn.moving_variance.numpy()+bn.epsilon))
    expected=expected.transpose(3,0,1,2)
    assert np.all(np.abs(dequantized-expected)<=scale*0.51+1e-7)
    assert first["StrideH"]==2 and first["fused_activation"]=="RELU6"


def test_mobilenet_resize_and_normalization():
    images=np.zeros((2,32,32,3),dtype=np.uint8)
    assert resize_images(images,64).shape==(2,64,64,3)
    np.testing.assert_allclose(normalize(np.array([0,255],dtype=np.uint8)),[-1,1])


def test_mobilenet_cli_training_resume_evaluate_predict(tmp_path,monkeypatch,capsys):
    from animal_cnn import mobilenet
    rng=np.random.default_rng(5)
    pair=(rng.integers(0,256,(8,32,32,3),dtype=np.uint8),np.arange(8)%4)
    classes=("bird","cat","dog","horse")
    monkeypatch.setattr(mobilenet,"load_data",lambda **kwargs:(pair,pair,pair,classes))

    def run(*args):
        monkeypatch.setattr("sys.argv",["mobilenet",*map(str,args)])
        mobilenet.main()

    out=tmp_path/"run"
    run("train","--output",out,"--random-init","--head-epochs","1","--finetune-epochs","1",
        "--calibration-samples","4","--batch-size","4")
    metadata=json.loads((out/"training.json").read_text())
    assert metadata["pretrained_source"] is None
    original=hashlib.sha256((out/"model.keras").read_bytes()).hexdigest()
    continued=tmp_path/"continued"
    run("train","--resume",out,"--output",continued,"--finetune-epochs","1",
        "--calibration-samples","4","--batch-size","4")
    metadata=json.loads((continued/"training.json").read_text())
    assert metadata["source_model_sha256"]==original
    assert set(metadata["phases"])=={"finetune"}
    assert hashlib.sha256((out/"model.keras").read_bytes()).hexdigest()==original
    (continued/"model.keras").unlink()  # Deployment evaluation needs no float checkpoint.
    run("evaluate","--run",continued)
    assert json.loads((continued/"evaluation.json").read_text())["samples"]==8
    from PIL import Image
    Image.fromarray(pair[0][0]).save(tmp_path/"image.png")
    capsys.readouterr()
    run("predict","--run",continued,"--image",tmp_path/"image.png")
    assert json.loads(capsys.readouterr().out)["animal"] in classes
