"""Full-integer MobileNet reference/export for FPGA implementation.

TFLite carries
per-channel quantization, fused activation limits, and exact operator options.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from .io import write_json, write_mem
from .mobilenet import normalize
from . import DEFAULT_CLASSES


def interpreter(path, preserve=False):
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=str(path),num_threads=2,
        experimental_preserve_all_tensors=preserve,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)


def pixel_lut(detail):
    scale,zero=detail["quantization"]
    if scale<=0:
        raise ValueError("Missing integer input scale")
    bounds=np.iinfo(detail["dtype"])
    return np.clip(np.rint(normalize(np.arange(256,dtype=np.uint8))/scale+zero),
                   bounds.min,bounds.max).astype(detail["dtype"])


def integer_predictions(images,path,batch_size=32):
    runner=interpreter(path)
    inp=runner.get_input_details()[0]
    lut=pixel_lut(inp)
    results=[]
    current=None
    for start in range(0,len(images),batch_size):
        batch=images[start:start+batch_size]
        if len(batch)!=current:
            runner.resize_tensor_input(inp["index"],list(batch.shape),strict=True)
            runner.allocate_tensors()
            current=len(batch)
        runner.set_tensor(inp["index"],lut[batch])
        runner.invoke()
        results.append(runner.get_tensor(runner.get_output_details()[0]["index"]))
    return np.concatenate(results)


def evaluate_integer(images,labels,classes,path):
    report=dict(samples=len(images), classes=list(classes),
                confusion_matrix_axes="rows=true, columns=predicted")
    scores=integer_predictions(images,path)
    prediction=scores.argmax(axis=1)
    confusion=np.zeros((len(classes),len(classes)),dtype=np.int64)
    np.add.at(confusion,(labels,prediction),1)
    correct=float(np.mean(prediction==labels))
    report.update(integer_accuracy=correct,
                  integer_confusion_matrix=confusion.tolist())
    return report


def export_model(model,calibration,classes,directory):
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=lambda: ([normalize(image[None])] for image in calibration)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.uint8
    converter.inference_output_type=tf.int8
    data=converter.convert()
    (out/"model.tflite").write_bytes(data)
    runner=interpreter(out/"model.tflite",preserve=True)
    runner.allocate_tensors()
    details=runner.get_tensor_details()
    allowed={np.dtype(np.int8),np.dtype(np.uint8),np.dtype(np.int32),np.dtype(np.int64),np.dtype(np.bool_)}
    if any(np.dtype(detail["dtype"]) not in allowed for detail in details):
        raise ValueError("Conversion left floating-point runtime tensors")
    graph=schema.Model.GetRootAsModel(data,0)
    subgraph=graph.Subgraphs(0)
    tensor_details={d["index"]:d for d in details}
    manifest=dict(format="mobilenet025-tflite-int8-v1",classes=list(classes),
        input_shape=runner.get_input_details()[0]["shape"].tolist(),
        model_sha256=hashlib.sha256(data).hexdigest(),model_bytes=len(data),
        input_tensor=int(runner.get_input_details()[0]["index"]),
        output_tensor=int(runner.get_output_details()[0]["index"]),
        input_pixel_lut=pixel_lut(runner.get_input_details()[0]).tolist(),
        layout="NHWC activations; OHWI Conv2D kernels; 1HWC depthwise kernels; OI dense",
        calibration_samples=len(calibration),tensors=[],operators=[])
    constants={}
    payload=bytearray()
    for index in range(subgraph.TensorsLength()):
        tensor=subgraph.Tensors(index)
        detail=tensor_details[index]
        q=detail["quantization_parameters"]
        entry=dict(index=index,name=detail["name"],shape=detail["shape"].tolist(),
            dtype=np.dtype(detail["dtype"]).name,scales=q["scales"].tolist(),
            zero_points=q["zero_points"].tolist(),quantized_dimension=int(q["quantized_dimension"]))
        buffer=graph.Buffers(tensor.Buffer())
        if buffer.DataLength():
            raw=buffer.DataAsNumpy().tobytes()
            payload.extend(b"\0"*((-len(payload))%4))
            entry.update(constant_offset=len(payload),constant_bytes=len(raw))
            payload.extend(raw)
            constants[f"tensor_{index}"]=np.frombuffer(raw,dtype=detail["dtype"]).reshape(detail["shape"])
        manifest["tensors"].append(entry)
    names={value:key for key,value in vars(schema.BuiltinOperator).items() if isinstance(value,int)}
    activations={value:key for key,value in vars(schema.ActivationFunctionType).items() if isinstance(value,int)}
    option_classes={"CONV_2D":schema.Conv2DOptions,"DEPTHWISE_CONV_2D":schema.DepthwiseConv2DOptions,
                    "MEAN":schema.ReducerOptions,"FULLY_CONNECTED":schema.FullyConnectedOptions}
    supported={"CONV_2D","DEPTHWISE_CONV_2D","MEAN","FULLY_CONNECTED","QUANTIZE","RESHAPE","PAD","PADV2"}
    for index in range(subgraph.OperatorsLength()):
        op=subgraph.Operators(index)
        name=names[graph.OperatorCodes(op.OpcodeIndex()).BuiltinCode()]
        if name not in supported:
            raise ValueError(f"Unexpected operator: {name}")
        entry=dict(index=index,kind=name,inputs=op.InputsAsNumpy().tolist(),outputs=op.OutputsAsNumpy().tolist())
        if name in option_classes:
            table=op.BuiltinOptions()
            options=option_classes[name]()
            options.Init(table.Bytes,table.Pos)
            for method in ("StrideH","StrideW","DilationHFactor","DilationWFactor","DepthMultiplier","KeepDims"):
                if hasattr(options,method):
                    entry[method]=getattr(options,method)()
            if hasattr(options,"Padding"):
                entry["padding"]="SAME" if options.Padding()==schema.Padding.SAME else "VALID"
            if hasattr(options,"FusedActivationFunction"):
                entry["fused_activation"]=activations[options.FusedActivationFunction()]
        manifest["operators"].append(entry)
    macs=0
    for op in manifest["operators"]:
        shape=tensor_details[op["outputs"][0]]["shape"]
        if op["kind"]=="CONV_2D":
            weights=tensor_details[op["inputs"][1]]["shape"]
            op["macs"]=int(np.prod(shape)*np.prod(weights[1:]))
        elif op["kind"]=="DEPTHWISE_CONV_2D":
            weights=tensor_details[op["inputs"][1]]["shape"]
            op["macs"]=int(np.prod(shape)*weights[1]*weights[2])
        elif op["kind"]=="FULLY_CONNECTED":
            op["macs"]=int(np.prod(tensor_details[op["inputs"][1]]["shape"]))
        macs+=op.get("macs",0)
    manifest["macs_per_image"]=macs
    manifest["max_activation_elements"]=max(int(np.prod(t["shape"])) for t in manifest["tensors"] if "constant_bytes" not in t)
    manifest["constant_bytes"]=len(payload)
    (out/"constants.bin").write_bytes(payload)
    manifest["constants_sha256"]=hashlib.sha256(payload).hexdigest()
    np.savez_compressed(out/"constants.npz",**constants)
    write_json(out/"manifest.json",manifest)
    # A layer-by-layer reference, preserving intermediate tensors in the native
    # interpreter. Includes the exact quantized input instead of assuming RGB=q.
    example=calibration[:1]
    inp=runner.get_input_details()[0]
    runner.set_tensor(inp["index"],pixel_lut(inp)[example])
    runner.invoke()
    trace={"raw_rgb":example}
    for op in manifest["operators"]:
        for index in op["outputs"]:
            trace[f"tensor_{index}"]=runner.get_tensor(index)
    scores=runner.get_tensor(manifest["output_tensor"])
    np.savez_compressed(out/"trace.npz",**trace)
    write_mem(out/"input.mem",pixel_lut(inp)[example],8)
    write_json(out/"expected.json",dict(integer_logits=scores.tolist(),class_ids=scores.argmax(axis=1).tolist(),
                                       classes=list(classes),model_sha256=manifest["model_sha256"]))
    return manifest


def load_bundle(directory):
    """Validate the sole deployment profile and both binary checksums."""
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "mobilenet025-tflite-int8-v1"
            or manifest.get("input_shape") != [1,64,64,3]
            or manifest.get("classes") != list(DEFAULT_CLASSES)):
        raise ValueError("Unsupported integer MobileNet profile")
    for filename, size_key, hash_key in (
        ("model.tflite", "model_bytes", "model_sha256"),
        ("constants.bin", "constant_bytes", "constants_sha256"),
    ):
        data = (root/filename).read_bytes()
        if len(data) != manifest[size_key] or hashlib.sha256(data).hexdigest() != manifest[hash_key]:
            raise ValueError(f"{filename} length/checksum mismatch")
    return manifest
