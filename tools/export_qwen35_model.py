#!/usr/bin/env python3
"""Export the text-only Qwen3.5 decoder as a fixed-shape Feather atomic graph.

The graph represents one token at a time.  Convolution/recurrent states and
full-attention KV caches are explicit graph values so the C++ runner owns all
sequence state.  Model-specific modules are expanded into ordinary ONNX
operators; no Qwen-specific operator is emitted.
"""

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "llm" / "qwen3.5-0.8b"

FTH_MAGIC = b"FTHMODL\x00"
FTH_FORMAT_VERSION = 2
FTH_WEIGHT_ALIGNMENT = 64
FTH_LAYOUT_ND = 2
FTH_DTYPE_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 9: 1, 11: 2, 12: 1, 13: 1}
FTH_INT8 = 1
FTH_UINT8 = 2
FTH_FP16 = 3
FTH_FP32 = 4
FTH_INT32 = 5
FTH_INT64 = 6
FTH_BOOL = 9
FTH_BF16 = 11
FTH_FP8E4M3 = 12
FTH_FP8E5M2 = 13

BF16 = FTH_BF16
FP32 = FTH_FP32
INT64 = FTH_INT64
FP8E4M3 = FTH_FP8E4M3
FP8E5M2 = FTH_FP8E5M2

FTH_TO_ONNX_DTYPE = {
    FTH_INT8: 3,
    FTH_UINT8: 2,
    FTH_FP16: 10,
    FTH_FP32: 1,
    FTH_INT32: 6,
    FTH_INT64: 7,
    FTH_BOOL: 9,
    FTH_BF16: 16,
    FTH_FP8E4M3: 17,
    FTH_FP8E5M2: 19,
}

FP8_MAX_FINITE = {
    FP8E4M3: 448.0,
    FP8E5M2: 57344.0,
}

FP8_NUMPY_DTYPE = {
    FP8E4M3: ml_dtypes.float8_e4m3fn,
    FP8E5M2: ml_dtypes.float8_e5m2,
}

FP8_FORMAT_NAMES = {
    FP8E4M3: "fp8e4m3",
    FP8E5M2: "fp8e5m2",
}


@dataclass
class FeatherTensorDesc:
    name: str
    dims: list
    data_type: int
    layout: int = FTH_LAYOUT_ND
    quantization_enabled: bool = False
    quantization_scale: float = 1.0
    quantization_granularity: int = 0
    quantization_axis: int = -1
    quantization_block_size: int = 0


@dataclass
class FeatherValueDesc:
    tensor: FeatherTensorDesc
    constant: bool = False
    weight_offset: int = 0
    weight_size: int = 0
    checksum: str = ""


@dataclass
class FeatherNodeDesc:
    name: str
    op_type: str
    inputs: list
    outputs: list
    attributes: dict


@dataclass
class FeatherWeightPayload:
    byte_size: int
    write: object


def is_fp8_dtype(dtype):
    return dtype in FP8_MAX_FINITE


def fp8_format_name(dtype):
    try:
        return FP8_FORMAT_NAMES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported FP8 dtype {dtype}") from error


def fp8_scale_for_amax(dtype, amax):
    if dtype not in FP8_MAX_FINITE:
        raise ValueError(f"unsupported FP8 dtype {dtype}")
    if not math.isfinite(float(amax)):
        raise ValueError("FP8 scale requires a finite maximum")
    if amax <= 0.0:
        return 1.0
    scale = 2.0 ** math.ceil(math.log2(float(amax) / FP8_MAX_FINITE[dtype]))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("FP8 scale is outside the finite range")
    return float(scale)


def fp8_scale_for_array(dtype, array):
    """Return a deterministic FP8 scale without materializing a full FP32 copy."""
    values = np.asarray(array)
    flat = values.reshape(-1)
    maximum = 0.0
    chunk_size = 1 << 20
    for start in range(0, flat.size, chunk_size):
        chunk = np.asarray(flat[start:start + chunk_size], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("FP8 model weights must be finite")
        if chunk.size:
            maximum = max(maximum, float(np.max(np.abs(chunk))))
    return fp8_scale_for_amax(dtype, maximum)


def fp8_encode(array, dtype, scale):
    if dtype not in FP8_NUMPY_DTYPE:
        raise ValueError(f"unsupported FP8 dtype {dtype}")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("FP8 scale must be finite and positive")
    values = np.asarray(array)
    encoded = np.empty(values.shape, dtype=FP8_NUMPY_DTYPE[dtype])
    flat_values = values.reshape(-1)
    flat_encoded = encoded.reshape(-1)
    chunk_size = 1 << 20
    for start in range(0, flat_values.size, chunk_size):
        chunk = np.asarray(flat_values[start:start + chunk_size], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("FP8 model weights must be finite")
        flat_encoded[start:start + chunk.size] = (chunk / float(scale)).astype(FP8_NUMPY_DTYPE[dtype])
    return np.ascontiguousarray(encoded.view(np.uint8))


def fth_align_up(value, alignment=FTH_WEIGHT_ALIGNMENT):
    return ((value + alignment - 1) // alignment) * alignment


def fth_encode_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def fth_encode_ints(values):
    return struct.pack("<Q", len(values)) + b"".join(struct.pack("<q", int(value)) for value in values)


def fth_encode_floats(values):
    return struct.pack("<Q", len(values)) + b"".join(struct.pack("<f", float(value)) for value in values)


def fth_encode_attribute(value):
    if isinstance(value, (bool, int, np.integer)):
        return struct.pack("<I", 0) + struct.pack("<q", int(value))
    if isinstance(value, (float, np.floating)):
        return struct.pack("<I", 1) + struct.pack("<f", float(value))
    if isinstance(value, str):
        return struct.pack("<I", 2) + fth_encode_string(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty Feather attribute vectors are not supported")
        if all(isinstance(item, (bool, int, np.integer)) for item in value):
            return struct.pack("<I", 3) + fth_encode_ints(value)
        if all(isinstance(item, (int, float, np.integer, np.floating)) for item in value):
            return struct.pack("<I", 4) + fth_encode_floats(value)
    raise TypeError(f"unsupported Feather attribute type: {type(value)!r}")


def fth_encode_tensor(desc):
    payload = (fth_encode_string(desc.name) + fth_encode_ints(desc.dims) +
               struct.pack("<i", desc.data_type) + struct.pack("<i", desc.layout))
    if FTH_FORMAT_VERSION >= 2:
        payload += (struct.pack("<B", 1 if desc.quantization_enabled else 0) +
                    struct.pack("<f", float(desc.quantization_scale)) +
                    struct.pack("<i", int(desc.quantization_granularity)) +
                    struct.pack("<q", int(desc.quantization_axis)) +
                    struct.pack("<q", int(desc.quantization_block_size)))
    return payload


def fth_encode_value(value, output_path):
    return (fth_encode_tensor(value.tensor) + struct.pack("<B", 1 if value.constant else 0) +
            fth_encode_string(value.tensor.name) + fth_encode_string(str(output_path)) +
            struct.pack("<Q", value.weight_offset) + struct.pack("<Q", value.weight_size) +
            fth_encode_string(value.checksum))


def fth_encode_names(values):
    return struct.pack("<Q", len(values)) + b"".join(fth_encode_string(value) for value in values)


def fth_encode_node(node):
    payload = (fth_encode_string(node.name) + fth_encode_string(node.op_type) + fth_encode_string("") +
               fth_encode_names(node.inputs) + fth_encode_names(node.outputs) +
               struct.pack("<Q", len(node.attributes)))
    for key, value in node.attributes.items():
        payload += fth_encode_string(key) + fth_encode_attribute(value)
    return payload


def fth_encode_model(name, graph_name, inputs, outputs, values, nodes, output_path):
    graph = (fth_encode_string(graph_name) + fth_encode_names(inputs) + fth_encode_names(outputs) +
             struct.pack("<Q", len(values)) +
             b"".join(fth_encode_value(value, output_path) for value in values) +
             struct.pack("<Q", len(nodes)) + b"".join(fth_encode_node(node) for node in nodes))
    return fth_encode_string(name) + struct.pack("<q", 1) + graph


def write_feather_model(path, name, graph_name, inputs, outputs, values, nodes, payloads):
    for _ in range(4):
        metadata = fth_encode_model(name, graph_name, inputs, outputs, values, nodes, path)
        offset = fth_align_up(24 + len(metadata))
        for value in values:
            if not value.constant:
                continue
            payload = payloads[value.tensor.name]
            value.weight_offset = offset
            value.weight_size = payload.byte_size
            value.checksum = ""
            offset = fth_align_up(offset + payload.byte_size)

    metadata = fth_encode_model(name, graph_name, inputs, outputs, values, nodes, path)
    with open(path, "wb") as output:
        output.write(struct.pack("<8sIIQ", FTH_MAGIC, FTH_FORMAT_VERSION, 0, len(metadata)))
        output.write(metadata)
        for value in values:
            if not value.constant:
                continue
            padding = value.weight_offset - output.tell()
            if padding < 0:
                raise RuntimeError(f"invalid FTH weight offset for {value.tensor.name}")
            output.write(b"\x00" * padding)
            start = output.tell()
            payloads[value.tensor.name].write(output)
            if output.tell() - start != value.weight_size:
                raise RuntimeError(f"incorrect FTH weight size for {value.tensor.name}")


def source_to_numpy(tensor):
    """Convert a safetensors torch tensor without losing BF16 payload bits."""
    import torch

    if tensor.dtype == torch.bfloat16:
        bits = tensor.detach().cpu().view(torch.uint16).numpy()
        return np.ascontiguousarray(bits.view(ml_dtypes.bfloat16).reshape(tuple(tensor.shape)))
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def normalize_constant(array):
    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    return np.ascontiguousarray(array)


def dtype_of(array):
    dtype = np.asarray(array).dtype
    if str(dtype) == "bfloat16":
        return FTH_BF16
    if dtype == np.dtype(np.int8):
        return FTH_INT8
    if dtype == np.dtype(np.uint8):
        return FTH_UINT8
    if dtype == np.dtype(np.float16):
        return FTH_FP16
    if dtype == np.dtype(np.float32):
        return FTH_FP32
    if dtype == np.dtype(np.int32):
        return FTH_INT32
    if dtype == np.dtype(np.int64):
        return FTH_INT64
    if dtype == np.dtype(np.bool_):
        return FTH_BOOL
    raise ValueError(f"unsupported Feather tensor dtype: {dtype}")


def tensor_proto_from_array(array, name, data_type, tensor_proto, numpy_helper):
    array = normalize_constant(array)
    if data_type not in {FTH_BF16, FTH_FP8E4M3, FTH_FP8E5M2}:
        return numpy_helper.from_array(array, name=name)
    tensor = tensor_proto()
    tensor.name = name
    tensor.data_type = FTH_TO_ONNX_DTYPE[data_type]
    tensor.dims.extend(array.shape)
    tensor.raw_data = raw_array_view(array).tobytes()
    return tensor


def raw_array_view(array):
    array = np.ascontiguousarray(array)
    if str(array.dtype) == "bfloat16":
        array = array.view(np.uint16)
    return memoryview(array).cast("B")


def feather_attributes(op_type, onnx_attributes):
    """Map ONNX spelling to the attribute names consumed by Feather operators."""
    attributes = dict(onnx_attributes)
    if op_type == "Split" and "split" in attributes:
        attributes["split_sizes"] = attributes.pop("split")
    return attributes


class AtomicGraph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.value_info = {}
        self.constants = set()
        self.constant_names = []
        self.constant_values = {}
        self.external_constant_loaders = {}
        self.quantization_scales = {}
        self.counter = 0

    def fresh(self, stem):
        self.counter += 1
        return f"{stem}_{self.counter}"

    def register(self, name, shape, dtype, quantization_scale=None):
        dtype = int(dtype)
        if is_fp8_dtype(dtype):
            if quantization_scale is None:
                quantization_scale = self.quantization_scales.get(name)
            if quantization_scale is None or not math.isfinite(float(quantization_scale)) or quantization_scale <= 0.0:
                raise ValueError(f"FP8 value {name} requires a finite positive per-tensor scale")
            self.quantization_scales[name] = float(quantization_scale)
        else:
            if quantization_scale is not None:
                raise ValueError(f"non-FP8 value {name} must not carry FP8 quantization metadata")
            self.quantization_scales.pop(name, None)
        self.value_info[name] = (list(shape), dtype)
        return name

    def input(self, name, shape, dtype, quantization_scale=None):
        self.register(name, shape, dtype, quantization_scale)
        self.inputs.append(name)
        return name

    def output(self, name, shape=None, dtype=None, quantization_scale=None):
        if shape is not None:
            self.register(name, shape, dtype, quantization_scale)
        self.outputs.append(name)
        return name

    def const(self, name, array):
        if name in self.constants:
            return name
        array = normalize_constant(array)
        self.constants.add(name)
        self.constant_names.append(name)
        self.constant_values[name] = array
        self.register(name, list(array.shape), dtype_of(array))
        return name

    def external_const(self, name, shape, dtype, loader, quantization_scale=None):
        if name in self.constants:
            return name
        self.constants.add(name)
        self.constant_names.append(name)
        self.external_constant_loaders[name] = loader
        self.register(name, shape, dtype, quantization_scale)
        return name

    def scalar(self, name, value, dtype=np.float32):
        return self.const(name, np.asarray([value], dtype=dtype))

    def ints(self, name, values):
        return self.const(name, np.asarray(values, dtype=np.int64))

    def node(self, op_type, inputs, outputs, shapes, dtypes, name=None, quantization_scales=None, **attrs):
        if isinstance(outputs, str):
            outputs = [outputs]
        if isinstance(shapes, tuple) and shapes and isinstance(shapes[0], int):
            shapes = [shapes]
        if isinstance(dtypes, int):
            dtypes = [dtypes] * len(outputs)
        if quantization_scales is None:
            quantization_scales = [None] * len(outputs)
        elif isinstance(quantization_scales, (float, int, np.floating, np.integer)):
            quantization_scales = [quantization_scales] * len(outputs)
        if len(outputs) != len(shapes) or len(outputs) != len(dtypes) or len(outputs) != len(quantization_scales):
            raise ValueError(f"invalid output metadata for {op_type}")
        if name is None:
            name = self.fresh(op_type.lower())
        for output, shape, dtype, quantization_scale in zip(outputs, shapes, dtypes, quantization_scales):
            self.register(output, shape, dtype, quantization_scale)
        self.nodes.append(FeatherNodeDesc(
            name=name,
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            attributes=dict(attrs),
        ))
        return outputs[0] if len(outputs) == 1 else outputs

    def value_quantization_scale(self, name):
        if not is_fp8_dtype(self.value_info[name][1]):
            return None
        return self.quantization_scales[name]

    def inherited_fp8_scale(self, values, dtype):
        if not is_fp8_dtype(dtype):
            return None
        scales = [self.value_quantization_scale(value) for value in values]
        if not scales or any(scale is None for scale in scales):
            raise ValueError("FP8 operation inputs require per-tensor scales")
        first = scales[0]
        if any(not math.isclose(scale, first, rel_tol=0.0, abs_tol=0.0) for scale in scales[1:]):
            raise ValueError("FP8 view operations require equal input scales")
        return first

    def unary(self, op_type, x, shape, dtype, stem):
        return self.node(op_type, [x], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.inherited_fp8_scale([x], dtype))

    def binary(self, op_type, lhs, rhs, shape, dtype, stem):
        return self.node(op_type, [lhs, rhs], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.inherited_fp8_scale([lhs, rhs], dtype))

    def cast(self, x, shape, dtype, stem, quantization_scale=None):
        source_dtype = self.value_info[x][1]
        if source_dtype == dtype:
            if is_fp8_dtype(dtype) and quantization_scale is not None and \
                    self.value_quantization_scale(x) != float(quantization_scale):
                raise ValueError("a no-op FP8 Cast cannot change its quantization scale")
            return x
        return self.node("Cast", [x], self.fresh(stem), [shape], dtype,
                         quantization_scales=quantization_scale, to=FTH_TO_ONNX_DTYPE[dtype])

    def reshape(self, x, shape, stem):
        shape_name = self.ints(self.fresh(f"{stem}_shape"), shape)
        dtype = self.value_info[x][1]
        return self.node("Reshape", [x, shape_name], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.value_quantization_scale(x))

    def transpose(self, x, perm, shape, stem):
        dtype = self.value_info[x][1]
        return self.node("Transpose", [x], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.value_quantization_scale(x), perm=perm)

    def unsqueeze(self, x, axes, shape, stem):
        dtype = self.value_info[x][1]
        return self.node("Unsqueeze", [x], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.value_quantization_scale(x), axes=axes)

    def expand(self, x, shape, stem):
        shape_name = self.ints(self.fresh(f"{stem}_shape"), shape)
        dtype = self.value_info[x][1]
        return self.node("Expand", [x, shape_name], self.fresh(stem), [shape], dtype,
                         quantization_scales=self.value_quantization_scale(x))

    def concat(self, inputs, axis, shape, dtype, stem):
        return self.node("Concat", inputs, self.fresh(stem), [shape], dtype,
                         quantization_scales=self.inherited_fp8_scale(inputs, dtype), axis=axis)

    def split(self, x, sizes, axis, shapes, stem):
        names = [self.fresh(f"{stem}_{i}") for i in range(len(sizes))]
        dtype = self.value_info[x][1]
        return self.node("Split", [x], names, shapes, dtype,
                         quantization_scales=self.value_quantization_scale(x), axis=axis, split=sizes)

    def reduce(self, op_type, x, axes, keepdims, shape, dtype, stem):
        return self.node(op_type, [x], self.fresh(stem), [shape], dtype, axes=axes, keepdims=int(keepdims))

    def make_model(self, max_context):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        initializers = []
        for name in self.constant_names:
            if name in self.constant_values:
                array = self.constant_values[name]
            else:
                array = self.external_constant_loaders[name]()
            initializers.append(tensor_proto_from_array(array, name, self.value_info[name][1], TensorProto, numpy_helper))
        input_infos = []
        for name in self.inputs:
            shape, dtype = self.value_info[name]
            input_infos.append(helper.make_tensor_value_info(name, FTH_TO_ONNX_DTYPE[dtype], shape))
        output_infos = []
        for name in self.outputs:
            shape, dtype = self.value_info[name]
            output_infos.append(helper.make_tensor_value_info(name, FTH_TO_ONNX_DTYPE[dtype], shape))
        value_infos = []
        for name, (shape, dtype) in self.value_info.items():
            if name in self.inputs or name in self.outputs or name in self.constants:
                continue
            value_infos.append(helper.make_tensor_value_info(name, FTH_TO_ONNX_DTYPE[dtype], shape))
        onnx_nodes = [
            helper.make_node(node.op_type, node.inputs, node.outputs, name=node.name, **node.attributes)
            for node in self.nodes
        ]
        graph = helper.make_graph(
            onnx_nodes,
            "qwen35_decode_atomic",
            input_infos,
            output_infos,
            initializer=initializers,
            value_info=value_infos,
        )
        model = helper.make_model(
            graph,
            producer_name="feather-qwen35-exporter",
            opset_imports=[helper.make_opsetid("", 19 if any(is_fp8_dtype(dtype) for _, dtype in self.value_info.values()) else 11)],
        )
        model.ir_version = 7
        model.doc_string = (
            "Fixed batch=1 single-token Qwen3.5 text decoder; "
            f"explicit recurrent state and max_context={max_context}."
        )
        return model

    def write_fth(self, path, model_name):
        values = []
        for name, (shape, dtype) in self.value_info.items():
            quantization_scale = self.quantization_scales.get(name)
            values.append(FeatherValueDesc(
                tensor=FeatherTensorDesc(name=name, dims=[int(value) for value in shape],
                                         data_type=int(dtype),
                                         quantization_enabled=is_fp8_dtype(dtype),
                                         quantization_scale=1.0 if quantization_scale is None else quantization_scale),
                constant=name in self.constants,
            ))

        nodes = []
        for node in self.nodes:
            nodes.append(FeatherNodeDesc(
                name=node.name,
                op_type=node.op_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                attributes=feather_attributes(node.op_type, node.attributes),
            ))

        payloads = {}
        for name in self.constant_names:
            if name in self.constant_values:
                raw = raw_array_view(self.constant_values[name])
                payloads[name] = FeatherWeightPayload(
                    byte_size=len(raw),
                    write=lambda output, raw=raw: output.write(raw),
                )
                continue

            loader = self.external_constant_loaders[name]
            shape, dtype = self.value_info[name]
            byte_size = int(np.prod(shape, dtype=np.int64)) * FTH_DTYPE_BYTES[int(dtype)]

            def write_external(output, loader=loader, expected_size=byte_size, constant_name=name):
                array = loader()
                raw = raw_array_view(array)
                if len(raw) != expected_size:
                    raise RuntimeError(f"unexpected safetensors payload size for {constant_name}")
                output.write(raw)

            payloads[name] = FeatherWeightPayload(byte_size=byte_size, write=write_external)

        write_feather_model(path, model_name, "qwen35_decode_atomic", self.inputs, self.outputs, values, nodes, payloads)


class QwenBuilder:
    def __init__(self, model_dir, max_context, fp8_dtype=None, fp8_activation_amax=96.0):
        if fp8_dtype is not None and not is_fp8_dtype(fp8_dtype):
            raise ValueError(f"unsupported Qwen FP8 format {fp8_dtype}")
        if fp8_dtype is not None and (
                not math.isfinite(float(fp8_activation_amax)) or fp8_activation_amax <= 0.0):
            raise ValueError("FP8 activation maximum must be finite and positive")
        self.model_dir = Path(model_dir)
        self.max_context = max_context
        self.fp8_dtype = fp8_dtype
        self.fp8_activation_scale = (
            fp8_scale_for_amax(fp8_dtype, fp8_activation_amax) if fp8_dtype is not None else None
        )
        self.fp8_weight_scales = {}
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.text = self.config["text_config"]
        self.hidden = int(self.text["hidden_size"])
        self.intermediate = int(self.text["intermediate_size"])
        self.layers = int(self.text["num_hidden_layers"])
        self.vocab = int(self.text["vocab_size"])
        self.head_dim = int(self.text["head_dim"])
        self.full_heads = int(self.text["num_attention_heads"])
        self.kv_heads = int(self.text["num_key_value_heads"])
        self.linear_heads = int(self.text["linear_num_value_heads"])
        self.linear_key_dim = int(self.text["linear_key_head_dim"])
        self.linear_value_dim = int(self.text["linear_value_head_dim"])
        self.conv_dim = self.linear_heads * self.linear_value_dim + 2 * self.text["linear_num_key_heads"] * self.text["linear_key_head_dim"]
        self.rope_dim = int(self.head_dim * self.text["rope_parameters"].get("partial_rotary_factor", 1.0))
        self.rope_theta = float(self.text["rope_parameters"]["rope_theta"])
        self.eps = float(self.text["rms_norm_eps"])
        self.g = AtomicGraph()
        self.source = None
        self.cast_cache = {}

    def weight(self, key, transpose=False, force_dtype=None):
        source_slice = self.source.get_slice(key)
        shape = list(source_slice.get_shape())
        source_dtype = str(source_slice.get_dtype())
        if source_dtype == "BF16":
            dtype = BF16
        elif source_dtype in {"F32", "FLOAT32"}:
            dtype = FP32
        else:
            raise ValueError(f"unsupported safetensors dtype for {key}: {source_dtype}")
        if transpose:
            if len(shape) != 2:
                raise ValueError(f"only rank-2 projection weights may be transposed: {key}")
            shape.reverse()
        if force_dtype is not None:
            dtype = force_dtype
        suffix = "_t" if transpose else ""
        quantization_scale = None
        if is_fp8_dtype(dtype):
            suffix += f"__{fp8_format_name(dtype)}"
            quantization_scale = self.fp8_weight_scale(key, dtype)
        name = "weight_" + key.replace(".", "_") + suffix
        if name in self.g.constants:
            return name
        return self.g.external_const(
            name,
            shape,
            dtype,
            lambda key=key, transpose=transpose, force_dtype=force_dtype: self.load_weight(
                key, transpose=transpose, force_dtype=force_dtype),
            quantization_scale=quantization_scale,
        )

    def fp8_weight_scale(self, key, dtype):
        cache_key = (key, dtype)
        if cache_key not in self.fp8_weight_scales:
            if self.source is None:
                raise RuntimeError("Qwen weights require an open safetensors source")
            self.fp8_weight_scales[cache_key] = fp8_scale_for_array(
                dtype, source_to_numpy(self.source.get_tensor(key))
            )
        return self.fp8_weight_scales[cache_key]

    def load_weight(self, key, transpose=False, force_dtype=None):
        array = source_to_numpy(self.source.get_tensor(key))
        if is_fp8_dtype(force_dtype):
            encoded = fp8_encode(array, force_dtype, self.fp8_weight_scale(key, force_dtype))
            return np.ascontiguousarray(encoded.T) if transpose else encoded
        if transpose:
            array = np.ascontiguousarray(array.T)
        if force_dtype is None or dtype_of(array) == force_dtype:
            return array
        if force_dtype == FP32:
            return np.asarray(array, dtype=np.float32)
        if force_dtype == BF16:
            return np.asarray(np.asarray(array, dtype=np.float32), dtype=ml_dtypes.bfloat16)
        raise ValueError(f"unsupported forced dtype {force_dtype}")

    def cast(self, x, dtype, stem, quantization_scale=None):
        shape = self.g.value_info[x][0]
        if is_fp8_dtype(dtype):
            if quantization_scale is None:
                quantization_scale = self.fp8_activation_scale
            if quantization_scale is None:
                raise ValueError("FP8 casts require an explicit activation scale")
            quantization_scale = float(quantization_scale)
        elif quantization_scale is not None:
            raise ValueError("non-FP8 casts cannot carry quantization metadata")
        key = (x, dtype, quantization_scale)
        if key not in self.cast_cache:
            self.cast_cache[key] = self.g.cast(x, shape, dtype, stem, quantization_scale=quantization_scale)
        return self.cast_cache[key]

    def rms_norm(self, x, weight_key, stem):
        shape = self.g.value_info[x][0]
        x32 = self.cast(x, FP32, f"{stem}_to_fp32")
        square = self.g.binary("Mul", x32, x32, shape, FP32, f"{stem}_square")
        mean_shape = list(shape)
        mean_shape[-1] = 1
        mean = self.g.reduce("ReduceMean", square, [-1], True, mean_shape, FP32, f"{stem}_mean")
        eps = self.g.scalar("qwen_rms_eps", self.eps)
        mean_eps = self.g.binary("Add", mean, eps, mean_shape, FP32, f"{stem}_eps")
        root = self.g.unary("Sqrt", mean_eps, mean_shape, FP32, f"{stem}_sqrt")
        normalized = self.g.binary("Div", x32, root, shape, FP32, f"{stem}_normalize")
        weight = self.weight(weight_key)
        weight32 = self.cast(weight, FP32, f"{stem}_weight_to_fp32")
        one = self.g.scalar("qwen_rms_one", 1.0)
        scale = self.g.binary("Add", weight32, one, self.g.value_info[weight32][0], FP32, f"{stem}_scale")
        scaled = self.g.binary("Mul", normalized, scale, shape, FP32, f"{stem}_scale_value")
        return self.cast(scaled, BF16, f"{stem}_to_bf16")

    def l2_norm(self, x, stem):
        shape = self.g.value_info[x][0]
        x32 = self.cast(x, FP32, f"{stem}_to_fp32")
        square = self.g.binary("Mul", x32, x32, shape, FP32, f"{stem}_square")
        sum_shape = list(shape)
        sum_shape[-1] = 1
        total = self.g.reduce("ReduceSum", square, [-1], True, sum_shape, FP32, f"{stem}_sum")
        eps = self.g.scalar("qwen_l2_eps", 1e-6)
        total = self.g.binary("Add", total, eps, sum_shape, FP32, f"{stem}_eps")
        root = self.g.unary("Sqrt", total, sum_shape, FP32, f"{stem}_sqrt")
        return self.g.binary("Div", x32, root, shape, FP32, f"{stem}_normalize")

    def projection(self, x, key, out_size, stem):
        x_shape = self.g.value_info[x][0]
        out_shape = list(x_shape)
        out_shape[-1] = out_size
        if self.fp8_dtype is None:
            w = self.weight(key, transpose=True)
            return self.g.node("MatMul", [x, w], self.g.fresh(stem), [out_shape], BF16)
        fp8_input = self.cast(x, self.fp8_dtype, f"{stem}_to_fp8", self.fp8_activation_scale)
        fp8_weight = self.weight(key, transpose=True, force_dtype=self.fp8_dtype)
        fp8_output = self.g.node(
            "MatMul",
            [fp8_input, fp8_weight],
            self.g.fresh(stem),
            [out_shape],
            self.fp8_dtype,
            quantization_scales=self.fp8_activation_scale,
        )
        return self.cast(fp8_output, BF16, f"{stem}_to_bf16")

    def silu(self, x, stem):
        shape = self.g.value_info[x][0]
        sig = self.g.unary("Sigmoid", x, shape, self.g.value_info[x][1], f"{stem}_sigmoid")
        return self.g.binary("Mul", x, sig, shape, self.g.value_info[x][1], f"{stem}_mul")

    def build_linear_layer(self, layer, hidden):
        stem = f"layer_{layer}_linear"
        residual = hidden
        hidden = self.rms_norm(hidden, f"model.language_model.layers.{layer}.input_layernorm.weight", f"{stem}_input_norm")

        mixed = self.projection(hidden, f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight", self.conv_dim, f"{stem}_qkv_proj")
        mixed_t = self.g.transpose(mixed, [0, 2, 1], [1, self.conv_dim, 1], f"{stem}_qkv_transpose")
        state = f"conv_state_{layer}"
        joined = self.g.concat([state, mixed_t], 2, [1, self.conv_dim, 4], BF16, f"{stem}_state_join")
        joined_2d = self.g.reshape(joined, [1, self.conv_dim, 1, 4], f"{stem}_conv_input")
        conv_weight = self.g.reshape(
            self.weight(
                f"model.language_model.layers.{layer}.linear_attn.conv1d.weight",
                force_dtype=self.fp8_dtype,
            ) if self.fp8_dtype is not None else self.weight(
                f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
            ),
            [self.conv_dim, 1, 1, 4],
            f"{stem}_conv_weight",
        )
        conv_input = joined_2d
        conv_dtype = BF16
        conv_quantization_scale = None
        if self.fp8_dtype is not None:
            conv_input = self.cast(joined_2d, self.fp8_dtype, f"{stem}_conv_input_to_fp8",
                                   self.fp8_activation_scale)
            conv_dtype = self.fp8_dtype
            conv_quantization_scale = self.fp8_activation_scale
        conv = self.g.node(
            "Conv",
            [conv_input, conv_weight],
            self.g.fresh(f"{stem}_conv"),
            [[1, self.conv_dim, 1, 1]],
            conv_dtype,
            quantization_scales=conv_quantization_scale,
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=self.conv_dim,
        )
        if self.fp8_dtype is not None:
            conv = self.cast(conv, BF16, f"{stem}_conv_to_bf16")
        conv = self.g.reshape(conv, [1, self.conv_dim, 1], f"{stem}_conv_squeeze")
        conv = self.silu(conv, f"{stem}_conv_activation")
        _, next_conv_state = self.g.split(joined, [1, 3], 2, [[1, self.conv_dim, 1], [1, self.conv_dim, 3]], f"{stem}_state_split")

        qkv = self.g.transpose(conv, [0, 2, 1], [1, 1, self.conv_dim], f"{stem}_qkv_restore")
        q_flat, k_flat, v_flat = self.g.split(
            qkv,
            [self.linear_heads * self.linear_key_dim, self.linear_heads * self.linear_key_dim, self.linear_heads * self.linear_value_dim],
            2,
            [[1, 1, self.linear_heads * self.linear_key_dim], [1, 1, self.linear_heads * self.linear_key_dim], [1, 1, self.linear_heads * self.linear_value_dim]],
            f"{stem}_qkv_split",
        )
        q = self.g.reshape(q_flat, [1, self.linear_heads, self.linear_key_dim], f"{stem}_q_reshape")
        k = self.g.reshape(k_flat, [1, self.linear_heads, self.linear_key_dim], f"{stem}_k_reshape")
        v = self.g.reshape(v_flat, [1, self.linear_heads, self.linear_value_dim], f"{stem}_v_reshape")
        q = self.l2_norm(q, f"{stem}_q_l2")
        k = self.l2_norm(k, f"{stem}_k_l2")

        z = self.projection(hidden, f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight", self.linear_heads * self.linear_value_dim, f"{stem}_z_proj")
        z = self.g.reshape(z, [1, self.linear_heads, self.linear_value_dim], f"{stem}_z_reshape")
        b = self.projection(hidden, f"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight", self.linear_heads, f"{stem}_b_proj")
        b = self.g.reshape(b, [1, self.linear_heads], f"{stem}_b_reshape")
        a = self.projection(hidden, f"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight", self.linear_heads, f"{stem}_a_proj")
        a = self.g.reshape(a, [1, self.linear_heads], f"{stem}_a_reshape")
        q32 = self.g.binary(
            "Mul", q, self.g.scalar("qwen_linear_query_scale", 1.0 / (self.linear_key_dim ** 0.5)),
            [1, self.linear_heads, self.linear_key_dim], FP32, f"{stem}_q_scale"
        )
        k32 = k
        v32 = self.cast(v, FP32, f"{stem}_v_to_fp32")
        b32 = self.cast(b, FP32, f"{stem}_b_to_fp32")
        a32 = self.cast(a, FP32, f"{stem}_a_to_fp32")
        beta = self.g.unary("Sigmoid", b32, [1, self.linear_heads], FP32, f"{stem}_beta")
        beta = self.g.unsqueeze(beta, [2], [1, self.linear_heads, 1], f"{stem}_beta_unsqueeze")
        dt_bias = self.cast(self.weight(f"model.language_model.layers.{layer}.linear_attn.dt_bias"), FP32, f"{stem}_dt_bias")
        a_plus = self.g.binary("Add", a32, dt_bias, [1, self.linear_heads], FP32, f"{stem}_a_dt")
        softplus = self.g.unary("Softplus", a_plus, [1, self.linear_heads], FP32, f"{stem}_softplus")
        a_log = self.cast(self.weight(f"model.language_model.layers.{layer}.linear_attn.A_log"), FP32, f"{stem}_a_log")
        a_exp = self.g.unary("Exp", a_log, [self.linear_heads], FP32, f"{stem}_a_log_exp")
        neg_a_exp = self.g.unary("Neg", a_exp, [self.linear_heads], FP32, f"{stem}_neg_a")
        g = self.g.binary("Mul", softplus, neg_a_exp, [1, self.linear_heads], FP32, f"{stem}_decay")

        state = f"recurrent_state_{layer}"
        g_exp = self.g.unary("Exp", g, [1, self.linear_heads], FP32, f"{stem}_decay_exp")
        g_exp = self.g.unsqueeze(g_exp, [2], [1, self.linear_heads, 1], f"{stem}_decay_unsqueeze")
        g_exp = self.g.unsqueeze(g_exp, [3], [1, self.linear_heads, 1, 1], f"{stem}_decay_unsqueeze_last")
        state_decay = self.g.binary("Mul", state, g_exp, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32, f"{stem}_state_decay")
        k_col = self.g.unsqueeze(k32, [3], [1, self.linear_heads, self.linear_key_dim, 1], f"{stem}_k_col")
        kv_mem_full = self.g.binary("Mul", state_decay, k_col, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32, f"{stem}_kv_mem_mul")
        kv_mem = self.g.reduce("ReduceSum", kv_mem_full, [2], False, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_kv_mem")
        v_delta = self.g.binary("Sub", v32, kv_mem, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_delta_value")
        delta = self.g.binary("Mul", v_delta, beta, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_delta")
        delta_row = self.g.unsqueeze(delta, [2], [1, self.linear_heads, 1, self.linear_value_dim], f"{stem}_delta_row")
        update = self.g.binary("Mul", k_col, delta_row, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32, f"{stem}_state_update_mul")
        next_state = self.g.binary("Add", state_decay, update, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32, f"{stem}_state_update")
        q_col = self.g.unsqueeze(q32, [3], [1, self.linear_heads, self.linear_key_dim, 1], f"{stem}_q_col")
        output_full = self.g.binary("Mul", next_state, q_col, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32, f"{stem}_output_mul")
        core = self.g.reduce("ReduceSum", output_full, [2], False, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_core")
        core_bf16 = self.cast(core, BF16, f"{stem}_core_to_bf16")
        norm_weight = self.weight(f"model.language_model.layers.{layer}.linear_attn.norm.weight")
        core32 = self.cast(core_bf16, FP32, f"{stem}_norm_input")
        variance = self.g.binary("Mul", core32, core32, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_norm_square")
        variance = self.g.reduce("ReduceMean", variance, [-1], True, [1, self.linear_heads, 1], FP32, f"{stem}_norm_mean")
        variance = self.g.binary("Add", variance, self.g.scalar("qwen_rms_eps_gated", self.eps), [1, self.linear_heads, 1], FP32, f"{stem}_norm_eps")
        variance = self.g.unary("Sqrt", variance, [1, self.linear_heads, 1], FP32, f"{stem}_norm_sqrt")
        normalized = self.g.binary("Div", core32, variance, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_norm_div")
        normalized = self.g.binary("Mul", normalized, norm_weight, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_norm_weight")
        z32 = self.cast(z, FP32, f"{stem}_gate_to_fp32")
        z_silu = self.silu(z32, f"{stem}_gate_silu")
        gated = self.g.binary("Mul", normalized, z_silu, [1, self.linear_heads, self.linear_value_dim], FP32, f"{stem}_gated")
        gated = self.cast(gated, BF16, f"{stem}_gated_to_bf16")
        gated = self.g.reshape(gated, [1, 1, self.linear_heads * self.linear_value_dim], f"{stem}_gated_flat")
        mixed_out = self.projection(gated, f"model.language_model.layers.{layer}.linear_attn.out_proj.weight", self.hidden, f"{stem}_out_proj")
        hidden = self.g.binary("Add", residual, mixed_out, [1, 1, self.hidden], BF16, f"{stem}_residual")

        residual = hidden
        hidden = self.rms_norm(hidden, f"model.language_model.layers.{layer}.post_attention_layernorm.weight", f"{stem}_post_norm")
        gate = self.projection(hidden, f"model.language_model.layers.{layer}.mlp.gate_proj.weight", self.intermediate, f"{stem}_mlp_gate")
        up = self.projection(hidden, f"model.language_model.layers.{layer}.mlp.up_proj.weight", self.intermediate, f"{stem}_mlp_up")
        gate = self.silu(gate, f"{stem}_mlp_silu")
        product = self.g.binary("Mul", gate, up, [1, 1, self.intermediate], BF16, f"{stem}_mlp_product")
        down = self.projection(product, f"model.language_model.layers.{layer}.mlp.down_proj.weight", self.hidden, f"{stem}_mlp_down")
        hidden = self.g.binary("Add", residual, down, [1, 1, self.hidden], BF16, f"{stem}_mlp_residual")
        return hidden, next_conv_state, next_state

    def rope(self, x, position, stem):
        # Text-only Qwen3.5 uses identical temporal/height/width positions, so
        # interleaved MRoPE reduces to the regular text RoPE frequencies.
        position32 = self.cast(position, FP32, f"{stem}_position_fp32")
        position32 = self.g.unsqueeze(position32, [0], [1, 1], f"{stem}_position_unsqueeze")
        inv = self.g.const(
            "qwen_rope_inv_freq",
            np.asarray(1.0 / (self.rope_theta ** (np.arange(0, self.rope_dim, 2, dtype=np.float32) / self.rope_dim)), dtype=np.float32),
        )
        freq = self.g.binary("Mul", position32, inv, [1, self.rope_dim // 2], FP32, f"{stem}_freq")
        freq = self.g.concat([freq, freq], 1, [1, self.rope_dim], FP32, f"{stem}_freq_concat")
        freq = self.g.reshape(freq, [1, 1, 1, self.rope_dim], f"{stem}_freq_reshape")
        cos = self.g.unary("Cos", freq, [1, 1, 1, self.rope_dim], FP32, f"{stem}_cos")
        sin = self.g.unary("Sin", freq, [1, 1, 1, self.rope_dim], FP32, f"{stem}_sin")
        cos = self.cast(cos, BF16, f"{stem}_cos_bf16")
        sin = self.cast(sin, BF16, f"{stem}_sin_bf16")
        shape = self.g.value_info[x][0]
        rot_shape = list(shape)
        rot_shape[-1] = self.rope_dim
        pass_shape = list(shape)
        pass_shape[-1] -= self.rope_dim
        x_rot, x_pass = self.g.split(x, [self.rope_dim, shape[-1] - self.rope_dim], 3, [rot_shape, pass_shape], f"{stem}_split")
        first, second = self.g.split(x_rot, [self.rope_dim // 2, self.rope_dim // 2], 3, [[*shape[:-1], self.rope_dim // 2], [*shape[:-1], self.rope_dim // 2]], f"{stem}_half_split")
        rotated = self.g.concat([self.g.unary("Neg", second, [*shape[:-1], self.rope_dim // 2], BF16, f"{stem}_neg"), first], 3, rot_shape, BF16, f"{stem}_rotate")
        rotated = self.g.binary("Mul", x_rot, cos, rot_shape, BF16, f"{stem}_cos_mul")
        rotated_part = self.g.binary("Mul", self.g.concat([self.g.unary("Neg", second, [*shape[:-1], self.rope_dim // 2], BF16, f"{stem}_neg_again"), first], 3, rot_shape, BF16, f"{stem}_rotate_again"), sin, rot_shape, BF16, f"{stem}_sin_mul")
        rotated = self.g.binary("Add", rotated, rotated_part, rot_shape, BF16, f"{stem}_add")
        return self.g.concat([rotated, x_pass], 3, shape, BF16, f"{stem}_output")

    def build_full_layer(self, layer, hidden, position, attention_mask):
        stem = f"layer_{layer}_full"
        residual = hidden
        hidden = self.rms_norm(hidden, f"model.language_model.layers.{layer}.input_layernorm.weight", f"{stem}_input_norm")
        q_gate = self.projection(hidden, f"model.language_model.layers.{layer}.self_attn.q_proj.weight", self.full_heads * self.head_dim * 2, f"{stem}_q_gate_proj")
        q_gate = self.g.reshape(
            q_gate, [1, 1, self.full_heads, self.head_dim * 2], f"{stem}_q_gate_reshape"
        )
        q_heads, gate_heads = self.g.split(
            q_gate,
            [self.head_dim, self.head_dim],
            3,
            [[1, 1, self.full_heads, self.head_dim], [1, 1, self.full_heads, self.head_dim]],
            f"{stem}_q_gate_split",
        )
        q = q_heads
        q = self.g.transpose(q, [0, 2, 1, 3], [1, self.full_heads, 1, self.head_dim], f"{stem}_q_transpose")
        gate = self.g.reshape(
            gate_heads, [1, 1, self.full_heads * self.head_dim], f"{stem}_gate_reshape"
        )
        k = self.projection(hidden, f"model.language_model.layers.{layer}.self_attn.k_proj.weight", self.kv_heads * self.head_dim, f"{stem}_k_proj")
        k = self.g.reshape(k, [1, 1, self.kv_heads, self.head_dim], f"{stem}_k_reshape")
        k = self.g.transpose(k, [0, 2, 1, 3], [1, self.kv_heads, 1, self.head_dim], f"{stem}_k_transpose")
        v = self.projection(hidden, f"model.language_model.layers.{layer}.self_attn.v_proj.weight", self.kv_heads * self.head_dim, f"{stem}_v_proj")
        v = self.g.reshape(v, [1, 1, self.kv_heads, self.head_dim], f"{stem}_v_reshape")
        v = self.g.transpose(v, [0, 2, 1, 3], [1, self.kv_heads, 1, self.head_dim], f"{stem}_v_transpose")
        q = self.rms_norm(q, f"model.language_model.layers.{layer}.self_attn.q_norm.weight", f"{stem}_q_norm")
        k = self.rms_norm(k, f"model.language_model.layers.{layer}.self_attn.k_norm.weight", f"{stem}_k_norm")
        q = self.rope(q, position, f"{stem}_q_rope")
        k = self.rope(k, position, f"{stem}_k_rope")
        current_k = k
        current_v = v

        k_cache = self.g.concat(
            [f"k_cache_{layer}", current_k], 2,
            [1, self.kv_heads, self.max_context, self.head_dim], BF16, f"{stem}_k_cache_with_current"
        )
        k_cache = self.g.unsqueeze(k_cache, [2], [1, self.kv_heads, 1, self.max_context, self.head_dim], f"{stem}_k_cache_unsqueeze")
        k_cache = self.g.expand(k_cache, [1, self.kv_heads, self.full_heads // self.kv_heads, self.max_context, self.head_dim], f"{stem}_k_cache_expand")
        k_cache = self.g.reshape(k_cache, [1, self.full_heads, self.max_context, self.head_dim], f"{stem}_k_cache_reshape")
        v_cache = self.g.concat(
            [f"v_cache_{layer}", current_v], 2,
            [1, self.kv_heads, self.max_context, self.head_dim], BF16, f"{stem}_v_cache_with_current"
        )
        v_cache = self.g.unsqueeze(v_cache, [2], [1, self.kv_heads, 1, self.max_context, self.head_dim], f"{stem}_v_cache_unsqueeze")
        v_cache = self.g.expand(v_cache, [1, self.kv_heads, self.full_heads // self.kv_heads, self.max_context, self.head_dim], f"{stem}_v_cache_expand")
        v_cache = self.g.reshape(v_cache, [1, self.full_heads, self.max_context, self.head_dim], f"{stem}_v_cache_reshape")
        k_cache_t = self.g.transpose(k_cache, [0, 1, 3, 2], [1, self.full_heads, self.head_dim, self.max_context], f"{stem}_k_cache_transpose")
        scores = self.g.node("MatMul", [q, k_cache_t], self.g.fresh(f"{stem}_scores"), [[1, self.full_heads, 1, self.max_context]], BF16)
        scores = self.cast(scores, FP32, f"{stem}_scores_fp32")
        scores = self.g.binary("Mul", scores, self.g.scalar(f"{stem}_scale", 1.0 / (self.head_dim ** 0.5)), [1, self.full_heads, 1, self.max_context], FP32, f"{stem}_scale_scores")
        scores = self.g.binary("Add", scores, self.cast(attention_mask, FP32, f"{stem}_mask_fp32"), [1, self.full_heads, 1, self.max_context], FP32, f"{stem}_mask")
        probs = self.g.node("Softmax", [scores], self.g.fresh(f"{stem}_softmax"), [[1, self.full_heads, 1, self.max_context]], FP32, axis=-1)
        probs = self.cast(probs, BF16, f"{stem}_probs_bf16")
        attended = self.g.node("MatMul", [probs, v_cache], self.g.fresh(f"{stem}_attended"), [[1, self.full_heads, 1, self.head_dim]], BF16)
        attended = self.g.transpose(attended, [0, 2, 1, 3], [1, 1, self.full_heads, self.head_dim], f"{stem}_attended_transpose")
        attended = self.g.reshape(attended, [1, 1, self.full_heads * self.head_dim], f"{stem}_attended_reshape")
        gate = self.g.unary(
            "Sigmoid", gate, [1, 1, self.full_heads * self.head_dim], BF16, f"{stem}_gate_sigmoid"
        )
        attended = self.g.binary("Mul", attended, gate, [1, 1, self.full_heads * self.head_dim], BF16, f"{stem}_gate_mul")
        attended = self.projection(attended, f"model.language_model.layers.{layer}.self_attn.o_proj.weight", self.hidden, f"{stem}_out_proj")
        hidden = self.g.binary("Add", residual, attended, [1, 1, self.hidden], BF16, f"{stem}_residual")

        residual = hidden
        hidden = self.rms_norm(hidden, f"model.language_model.layers.{layer}.post_attention_layernorm.weight", f"{stem}_post_norm")
        gate = self.silu(self.projection(hidden, f"model.language_model.layers.{layer}.mlp.gate_proj.weight", self.intermediate, f"{stem}_mlp_gate"), f"{stem}_mlp_silu")
        up = self.projection(hidden, f"model.language_model.layers.{layer}.mlp.up_proj.weight", self.intermediate, f"{stem}_mlp_up")
        product = self.g.binary("Mul", gate, up, [1, 1, self.intermediate], BF16, f"{stem}_mlp_product")
        down = self.projection(product, f"model.language_model.layers.{layer}.mlp.down_proj.weight", self.hidden, f"{stem}_mlp_down")
        hidden = self.g.binary("Add", residual, down, [1, 1, self.hidden], BF16, f"{stem}_mlp_residual")
        return hidden, current_k, current_v

    def build(self):
        g = self.g
        token_ids = g.input("token_ids", [1, 1], INT64)
        position = g.input("position_id", [1], INT64)
        attention_mask = g.input("attention_mask", [1, 1, 1, self.max_context], BF16)
        for layer in range(self.layers):
            if self.text["layer_types"][layer] == "linear_attention":
                g.input(f"conv_state_{layer}", [1, self.conv_dim, int(self.text["linear_conv_kernel_dim"]) - 1], BF16)
                g.input(f"recurrent_state_{layer}", [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32)
            else:
                g.input(f"k_cache_{layer}", [1, self.kv_heads, self.max_context - 1, self.head_dim], BF16)
                g.input(f"v_cache_{layer}", [1, self.kv_heads, self.max_context - 1, self.head_dim], BF16)
        embedding = self.weight(
            "model.language_model.embed_tokens.weight", force_dtype=self.fp8_dtype
        ) if self.fp8_dtype is not None else self.weight("model.language_model.embed_tokens.weight")
        hidden = g.node(
            "Gather",
            [embedding, token_ids],
            g.fresh("embedding"),
            [[1, 1, self.hidden]],
            self.fp8_dtype if self.fp8_dtype is not None else BF16,
            quantization_scales=self.fp8_activation_scale if self.fp8_dtype is not None else None,
            axis=0,
        )
        if self.fp8_dtype is not None:
            hidden = self.cast(hidden, BF16, "embedding_to_bf16")
        for layer in range(self.layers):
            if self.text["layer_types"][layer] == "linear_attention":
                hidden, next_conv, next_state = self.build_linear_layer(layer, hidden)
                next_conv = g.node("Identity", [next_conv], f"next_conv_state_{layer}",
                                   [[1, self.conv_dim, 3]], BF16)
                next_state = g.node("Identity", [next_state], f"next_recurrent_state_{layer}",
                                    [[1, self.linear_heads, self.linear_key_dim, self.linear_value_dim]], FP32)
                g.output(next_conv, [1, self.conv_dim, 3], BF16)
                g.output(next_state, [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32)
            else:
                hidden, current_k, current_v = self.build_full_layer(layer, hidden, position, attention_mask)
                current_k = g.node("Identity", [current_k], f"next_k_cache_{layer}",
                                   [[1, self.kv_heads, 1, self.head_dim]], BF16)
                current_v = g.node("Identity", [current_v], f"next_v_cache_{layer}",
                                   [[1, self.kv_heads, 1, self.head_dim]], BF16)
                g.output(current_k, [1, self.kv_heads, 1, self.head_dim], BF16)
                g.output(current_v, [1, self.kv_heads, 1, self.head_dim], BF16)
        hidden = self.rms_norm(hidden, "model.language_model.norm.weight", "final_norm")
        hidden = g.reshape(hidden, [1, self.hidden], "lm_head_input")
        if self.fp8_dtype is None:
            embedding_logits = self.weight("model.language_model.embed_tokens.weight")
            logits = g.node(
                "Gemm",
                [hidden, embedding_logits],
                "logits",
                [[1, self.vocab]],
                BF16,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1,
            )
        else:
            hidden = self.cast(hidden, self.fp8_dtype, "lm_head_input_to_fp8", self.fp8_activation_scale)
            embedding_logits = self.weight("model.language_model.embed_tokens.weight", force_dtype=self.fp8_dtype)
            fp8_logits = g.node(
                "Gemm",
                [hidden, embedding_logits],
                g.fresh("logits_fp8"),
                [[1, self.vocab]],
                self.fp8_dtype,
                quantization_scales=self.fp8_activation_scale,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1,
            )
            logits = g.node(
                "Cast",
                [fp8_logits],
                "logits",
                [[1, self.vocab]],
                BF16,
                to=FTH_TO_ONNX_DTYPE[BF16],
            )
        g.output(logits, [1, self.vocab], BF16)
        return g

    def make_onnx_model(self):
        return self.g.make_model(self.max_context)

    def write_fth(self, path):
        precision = "bf16" if self.fp8_dtype is None else fp8_format_name(self.fp8_dtype)
        self.g.write_fth(path, f"qwen3.5-0.8b_decode_{precision}")


def resolve_shard(model_dir):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        if len(shards) != 1:
            raise NotImplementedError("the exporter currently expects one safetensors shard")
        return model_dir / shards[0]
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError("expected exactly one safetensors shard")
    return shards[0]


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-0.8B safetensors directly to atomic Feather FTH")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--fp8-format", choices=("e4m3", "e5m2"),
                        help="Export FP8 compute islands with the selected format.")
    parser.add_argument("--fp8-activation-amax", type=float, default=96.0,
                        help="Static finite activation bound used to derive per-tensor FP8 scales.")
    parser.add_argument("--output-fth", type=Path)
    parser.add_argument("--output-onnx", type=Path,
                        help="Optional debug ONNX path. This is not used by FTH export.")
    args = parser.parse_args()
    if args.max_context < 2:
        raise ValueError("--max-context must be at least 2")
    if args.fp8_format is not None and (
            not math.isfinite(args.fp8_activation_amax) or args.fp8_activation_amax <= 0.0):
        raise ValueError("--fp8-activation-amax must be finite and positive")
    model_dir = args.model_dir.resolve()
    fp8_dtype = {"e4m3": FP8E4M3, "e5m2": FP8E5M2}.get(args.fp8_format)
    precision = "bf16" if fp8_dtype is None else fp8_format_name(fp8_dtype)
    output_fth = args.output_fth or model_dir / f"qwen3.5-0.8b_decode_{precision}_ctx{args.max_context}.fth"
    output_fth.parent.mkdir(parents=True, exist_ok=True)
    if args.output_onnx is not None:
        args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    shard = resolve_shard(model_dir)
    builder = QwenBuilder(model_dir, args.max_context, fp8_dtype=fp8_dtype,
                          fp8_activation_amax=args.fp8_activation_amax)
    with safe_open(shard, framework="pt", device="cpu") as source:
        builder.source = source
        graph = builder.build()
        builder.write_fth(output_fth)
        if args.output_onnx is not None:
            import onnx

            model = builder.make_onnx_model()
            onnx.checker.check_model(model)
            onnx.save_model(model, args.output_onnx)
    counts = {}
    for node in graph.nodes:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    print("exported FTH:", output_fth)
    print("inputs:", len(graph.inputs), "outputs:", len(graph.outputs), "nodes:", len(graph.nodes))
    print("operators:", " ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    if args.output_onnx is not None:
        print("exported debug ONNX:", args.output_onnx)


if __name__ == "__main__":
    main()
