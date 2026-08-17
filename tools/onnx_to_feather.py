#!/home/jarvis/miniconda3/bin/python3

import argparse
import hashlib
import io
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


MAGIC = b"FTHMODL\x00"
FORMAT_VERSION = 1
WEIGHT_ALIGNMENT = 64

DTYPE_MAP = {
    onnx.TensorProto.BOOL: 9,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT8: 2,
    onnx.TensorProto.FLOAT16: 3,
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.INT32: 5,
    onnx.TensorProto.INT64: 6,
    onnx.TensorProto.BFLOAT16: 11,
}

LAYOUT_NCHW = 0
LAYOUT_NHWC = 1
LAYOUT_ND = 2


@dataclass
class TensorDesc:
    name: str
    dims: List[int]
    data_type: int
    layout: int = LAYOUT_ND


@dataclass
class ValueDesc:
    tensor: TensorDesc
    constant: bool = False
    weight_offset: int = 0
    weight_size: int = 0
    checksum: str = ""


@dataclass
class NodeDesc:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def encode_vector_int64(values: List[int]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(struct.pack("<q", int(v)) for v in values)


def encode_vector_float(values: List[float]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(struct.pack("<f", float(v)) for v in values)


def encode_attribute(value: Any) -> bytes:
    if isinstance(value, int):
        return struct.pack("<I", 0) + struct.pack("<q", value)
    if isinstance(value, float):
        return struct.pack("<I", 1) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<I", 2) + encode_string(value)
    if isinstance(value, list):
        if not value:
            raise ValueError("empty attribute list is not supported")
        if all(isinstance(v, int) for v in value):
            return struct.pack("<I", 3) + encode_vector_int64(value)
        if all(isinstance(v, (int, float)) for v in value):
            return struct.pack("<I", 4) + encode_vector_float([float(v) for v in value])
    raise TypeError(f"unsupported attribute type: {type(value)!r}")


def encode_tensor_desc(desc: TensorDesc) -> bytes:
    return (
        encode_string(desc.name)
        + encode_vector_int64(desc.dims)
        + struct.pack("<i", desc.data_type)
        + struct.pack("<i", desc.layout)
    )


def encode_weight_location(value: ValueDesc, output_path: str) -> bytes:
    return (
        encode_string(value.tensor.name)
        + encode_string(output_path)
        + struct.pack("<Q", value.weight_offset)
        + struct.pack("<Q", value.weight_size)
        + encode_string(value.checksum)
    )


def encode_value_desc(value: ValueDesc, output_path: str) -> bytes:
    return encode_tensor_desc(value.tensor) + struct.pack("<B", 1 if value.constant else 0) + encode_weight_location(value, output_path)


def encode_string_vector(values: List[str]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(encode_string(v) for v in values)


def encode_node_desc(node: NodeDesc) -> bytes:
    payload = (
        encode_string(node.name)
        + encode_string(node.op_type)
        + encode_string(node.domain)
        + encode_string_vector(node.inputs)
        + encode_string_vector(node.outputs)
        + struct.pack("<Q", len(node.attributes))
    )
    for key, value in node.attributes.items():
        payload += encode_string(key) + encode_attribute(value)
    return payload


def encode_graph(name: str, inputs: List[str], outputs: List[str], values: List[ValueDesc], nodes: List[NodeDesc], output_path: str) -> bytes:
    payload = encode_string(name) + encode_string_vector(inputs) + encode_string_vector(outputs)
    payload += struct.pack("<Q", len(values))
    for value in values:
        payload += encode_value_desc(value, output_path)
    payload += struct.pack("<Q", len(nodes))
    for node in nodes:
        payload += encode_node_desc(node)
    return payload


def encode_model(name: str, version: int, graph_name: str, inputs: List[str], outputs: List[str], values: List[ValueDesc], nodes: List[NodeDesc], output_path: str) -> bytes:
    return encode_string(name) + struct.pack("<q", version) + encode_graph(graph_name, inputs, outputs, values, nodes, output_path)


def tensor_proto_to_array(tensor: onnx.TensorProto) -> np.ndarray:
    return numpy_helper.to_array(tensor)


def node_constant_to_array(node: onnx.NodeProto) -> np.ndarray:
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(helper.get_attribute_value(attr))
    raise ValueError(f"Constant node {node.name or node.output[0]} is missing value")


def normalize_dims(shape_proto, allow_unknown: bool = False) -> List[int]:
    dims: List[int] = []
    for dim in shape_proto.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            if allow_unknown:
                dims.append(0)
                continue
            raise ValueError(f"dynamic dimension {dim.dim_param!r} is not supported yet")
        else:
            dims.append(0)
    return dims


def tensor_type_info(value_info, allow_unknown_dims: bool = False) -> Tuple[List[int], int]:
    tensor_type = value_info.type.tensor_type
    return normalize_dims(tensor_type.shape, allow_unknown=allow_unknown_dims), DTYPE_MAP.get(tensor_type.elem_type, 0)


def sanitize_node_name(node: onnx.NodeProto, index: int) -> str:
    if node.name:
        return node.name
    if node.output:
        return node.output[0].replace("/", "_")
    return f"{node.op_type}_{index}"


def dtype_from_numpy(array: np.ndarray) -> int:
    if is_bfloat16_dtype(array.dtype):
        return 11
    if array.dtype == np.bool_:
        return 9
    if array.dtype == np.float16:
        return 3
    if array.dtype == np.float32:
        return 4
    if array.dtype == np.int64:
        return 6
    if array.dtype == np.int32:
        return 5
    if array.dtype == np.int8:
        return 1
    if array.dtype == np.uint8:
        return 2
    raise TypeError(f"unsupported numpy dtype: {array.dtype}")


def is_bfloat16_dtype(dtype: np.dtype) -> bool:
    if str(dtype) == "bfloat16":
        return True
    fields = dtype.fields
    return fields is not None and set(fields) == {"bfloat16"} and fields["bfloat16"][0].itemsize == 2


def bytes_for_tensor(array: np.ndarray) -> bytes:
    if array.dtype == np.bool_:
        return array.astype(np.bool_, copy=False).tobytes()
    if array.dtype == np.float16:
        return array.astype(np.float16, copy=False).tobytes()
    if is_bfloat16_dtype(array.dtype):
        return array.tobytes()
    if array.dtype == np.float32:
        return array.astype(np.float32, copy=False).tobytes()
    if array.dtype == np.int64:
        return array.astype(np.int64, copy=False).tobytes()
    if array.dtype == np.int32:
        return array.astype(np.int32, copy=False).tobytes()
    if array.dtype == np.int8:
        return array.astype(np.int8, copy=False).tobytes()
    if array.dtype == np.uint8:
        return array.astype(np.uint8, copy=False).tobytes()
    raise TypeError(f"unsupported array dtype: {array.dtype}")


def get_attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def make_tensor_desc(name: str, dims: List[int], dtype: int) -> TensorDesc:
    return TensorDesc(name=name, dims=[int(v) for v in dims], data_type=dtype)

def permute_nchw_to_nhwc_4d(dims: List[int]) -> List[int]:
    if len(dims) != 4:
        return [int(v) for v in dims]
    return [int(dims[0]), int(dims[2]), int(dims[3]), int(dims[1])]


def remap_axis_nchw_to_nhwc(axis: int, rank: int) -> int:
    if rank != 4:
        return int(axis)
    normalized = axis if axis >= 0 else axis + rank
    mapping = {0: 0, 1: 3, 2: 1, 3: 2}
    return mapping.get(normalized, normalized)


def remap_scales_nchw_to_nhwc(scales: List[float]) -> List[float]:
    if len(scales) != 4:
        return [float(v) for v in scales]
    return [float(scales[0]), float(scales[2]), float(scales[3]), float(scales[1])]


def remap_yolo_head_shape_nchw_to_nhwc(shape: List[int]) -> List[int]:
    if len(shape) == 5 and shape[1] > 0 and shape[2] > 0 and shape[3] > 0 and shape[4] > 0:
        return [int(shape[0]), int(shape[3]), int(shape[4]), int(shape[1]), int(shape[2])]
    return [int(v) for v in shape]


def should_mark_nhwc_tensor(name: str, dims: List[int], graph_inputs: List[str], graph_outputs: List[str],
                            op_type_by_output: Dict[str, str]) -> bool:
    if len(dims) != 4:
        return False
    if name in graph_inputs:
        return True
    if op_type_by_output.get(name) in {"Conv", "MaxPool", "Resize", "Sigmoid", "Mul", "Add", "Concat"}:
        return True
    return False


def convert_model(input_path: str, output_path: str, layout: str = "nchw") -> None:
    model = onnx.load(input_path)
    inferred = shape_inference.infer_shapes(model)
    emit_nhwc = layout.lower() == "nhwc"

    value_infos: Dict[str, Tuple[List[int], int]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.output):
        dims, dtype = tensor_type_info(value)
        value_infos[value.name] = (dims, dtype)
    for value in inferred.graph.value_info:
        dims, dtype = tensor_type_info(value, allow_unknown_dims=True)
        value_infos[value.name] = (dims, dtype)

    constants: Dict[str, np.ndarray] = {}
    for init in inferred.graph.initializer:
        constants[init.name] = tensor_proto_to_array(init)
        if init.name not in value_infos:
            value_infos[init.name] = (list(constants[init.name].shape), dtype_from_numpy(constants[init.name]))

    for node in inferred.graph.node:
        if node.op_type == "Constant":
            const_name = node.output[0]
            const_value = node_constant_to_array(node)
            constants[const_name] = const_value
            value_infos[const_name] = (list(const_value.shape), dtype_from_numpy(np.asarray(const_value).reshape(-1)[:1] if np.asarray(const_value).shape == () else np.asarray(const_value)))

    op_type_by_output: Dict[str, str] = {}
    for node in inferred.graph.node:
        for output_name in node.output:
            if output_name:
                op_type_by_output[output_name] = node.op_type

    values: Dict[str, ValueDesc] = {}
    weights: Dict[str, bytes] = {}

    def ensure_value(name: str, constant: bool = False) -> None:
        if name in values:
            if constant:
                values[name].constant = True
            return
        if name not in value_infos:
            raise KeyError(f"missing shape/type info for value {name}")
        dims, dtype = value_infos[name]
        if dtype == 0:
            array = constants.get(name)
            if array is None:
                raise TypeError(f"unsupported dtype for value {name}")
            dtype = dtype_from_numpy(np.asarray(array))
        values[name] = ValueDesc(tensor=make_tensor_desc(name, dims, dtype), constant=constant)

    for graph_input in inferred.graph.input:
        if graph_input.name in constants:
            continue
        ensure_value(graph_input.name, constant=False)

    for const_name, const_array in constants.items():
        ensure_value(const_name, constant=True)
        array = np.asarray(const_array)
        if array.shape == ():
            array = array.reshape(1)
            if values[const_name].tensor.dims:
                values[const_name].tensor.dims = [1]
        else:
            values[const_name].tensor.dims = list(array.shape)
        values[const_name].tensor.data_type = dtype_from_numpy(array)
        raw = bytes_for_tensor(array)
        values[const_name].weight_size = len(raw)
        values[const_name].checksum = hashlib.sha256(raw).hexdigest()
        weights[const_name] = raw

    nodes: List[NodeDesc] = []

    def tensor_as_ints(name: str) -> List[int]:
        data = np.asarray(constants[name]).reshape(-1)
        return [int(v) for v in data.tolist()]

    def tensor_as_floats(name: str) -> List[float]:
        data = np.asarray(constants[name]).reshape(-1)
        return [float(v) for v in data.tolist()]

    for index, node in enumerate(inferred.graph.node):
        if node.op_type == "Constant":
            continue

        name = sanitize_node_name(node, index)
        attrs: Dict[str, Any] = {}
        op_type = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)

        if op_type == "Conv":
            strides = list(get_attr(node, "strides", [1, 1]))
            pads = list(get_attr(node, "pads", [0, 0, 0, 0]))
            dilations = list(get_attr(node, "dilations", [1, 1]))
            attrs["stride_h"] = int(strides[0])
            attrs["stride_w"] = int(strides[1])
            attrs["pad_h"] = int(pads[0])
            attrs["pad_w"] = int(pads[1])
            attrs["dilation_h"] = int(dilations[0])
            attrs["dilation_w"] = int(dilations[1])
            attrs["group"] = int(get_attr(node, "group", 1))
        elif op_type == "Concat":
            attrs["axis"] = int(get_attr(node, "axis", 1))
        elif op_type == "MaxPool":
            kernel_shape = list(get_attr(node, "kernel_shape", [1, 1]))
            strides = list(get_attr(node, "strides", [1, 1]))
            pads = list(get_attr(node, "pads", [0, 0, 0, 0]))
            attrs["kernel_h"] = int(kernel_shape[0])
            attrs["kernel_w"] = int(kernel_shape[1])
            attrs["stride_h"] = int(strides[0])
            attrs["stride_w"] = int(strides[1])
            attrs["pad_h"] = int(pads[0])
            attrs["pad_w"] = int(pads[1])
        elif op_type == "AveragePool":
            kernel_shape = list(get_attr(node, "kernel_shape", [1, 1]))
            strides = list(get_attr(node, "strides", [1, 1]))
            pads = list(get_attr(node, "pads", [0, 0, 0, 0]))
            attrs["kernel_h"] = int(kernel_shape[0])
            attrs["kernel_w"] = int(kernel_shape[1])
            attrs["stride_h"] = int(strides[0])
            attrs["stride_w"] = int(strides[1])
            attrs["pad_h"] = int(pads[0])
            attrs["pad_w"] = int(pads[1])
        elif op_type == "BatchNormalization":
            attrs["epsilon"] = float(get_attr(node, "epsilon", 1e-5))
        elif op_type == "GlobalAveragePool":
            pass
        elif op_type == "Flatten":
            attrs["axis"] = int(get_attr(node, "axis", 1))
        elif op_type == "Gemm":
            attrs["alpha"] = float(get_attr(node, "alpha", 1.0))
            attrs["beta"] = float(get_attr(node, "beta", 1.0))
            attrs["transA"] = int(get_attr(node, "transA", 0))
            attrs["transB"] = int(get_attr(node, "transB", 0))
        elif op_type == "Resize":
            pass
        elif op_type == "Reshape":
            pass
        elif op_type == "Transpose":
            attrs["perm"] = [int(v) for v in list(get_attr(node, "perm", []))]
        elif op_type in {"Unsqueeze", "Squeeze"}:
            if len(inputs) == 1:
                attrs["axes"] = [int(v) for v in list(get_attr(node, "axes", []))]
        elif op_type == "Cast":
            attrs["to"] = int(get_attr(node, "to", 1))
        elif op_type == "Shape":
            if get_attr(node, "start", None) is not None:
                attrs["start"] = int(get_attr(node, "start", 0))
            if get_attr(node, "end", None) is not None:
                attrs["end"] = int(get_attr(node, "end", 9223372036854775807))
        elif op_type == "ConstantOfShape":
            value = get_attr(node, "value", None)
            if value is None:
                attrs["value_int"] = 0
            else:
                array = np.asarray(numpy_helper.to_array(value)).reshape(-1)
                if array.size != 1:
                    raise NotImplementedError("ConstantOfShape only supports scalar values")
                if np.issubdtype(array.dtype, np.floating):
                    attrs["value_float"] = float(array[0])
                elif np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
                    attrs["value_int"] = int(array[0])
                else:
                    raise NotImplementedError(f"unsupported ConstantOfShape value dtype: {array.dtype}")
        elif op_type == "Expand":
            pass
        elif op_type == "ReduceMean":
            attrs["axes"] = [int(v) for v in list(get_attr(node, "axes", []))]
            attrs["keepdims"] = int(get_attr(node, "keepdims", 1))
        elif op_type == "ReduceSum":
            attrs["axes"] = [int(v) for v in list(get_attr(node, "axes", []))]
            attrs["keepdims"] = int(get_attr(node, "keepdims", 1))
        elif op_type == "Gather":
            attrs["axis"] = int(get_attr(node, "axis", 0))
        elif op_type == "Softmax":
            attrs["axis"] = int(get_attr(node, "axis", -1))
        elif op_type == "Slice":
            pass
        elif op_type == "Split":
            attrs["axis"] = int(get_attr(node, "axis", 0))
            split = get_attr(node, "split", None)
            if split is not None:
                attrs["split_sizes"] = [int(v) for v in list(split)]
        elif op_type == "Pow":
            pass
        elif op_type in {"Add", "Mul", "Sigmoid", "Relu", "Identity", "Sub", "Div", "Sqrt", "Tanh", "Erf", "MatMul",
                         "Equal", "Where", "Exp", "Sin", "Cos", "Neg", "Softplus"}:
            pass
        else:
            raise NotImplementedError(f"unsupported ONNX op: {op_type}")

        for input_name in inputs:
            if input_name:
                ensure_value(input_name, constant=input_name in constants)
        for output_name in outputs:
            ensure_value(output_name, constant=False)

        nodes.append(NodeDesc(name=name, op_type=op_type, inputs=inputs, outputs=outputs, attributes=attrs))

    graph_inputs = [value.name for value in inferred.graph.input if value.name not in constants]
    graph_outputs = [value.name for value in inferred.graph.output]
    for output_name in graph_outputs:
        ensure_value(output_name, constant=False)

    if emit_nhwc:
        for value in values.values():
            if should_mark_nhwc_tensor(value.tensor.name, value.tensor.dims, graph_inputs, graph_outputs, op_type_by_output):
                value.tensor.dims = permute_nchw_to_nhwc_4d(value.tensor.dims)
                value.tensor.layout = LAYOUT_NHWC

        for node in nodes:
            output_dims = []
            if node.outputs and node.outputs[0] in values:
                output_dims = values[node.outputs[0]].tensor.dims
            if node.op_type == "Concat":
                rank = len(values[node.inputs[0]].tensor.dims) if node.inputs and node.inputs[0] in values else 0
                if "axis" in node.attributes:
                    node.attributes["axis"] = remap_axis_nchw_to_nhwc(int(node.attributes["axis"]), rank)
            elif node.op_type == "Split":
                rank = len(values[node.inputs[0]].tensor.dims) if node.inputs and node.inputs[0] in values else 0
                if "axis" in node.attributes:
                    node.attributes["axis"] = remap_axis_nchw_to_nhwc(int(node.attributes["axis"]), rank)
            elif node.op_type == "Resize":
                scales = node.attributes.get("scales")
                if isinstance(scales, list):
                    node.attributes["scales"] = remap_scales_nchw_to_nhwc(scales)
            elif node.op_type == "Transpose":
                perm = node.attributes.get("perm")
                if perm == [0, 1, 3, 4, 2]:
                    node.attributes["perm"] = [0, 3, 1, 2, 4]
            elif node.op_type == "Reshape":
                shape = node.attributes.get("shape")
                if isinstance(shape, list):
                    node.attributes["shape"] = remap_yolo_head_shape_nchw_to_nhwc(shape)

    value_list = list(values.values())

    for _ in range(4):
        metadata = encode_model(
            name=os.path.basename(input_path),
            version=1,
            graph_name=inferred.graph.name or "main",
            inputs=graph_inputs,
            outputs=graph_outputs,
            values=value_list,
            nodes=nodes,
            output_path=output_path,
        )
        current_offset = align_up(24 + len(metadata), WEIGHT_ALIGNMENT)
        for value in value_list:
            if not value.constant:
                continue
            raw = weights[value.tensor.name]
            value.weight_offset = current_offset
            value.weight_size = len(raw)
            current_offset = align_up(current_offset + len(raw), WEIGHT_ALIGNMENT)

    metadata = encode_model(
        name=os.path.basename(input_path),
        version=1,
        graph_name=inferred.graph.name or "main",
        inputs=graph_inputs,
        outputs=graph_outputs,
        values=value_list,
        nodes=nodes,
        output_path=output_path,
    )

    with open(output_path, "wb") as f:
        header = struct.pack("<8sIIQ", MAGIC, FORMAT_VERSION, 0, len(metadata))
        f.write(header)
        f.write(metadata)
        current = len(header) + len(metadata)
        for value in value_list:
            if not value.constant:
                continue
            while current < value.weight_offset:
                f.write(b"\x00")
                current += 1
            raw = weights[value.tensor.name]
            f.write(raw)
            current += len(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a YOLOv5-style ONNX model into Feather .fth format")
    parser.add_argument("--input", required=True, help="Path to input ONNX model")
    parser.add_argument("--output", required=True, help="Path to output .fth model")
    parser.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw",
                        help="Target image layout recorded in the exported Feather model")
    args = parser.parse_args()
    convert_model(args.input, args.output, args.layout)


if __name__ == "__main__":
    main()
