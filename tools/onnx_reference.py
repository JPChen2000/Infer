#!/home/jarvis/miniconda3/bin/python3

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


DTYPE_MAP = {
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime on a raw tensor input and dump raw float32 output")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input-raw", required=True, action="append", help="Path to raw input tensor bytes. Repeat for multi-input models in graph input order.")
    parser.add_argument("--output-raw", required=True, help="Path to raw float32 output bytes")
    args = parser.parse_args()

    model = onnx.load(args.model)
    graph_output = model.graph.output[0]

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    feeds = {}
    graph_inputs = [value for value in model.graph.input if value.name in {item.name for item in session.get_inputs()}]
    if len(args.input_raw) != len(graph_inputs):
        raise ValueError(f"expected {len(graph_inputs)} input raw files, got {len(args.input_raw)}")
    for graph_input, input_raw in zip(graph_inputs, args.input_raw):
        input_shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
        input_dtype = DTYPE_MAP[graph_input.type.tensor_type.elem_type]
        raw = Path(input_raw).read_bytes()
        feeds[graph_input.name] = np.frombuffer(raw, dtype=input_dtype).reshape(input_shape)
    output_name = graph_output.name
    output = session.run([output_name], feeds)[0].astype(np.float32, copy=False)
    Path(args.output_raw).write_bytes(output.tobytes())


if __name__ == "__main__":
    main()
