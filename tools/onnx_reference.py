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
    parser.add_argument("--input-raw", required=True, help="Path to raw input tensor bytes")
    parser.add_argument("--output-raw", required=True, help="Path to raw float32 output bytes")
    args = parser.parse_args()

    model = onnx.load(args.model)
    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]

    input_shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
    input_dtype = DTYPE_MAP[graph_input.type.tensor_type.elem_type]

    raw = Path(args.input_raw).read_bytes()
    input_array = np.frombuffer(raw, dtype=input_dtype).reshape(input_shape)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    output_name = graph_output.name
    input_name = graph_input.name
    output = session.run([output_name], {input_name: input_array})[0].astype(np.float32, copy=False)
    Path(args.output_raw).write_bytes(output.tobytes())


if __name__ == "__main__":
    main()
