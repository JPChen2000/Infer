#!/usr/bin/env python3

import argparse
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "third_party" / "models" / "yolov5n.onnx"
LETTERBOX_VALUE = 114


ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-frame YOLOv5 ONNX inference with ONNX Runtime."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to yolov5n.onnx. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--image",
        help="Optional image path. If omitted, a deterministic random single-frame tensor is used.",
    )
    parser.add_argument(
        "--provider",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Execution provider to use. Default: cuda",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id. Default: 0")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPUExecutionProvider fallback after CUDAExecutionProvider.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup inference runs before timing. Default: 5",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Timed inference runs. Default: 1 for single-frame timing.",
    )
    parser.add_argument(
        "--dynamic-size",
        type=int,
        default=640,
        help="Input H/W used only when the model input has dynamic spatial dims. Default: 640",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random input seed. Default: 0")
    return parser.parse_args()


def create_session(
    model_path: Path, provider: str, device_id: int, allow_cpu_fallback: bool
) -> ort.InferenceSession:
    available = ort.get_available_providers()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Install onnxruntime-gpu and make sure "
                f"CUDA/cuDNN can be found. Available providers: {available}"
            )
        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": device_id}]
        if allow_cpu_fallback:
            providers.append("CPUExecutionProvider")
            provider_options.append({})
    else:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(f"CPUExecutionProvider is not available. Available providers: {available}")
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
        provider_options=provider_options,
    )
    if provider == "cuda" and not allow_cpu_fallback and hasattr(session, "disable_fallback"):
        session.disable_fallback()
    return session


def numpy_dtype(ort_type: str) -> np.dtype:
    dtype = ORT_DTYPE_TO_NUMPY.get(ort_type)
    if dtype is None:
        raise ValueError(f"unsupported model input dtype: {ort_type}")
    return dtype


def normalize_dim(value, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def input_shape(meta, dynamic_size: int) -> List[int]:
    shape = list(meta.shape)
    if len(shape) != 4:
        raise ValueError(f"expected a 4D YOLO input, got shape={shape}")

    batch = normalize_dim(shape[0], 1)
    if shape[1] == 3 or shape[1] == "3":
        return [
            batch,
            3,
            normalize_dim(shape[2], dynamic_size),
            normalize_dim(shape[3], dynamic_size),
        ]
    if shape[3] == 3 or shape[3] == "3":
        return [
            batch,
            normalize_dim(shape[1], dynamic_size),
            normalize_dim(shape[2], dynamic_size),
            3,
        ]
    raise ValueError(f"cannot infer NCHW/NHWC layout from input shape={shape}")


def input_layout(shape: Sequence[int]) -> str:
    if len(shape) == 4 and shape[1] == 3:
        return "nchw"
    if len(shape) == 4 and shape[3] == 3:
        return "nhwc"
    raise ValueError(f"cannot infer input layout from shape={shape}")


def load_rgb_image(image_path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("loading --image requires Pillow: pip install pillow") from exc

    with Image.open(image_path) as image:
        return image.convert("RGB")


def letterbox_image(image, target_h: int, target_w: int) -> Tuple[np.ndarray, float, int, int]:
    from PIL import Image

    width, height = image.size
    scale = min(target_w / float(width), target_h / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((resized_w, resized_h), resampling)
    canvas = Image.new("RGB", (target_w, target_h), (LETTERBOX_VALUE,) * 3)
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.float32) / 255.0, scale, pad_x, pad_y


def make_image_input(image_path: Path, shape: Sequence[int], dtype: np.dtype) -> Tuple[np.ndarray, float]:
    layout = input_layout(shape)
    target_h = shape[2] if layout == "nchw" else shape[1]
    target_w = shape[3] if layout == "nchw" else shape[2]

    begin = time.perf_counter()
    image = load_rgb_image(image_path)
    image_array, _, _, _ = letterbox_image(image, target_h, target_w)
    if layout == "nchw":
        tensor = np.transpose(image_array, (2, 0, 1))[None, ...]
    else:
        tensor = image_array[None, ...]
    tensor = tensor.astype(dtype, copy=False)
    preprocess_ms = (time.perf_counter() - begin) * 1000.0
    return np.ascontiguousarray(tensor), preprocess_ms


def make_random_input(shape: Sequence[int], dtype: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tensor = rng.random(shape, dtype=np.float32)
    return np.ascontiguousarray(tensor.astype(dtype, copy=False))


def summarize(values: Sequence[float]) -> str:
    if len(values) == 1:
        return f"single_frame_ms={values[0]:.4f}"

    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return (
        f"runs={len(values)} min_ms={ordered[0]:.4f} avg_ms={statistics.mean(values):.4f} "
        f"p50_ms={p50:.4f} p95_ms={ordered[p95_index]:.4f} max_ms={ordered[-1]:.4f}"
    )


def shape_text(shape: Iterable[int]) -> str:
    return "[" + ",".join(str(dim) for dim in shape) + "]"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if args.image and not Path(args.image).exists():
        raise FileNotFoundError(f"image not found: {args.image}")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be >= 0 and --runs must be > 0")

    session_create_begin = time.perf_counter()
    session = create_session(model_path, args.provider, args.device_id, args.allow_cpu_fallback)
    session_create_ms = (time.perf_counter() - session_create_begin) * 1000.0

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_shape(input_meta, args.dynamic_size)
    dtype = numpy_dtype(input_meta.type)

    if args.image:
        input_tensor, preprocess_ms = make_image_input(Path(args.image), shape, dtype)
        source = str(Path(args.image))
    else:
        input_tensor = make_random_input(shape, dtype, args.seed)
        preprocess_ms = 0.0
        source = f"random(seed={args.seed})"

    feeds = {input_name: input_tensor}
    outputs = None
    for _ in range(args.warmup):
        outputs = session.run(None, feeds)

    timings_ms: List[float] = []
    for _ in range(args.runs):
        begin = time.perf_counter()
        outputs = session.run(None, feeds)
        timings_ms.append((time.perf_counter() - begin) * 1000.0)

    output_shapes = [shape_text(output.shape) for output in (outputs or [])]
    print(f"model={model_path}")
    print(f"providers={session.get_providers()}")
    print(f"input={input_name} shape={shape_text(shape)} dtype={input_meta.type} source={source}")
    print(f"session_create_ms={session_create_ms:.4f}")
    if args.image:
        print(f"preprocess_ms={preprocess_ms:.4f}")
    print(summarize(timings_ms))
    print(f"output_shapes={output_shapes}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
