#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export official detection checkpoints and attach Feather head metadata.

Example:
    python tools/export_yolo_models.py --families yolov6 yolov8 yolov10 yolo11 yolo26

Exports use CPU, one torch thread, batch 1, a square static FP32 input, and ONNX
opset 13. YOLOv10/YOLO26 retain their official one-to-one final detection output.
The pinned Ultralytics package and official YOLOv6 source live under /tmp; no
system Python packages are installed. Checkpoints, graphs, provenance and logs
are retained under models/yolo/<family>/.
"""

import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile


ULTRALYTICS_VERSION = "8.4.183"
ULTRALYTICS_RELEASE = "v8.4.0"
YOLOV6_RELEASE = "0.4.0"
FAMILIES = ("yolov6", "yolov8", "yolov10", "yolo11", "yolo26")
VARIANTS = {
    "yolov6": ("n", "s", "m", "l"),
    "yolov8": ("n", "s", "m", "l", "x"),
    "yolov10": ("n", "s", "m", "b", "l", "x"),
    "yolo11": ("n", "s", "m", "l", "x"),
    "yolo26": ("n", "s", "m", "l", "x"),
}
YOLOV6_SOURCE_URL = f"https://codeload.github.com/meituan/YOLOv6/zip/refs/tags/{YOLOV6_RELEASE}"


def checkpoint_name(family, variant="n"):
    if variant not in VARIANTS[family]:
        raise ValueError(f"Unsupported {family} variant: {variant}")
    return f"{family}{variant}.pt"


def checkpoint_url(family, variant="n"):
    repo = "meituan/YOLOv6" if family == "yolov6" else "ultralytics/assets"
    release = YOLOV6_RELEASE if family == "yolov6" else ULTRALYTICS_RELEASE
    return f"https://github.com/{repo}/releases/download/{release}/{checkpoint_name(family, variant)}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_ultralytics(dependencies):
    dependencies.mkdir(parents=True, exist_ok=True)
    if not (dependencies / "ultralytics" / "__init__.py").is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(dependencies),
                        f"ultralytics=={ULTRALYTICS_VERSION}"], check=True)
    sys.path.insert(0, str(dependencies))
    importlib.invalidate_caches()
    import ultralytics
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, found {ultralytics.__version__} in "
                           f"{ultralytics.__file__}; use a clean --ultralytics-deps directory")
    return ultralytics


def prepare_yolov6(source):
    if (source / "deploy" / "ONNX" / "export_onnx.py").is_file():
        return
    print(f"Downloading official YOLOv6 source {YOLOV6_RELEASE}", flush=True)
    with urllib.request.urlopen(YOLOV6_SOURCE_URL, timeout=120) as response:
        archive_bytes = response.read()
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    prefix = archive.namelist()[0].split("/", 1)[0] + "/"
    source.mkdir(parents=True, exist_ok=True)
    for item in archive.infolist():
        relative = item.filename[len(prefix):]
        if not relative:
            continue
        destination = (source / relative).resolve()
        if not destination.is_relative_to(source):
            raise RuntimeError("Invalid path in YOLOv6 source archive")
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    (source / "feather_source.json").write_text(json.dumps({
        "url": YOLOV6_SOURCE_URL, "sha256": hashlib.sha256(archive_bytes).hexdigest(),
    }, indent=2) + "\n")


def export_yolov6(checkpoint, size, source):
    prepare_yolov6(source)
    script = source / "deploy" / "ONNX" / "export_onnx.py"
    # Use the exact current interpreter: pyenv may select a different Python in /tmp.
    launcher = ("import sys,runpy,torch; torch.set_num_threads(1); "
                "source=sys.argv[1]; sys.argv=sys.argv[2:]; sys.path.insert(0,source); "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable, "-c", launcher, str(source), str(script), "--weights", str(checkpoint),
               "--img-size", str(size), str(size), "--batch-size", "1", "--device", "cpu"]
    path = checkpoint.with_suffix(".onnx")
    path.unlink(missing_ok=True)
    log_path = checkpoint.with_name("export.log" if checkpoint.stem == "yolov6n" else f"{checkpoint.stem}.export.log")
    with log_path.open("w") as log:
        subprocess.run(command, cwd=source, check=True, stdout=log, stderr=subprocess.STDOUT)
    # The upstream exporter catches ONNX exceptions; check the artifact as well as its exit status.
    if not path.is_file():
        raise RuntimeError(f"YOLOv6 did not produce {path}; see {log_path}")
    return path


def export_ultralytics(family, checkpoint, size, dependencies):
    package = prepare_ultralytics(dependencies)
    import torch
    torch.set_num_threads(1)
    model = package.YOLO(str(checkpoint), task="detect")
    if len(model.names) != 80:
        raise RuntimeError("The official COCO checkpoint must expose 80 detection classes")
    options = dict(format="onnx", imgsz=size, batch=1, device="cpu", dynamic=False,
                   simplify=False, opset=13, nms=False)
    if family in ("yolov10", "yolo26"):
        options["end2end"] = True
    return Path(model.export(**options)).resolve()


def inspect_contract(family, path, size):
    import onnx
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError(f"{family} must expose one image input and one detection output")

    def shape(value):
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in value.type.tensor_type.shape.dim):
            raise RuntimeError(f"{family} has a dynamic or invalid shape: {value.name}")
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"{family} must use an FP32 input and output")
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    image_shape = shape(graph.graph.input[0])
    output_shape = shape(graph.graph.output[0])
    if image_shape != [1, 3, size, size]:
        raise RuntimeError(f"Unexpected image input shape: {image_shape}")
    if family == "yolov6":
        if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[2] != 85:
            raise RuntimeError(f"Unexpected YOLOv6 objectness head: {output_shape}")
        output_format, layout, nms = "cxcywh_objectness", "bnc", 1
    elif family in ("yolov8", "yolo11"):
        if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[1] != 84:
            raise RuntimeError(f"Unexpected {family} class-score head: {output_shape}")
        output_format, layout, nms = "cxcywh_scores", "bcn", 1
    else:
        if (len(output_shape) != 3 or output_shape[0] != 1 or output_shape[2] != 6 or
                not 0 < output_shape[1] <= 300):
            raise RuntimeError(f"Expected the official end-to-end [1,N,6] output with 0<N<=300, found {output_shape}")
        output_format, layout, nms = "xyxy_score_class", "bnc", 0
    metadata = {"yolo.version": 1, "yolo.family": family, "yolo.num_classes": 80,
                "yolo.output_format": output_format, "yolo.output_layout": layout, "yolo.nms_required": nms}
    inventory = {"input": {"name": graph.graph.input[0].name, "shape": image_shape},
                 "output": {"name": graph.graph.output[0].name, "shape": output_shape},
                 "operators": dict(sorted(Counter(node.op_type for node in graph.graph.node).items()))}
    return metadata, inventory


def export_family(family, args):
    directory = args.output_dir / family
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / checkpoint_name(family, args.variant)
    url = checkpoint_url(family, args.variant)
    download(url, checkpoint)
    onnx_path = checkpoint.with_suffix(".onnx")
    if not args.reuse_onnx or not onnx_path.is_file():
        print(f"Exporting {family} on CPU", flush=True)
        if family == "yolov6":
            onnx_path = export_yolov6(checkpoint, args.size, args.yolov6_source)
        else:
            onnx_path = export_ultralytics(family, checkpoint, args.size, args.ultralytics_deps)
    metadata, inventory = inspect_contract(family, onnx_path, args.size)
    provenance = {"family": family, "variant": args.variant,
                  "checkpoint_url": url, "checkpoint_sha256": sha256(checkpoint),
                  "checkpoint": checkpoint.name, "onnx": onnx_path.name, "onnx_sha256": sha256(onnx_path),
                  "exporter": (f"Meituan YOLOv6 {YOLOV6_RELEASE}" if family == "yolov6" else
                               f"Ultralytics {ULTRALYTICS_VERSION}"),
                  "source_url": (YOLOV6_SOURCE_URL if family == "yolov6" else
                                 f"https://pypi.org/project/ultralytics/{ULTRALYTICS_VERSION}/"),
                  "export_options": {"device": "cpu", "threads": 1, "batch": 1, "size": args.size,
                                     "dtype": "float32", "opset": 13, "dynamic": False,
                                     "simplify": False, "nms_wrapper": False},
                  "model_metadata": metadata, **inventory}
    provenance_name = "provenance.json" if args.variant == "n" else f"{checkpoint.stem}.provenance.json"
    (directory / provenance_name).write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"family": family, "model_metadata": metadata, **inventory}), flush=True)
    if not args.skip_convert:
        from onnx_to_feather import convert_model
        destination = checkpoint.with_suffix(".fth")
        convert_model(str(onnx_path), str(destination), layout=args.layout, model_metadata=metadata)
        print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--variant", choices=("n", "s", "m", "b", "l", "x"), default="n",
                        help="Official model size; every selected family must provide this variant (default: n)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/yolo"))
    parser.add_argument("--size", type=int, default=640, help="Square static input size, divisible by 32")
    parser.add_argument("--layout", choices=("nchw",), default="nchw")
    parser.add_argument("--ultralytics-deps", type=Path, default=Path("/tmp/infer-yolo-export-deps"))
    parser.add_argument("--yolov6-source", type=Path, default=Path("/tmp/infer-yolov6-0.4.0"))
    parser.add_argument("--reuse-onnx", action="store_true", help="Validate and convert an existing ONNX export")
    parser.add_argument("--skip-convert", action="store_true", help="Obtain ONNX graphs without converting to FTH")
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        parser.error("--size must be positive and divisible by 32")
    incompatible = [family for family in args.families if args.variant not in VARIANTS[family]]
    if incompatible:
        supported = "; ".join(f"{family}: {','.join(VARIANTS[family])}" for family in incompatible)
        parser.error(f"--variant {args.variant} is unavailable for selected families ({supported})")
    args.output_dir = args.output_dir.resolve()
    args.ultralytics_deps = args.ultralytics_deps.resolve()
    args.yolov6_source = args.yolov6_source.resolve()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output_dir / ".ultralytics"))
    for family in args.families:
        export_family(family, args)


if __name__ == "__main__":
    main()
