#!/usr/bin/env python3
"""Download and validate the real ONNX classification model fixtures.

Persistent model assets are stored below the repository's models directory by
default. Hugging Face downloads use hf-mirror.com unless HF_ENDPOINT is
explicitly set, which keeps model acquisition reproducible in this environment.
"""

import argparse
import hashlib
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import onnx


MODELS = {
    "resnet50": {
        "filename": "gluon_resnet50_v1b_Opset17.onnx",
        "repo": "onnxmodelzoo/gluon_resnet50_v1b_Opset17",
        "hf_filename": "gluon_resnet50_v1b_Opset17.onnx",
        "size": 102146206,
    },
    "repvgg": {
        "filename": "repvgg_b0_Opset17.onnx",
        "repo": "onnxmodelzoo/repvgg_b0_Opset17",
        "hf_filename": "repvgg_b0_Opset17.onnx",
        "size": 63303775,
    },
}

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "models" / "classification"


def hf_url(model: dict) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    return f"{endpoint}/{model['repo']}/resolve/main/{model['hf_filename']}"


def model_url(model: dict) -> str:
    return model.get("url") or hf_url(model)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_onnx(path: pathlib.Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    print(
        f"validated {path.name}: opset="
        f"{','.join(str(opset.version) for opset in model.opset_import)} "
        f"nodes={len(model.graph.node)} "
        f"inputs={len(model.graph.input)} outputs={len(model.graph.output)}"
    )


def download_one(name: str, destination: pathlib.Path, retries: int, force: bool) -> pathlib.Path:
    model = MODELS[name]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model["filename"]
    partial = target.with_suffix(target.suffix + ".part")

    if force:
        target.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    if target.exists():
        if target.stat().st_size != model["size"]:
            print(f"discarding incomplete target {target}", file=sys.stderr)
            target.unlink()
        else:
            validate_onnx(target)
            print(f"ready {target} sha256={sha256(target)}")
            return target

    url = model_url(model)
    headers = {"User-Agent": "feather-classification-model-downloader/1.0"}
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > model["size"]:
            partial.unlink()
            offset = 0
        if offset == model["size"]:
            try:
                validate_onnx(partial)
                partial.replace(target)
                print(f"ready {target} sha256={sha256(target)}")
                return target
            except Exception as error:
                print(f"partial validation failed ({error}); restarting", file=sys.stderr)
                partial.unlink(missing_ok=True)
                offset = 0

        request = urllib.request.Request(url, headers=headers)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                append = offset > 0 and status == 206
                if offset and not append:
                    print("server ignored resume range; restarting download", file=sys.stderr)
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    copied = offset
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        copied += len(chunk)
                        print(f"\r{name}: {copied}/{model['size']} bytes", end="", flush=True)
                print()
            if partial.stat().st_size != model["size"]:
                raise IOError(
                    f"size mismatch: got {partial.stat().st_size}, expected {model['size']}"
                )
            validate_onnx(partial)
            partial.replace(target)
            print(f"ready {target} sha256={sha256(target)}")
            return target
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as error:
            print(f"attempt {attempt}/{retries} for {name} failed: {error}", file=sys.stderr)
            if attempt != retries:
                time.sleep(min(2 ** (attempt - 1), 10))

    raise RuntimeError(f"unable to download {name} from {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Persistent model directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.retries <= 0:
        parser.error("--retries must be positive")

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        download_one(name, args.output_dir, args.retries, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
