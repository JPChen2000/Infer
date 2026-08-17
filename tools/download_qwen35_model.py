#!/usr/bin/env python3
"""Download Qwen3.5-0.8B from ModelScope into the persistent models directory."""

import argparse
import os
import pathlib
import shutil
from typing import Sequence


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "models" / "llm" / "qwen3.5-0.8b"


def has_required_files(model_dir: pathlib.Path) -> bool:
    has_weight = (model_dir / "model.safetensors").is_file() or any(
        model_dir.glob("model.safetensors-*.safetensors")
    )
    return (
        (model_dir / "config.json").is_file()
        and (model_dir / "model.safetensors.index.json").is_file()
        and (model_dir / "tokenizer.json").is_file()
        and (model_dir / "vocab.json").is_file()
        and (model_dir / "merges.txt").is_file()
        and has_weight
    )


def download(model_id: str, output_dir: pathlib.Path) -> pathlib.Path:
    if has_required_files(output_dir):
        print(f"ready {output_dir}")
        return output_dir
    if output_dir.exists():
        raise RuntimeError(
            f"{output_dir} exists but does not contain a complete {model_id} snapshot; "
            "remove it manually before retrying"
        )

    from modelscope import snapshot_download

    cache_root = output_dir.parent / ".modelscope_download"
    cache_root.mkdir(parents=True, exist_ok=True)
    # A cancelled transfer can leave a ModelScope TemporaryDirectory behind.
    # It is only a staging area owned by this script and is never a model asset.
    shutil.rmtree(cache_root / "temp", ignore_errors=True)
    # ModelScope uses cache_root/temp for temporary chunks, keeping all model
    # data under models/ rather than an ephemeral system temporary directory.
    snapshot_dir = pathlib.Path(snapshot_download(model_id, cache_dir=str(cache_root))).resolve()
    if not has_required_files(snapshot_dir):
        raise RuntimeError(f"ModelScope snapshot is incomplete: {snapshot_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(snapshot_dir, output_dir)
    empty_owner_dir = snapshot_dir.parent
    if empty_owner_dir.exists() and not any(empty_owner_dir.iterdir()):
        empty_owner_dir.rmdir()
    stale_temp_dir = cache_root / "temp"
    if stale_temp_dir.exists() and not any(stale_temp_dir.iterdir()):
        stale_temp_dir.rmdir()
    if cache_root.exists() and not any(cache_root.iterdir()):
        cache_root.rmdir()
    print(f"ready {output_dir}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    model_dir = download(args.model_id, args.output_dir)
    files = sorted(path.relative_to(model_dir).as_posix() for path in model_dir.rglob("*") if path.is_file())
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
