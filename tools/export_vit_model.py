#!/usr/bin/env python3
"""Export a real ViT classifier to fixed-shape ONNX and Feather formats.

The source checkpoint is retrieved through the configured Hugging Face endpoint.
By default, it uses hf-mirror and writes all persistent assets below models/vit.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import torch


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "google/vit-base-patch16-224"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "models" / "vit"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


class PrimitiveLayerNorm(torch.nn.Module):
    """Export LayerNorm as the standard atomic ONNX operator graph."""

    def __init__(self, source: torch.nn.LayerNorm):
        super().__init__()
        self.eps = source.eps
        self.weight = torch.nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(value, dim=-1, keepdim=True)
        centered = value - mean
        variance = torch.mean(torch.pow(centered, 2.0), dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


def replace_layer_norm_modules(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, PrimitiveLayerNorm(child))
        else:
            replace_layer_norm_modules(child)


class ViTLogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values, return_dict=True).logits


def configure_environment(output_dir: pathlib.Path) -> None:
    endpoint = os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
    os.environ.setdefault("HF_HOME", str(output_dir / ".hf_cache"))
    print(f"Hugging Face endpoint: {endpoint}")
    print(f"Hugging Face cache: {os.environ['HF_HOME']}")


def export_model(model_id: str, output_dir: pathlib.Path, force: bool) -> tuple[pathlib.Path, pathlib.Path]:
    from transformers import AutoModelForImageClassification

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vit_base_patch16_224_static.onnx"
    fth_path = output_dir / "vit_base_patch16_224_static.fth"
    if onnx_path.exists() and fth_path.exists() and not force:
        print(f"ready {onnx_path}")
        print(f"ready {fth_path}")
        return onnx_path, fth_path

    model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=str(output_dir / ".hf_cache"))
    model.eval()
    replace_layer_norm_modules(model)
    wrapper = ViTLogitsWrapper(model)
    wrapper.eval()

    temporary_path = onnx_path.with_suffix(".onnx.part")
    temporary_path.unlink(missing_ok=True)
    pixels = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    print(f"exporting {model_id} -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pixels,),
            str(temporary_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            export_params=True,
        )
    temporary_path.replace(onnx_path)

    converter = REPOSITORY_ROOT / "tools" / "onnx_to_feather.py"
    command = [sys.executable, str(converter), "--input", str(onnx_path), "--output", str(fth_path)]
    print("converting ONNX to Feather:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"ready {onnx_path} ({onnx_path.stat().st_size} bytes)")
    print(f"ready {fth_path} ({fth_path.stat().st_size} bytes)")
    return onnx_path, fth_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_environment(args.output_dir)
    export_model(args.model_id, args.output_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
