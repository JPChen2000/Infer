#!/usr/bin/env python3
"""Download small real Hugging Face Transformer models and export static ONNX graphs.

The exported ONNX files are persistent test assets.  They are intentionally
exported with fixed [1, 8] token inputs so the importer can validate the full
operator graph without introducing dynamic-shape support into the runtime.
"""

import argparse
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import torch


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "models" / "transformer"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "tiny-bert": {
        "model_id": "hf-internal-testing/tiny-random-BertModel",
        "filename": "tiny_random_bert_static.onnx",
        "kind": "bert",
        "output_name": "last_hidden_state",
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "input_ids": [2, 5, 9, 17, 4, 3, 8, 11],
    },
    "bert-spam": {
        "model_id": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
        "filename": "bert_tiny_spam_static.onnx",
        "kind": "sequence-classification",
        "output_name": "logits",
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "input_ids": [101, 23156, 999, 2017, 2031, 2180, 1037, 102],
    },
    "distilbert-sst2": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "filename": "distilbert_sst2_static.onnx",
        "kind": "sequence-classification",
        "output_name": "logits",
        "input_names": ["input_ids", "attention_mask"],
        "input_ids": [101, 2023, 2003, 2019, 14153, 3048, 0, 0],
    },
}


class TransformerExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.model(**kwargs, return_dict=True)
        if self.kind == "bert":
            return output.last_hidden_state
        return output.logits


class PrimitiveLayerNorm(torch.nn.Module):
    """Keep LayerNorm visible as primitive ONNX operators during export."""

    def __init__(self, source: torch.nn.LayerNorm):
        super().__init__()
        self.normalized_shape = tuple(source.normalized_shape)
        self.eps = source.eps
        self.weight = torch.nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(value, dim=-1, keepdim=True)
        centered = value - mean
        variance = torch.mean(torch.pow(centered, 2.0), dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


def ReplaceLayerNormModules(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, PrimitiveLayerNorm(child))
        else:
            ReplaceLayerNormModules(child)


def configure_huggingface_environment(output_dir: pathlib.Path) -> None:
    endpoint = os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
    os.environ.setdefault("HF_HOME", str(output_dir / ".hf_cache"))
    print(f"Hugging Face endpoint: {endpoint}")
    print(f"Hugging Face cache: {os.environ['HF_HOME']}")


def load_model(spec: Dict[str, object], output_dir: pathlib.Path) -> torch.nn.Module:
    from transformers import AutoModel, AutoModelForSequenceClassification

    model_id = str(spec["model_id"])
    cache_dir = output_dir / ".hf_cache"
    if spec["kind"] == "bert":
        model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=str(cache_dir))
    model.eval()
    return model


def export_model(spec: Dict[str, object], output_dir: pathlib.Path, force: bool) -> pathlib.Path:
    output_path = output_dir / str(spec["filename"])
    if output_path.exists() and not force:
        print(f"ready {output_path}")
        return output_path

    model = load_model(spec, output_dir)
    ReplaceLayerNormModules(model)
    wrapper = TransformerExportWrapper(model, str(spec["kind"]))
    wrapper.eval()

    input_names = [str(name) for name in spec["input_names"]]
    input_ids = torch.tensor([list(spec["input_ids"])], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    input_tensors: Tuple[torch.Tensor, ...]
    if input_names == ["input_ids", "attention_mask"]:
        input_tensors = (input_ids, attention_mask)
    else:
        input_tensors = (input_ids, attention_mask, token_type_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    print(f"exporting {spec['model_id']} -> {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            input_tensors,
            str(temporary_path),
            input_names=input_names,
            output_names=[str(spec["output_name"])],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            export_params=True,
        )
    temporary_path.replace(output_path)
    print(f"ready {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_SPECS], default="all")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_huggingface_environment(args.output_dir)
    names: List[str] = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for name in names:
        export_model(MODEL_SPECS[name], args.output_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
