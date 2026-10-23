#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compare one final-token logit vector from identical teacher-forced tokens.

Example (the raw dump is little-endian BF16, the reference is a NumPy array):
  python3 -B tools/compare_qwen_logits.py --actual runtime.logits.bin \
      --reference hf.npy --output comparison.json

This measures one vector; it does not evaluate a dataset or generation quality.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_logits(path, data_format):
    if data_format == "npy":
        values = np.load(path, allow_pickle=False)
    elif data_format == "bf16":
        if Path(path).stat().st_size % 2:
            raise ValueError("BF16 dump must contain complete 16-bit values")
        words = np.fromfile(path, dtype="<u2").astype("<u4")
        values = (words << 16).view("<f4")
    else:
        if Path(path).stat().st_size % 4:
            raise ValueError("FP32 dump must contain complete 32-bit values")
        values = np.fromfile(path, dtype="<f4")
    # Accept [vocab], [1,vocab], or [1,1,vocab], but not a sequence/batch.
    if values.ndim == 0 or any(size != 1 for size in values.shape[:-1]):
        raise ValueError("expected exactly one final-token logit vector")
    values = values.astype(np.float64).reshape(-1)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("logits must be nonempty and finite")
    return values


def compare(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"logit shapes differ: {actual.shape} vs {reference.shape}")
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(reference))
    if denominator == 0:
        raise ValueError("cosine similarity requires nonzero vectors")
    difference = actual - reference
    actual_token, reference_token = int(np.argmax(actual)), int(np.argmax(reference))
    return {
        "vocabulary_size": int(actual.size),
        "cosine_similarity": float(np.dot(actual, reference) / denominator),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "actual_greedy_token": actual_token,
        "reference_greedy_token": reference_token,
        "greedy_equal": actual_token == reference_token,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual-format", choices=("bf16", "fp32", "npy"), default="bf16")
    parser.add_argument("--reference-format", choices=("bf16", "fp32", "npy"), default="npy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(load_logits(args.actual, args.actual_format),
                         load_logits(args.reference, args.reference_format))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
