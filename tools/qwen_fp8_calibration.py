#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, strict per-projection Qwen FP8 calibration contract."""

import hashlib
import json
import math
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(path, model_dir, fp8_format):
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("format") != fp8_format:
        raise ValueError("calibration version/FP8 format does not match export")
    if data.get("model_config_sha256") != file_sha256(Path(model_dir) / "config.json"):
        raise ValueError("calibration model config hash does not match")
    for field in ("checkpoint_sha256", "corpus_sha256"):
        value = data.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"calibration requires a SHA256 {field}")
    projections = data.get("projections")
    if not isinstance(projections, dict) or not projections:
        raise ValueError("calibration requires nonempty projections")
    for key, ranges in projections.items():
        if not isinstance(key, str) or not isinstance(ranges, dict):
            raise ValueError("invalid calibration projection entry")
        for side in ("input", "output"):
            value = ranges.get(side + "_amax")
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid calibration {side}_amax for {key}")
            count = ranges.get(side + "_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"invalid calibration {side}_count for {key}")
    keep = data.get("keep_bf16", [])
    if not isinstance(keep, list) or any(not isinstance(item, str) or not item for item in keep):
        raise ValueError("calibration keep_bf16 must be a list of nonempty patterns")
    return data
