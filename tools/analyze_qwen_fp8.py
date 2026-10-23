#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Rank local projection quantization error using BF16 calibration activations.

Each projection is replayed independently, with the final input vector from
each calibration document. This is a local sensitivity estimate, not an
end-to-end accuracy result. Candidate BF16 exclusions need held-out evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from export_qwen35_model import (FP8E4M3, FP8E5M2, FP8_NUMPY_DTYPE,
                                 fp8_scale_for_amax, fp8_scale_for_array,
                                 resolve_shard, source_to_numpy)
from qwen_fp8_calibration import load_calibration, file_sha256


def quantize(values, dtype, scale):
    # Saturating finite conversion matches the runtime's FP8 cast behavior.
    limit = 448 if dtype == FP8E4M3 else 57344
    scaled = np.clip(np.asarray(values, dtype=np.float32) / scale, -limit, limit)
    return scaled.astype(FP8_NUMPY_DTYPE[dtype]).astype(np.float32) * scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-top", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    path = args.calibration_dir / "calibration.json"
    raw = json.loads(path.read_text())
    calibration = load_calibration(path, args.model_dir, raw["format"])
    if calibration["checkpoint_sha256"] != file_sha256(resolve_shard(args.model_dir)):
        raise ValueError("calibration checkpoint differs")
    if args.keep_top < 0 or args.keep_top > len(calibration["projections"]):
        parser.error("keep-top exceeds projection count")
    if args.output.exists():
        raise ValueError("output must not exist")
    dtype = {"e4m3": FP8E4M3, "e5m2": FP8E5M2}[calibration["format"]]
    results = []
    with safe_open(resolve_shard(args.model_dir), framework="pt", device="cpu") as source, torch.inference_mode():
        for key, ranges in sorted(calibration["projections"].items()):
            weight = source_to_numpy(source.get_tensor(key)).astype(np.float32)
            samples = args.calibration_dir / "samples"
            x = np.fromfile(samples / (key + ".input.fp32"), dtype="<f4").reshape(-1, weight.shape[1])
            y = np.fromfile(samples / (key + ".output.fp32"), dtype="<f4").reshape(-1, weight.shape[0])
            if x.shape[0] != y.shape[0] or not x.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("invalid calibration activation sample")
            input_scale = fp8_scale_for_amax(dtype, ranges["input_amax"])
            output_scale = fp8_scale_for_amax(dtype, ranges["output_amax"])
            qx = quantize(x, dtype, input_scale)
            qw = quantize(weight, dtype, fp8_scale_for_array(dtype, weight))
            qt = torch.from_numpy(qx).to(device=args.device, dtype=torch.bfloat16)
            wt = torch.from_numpy(qw).to(device=args.device, dtype=torch.bfloat16)
            projected = (qt @ wt.T).float().cpu().numpy()
            actual = quantize(projected, dtype, output_scale)
            difference = actual.astype(np.float64) - y
            mse = float(np.mean(difference * difference))
            power = float(np.mean(y.astype(np.float64) ** 2))
            # Per-output-channel weights are evaluated here only. Existing
            # exported kernels implement per-tensor scales; do not export this.
            maximum = np.max(np.abs(weight), axis=1)
            scales = np.asarray([fp8_scale_for_amax(dtype, v) for v in maximum], dtype=np.float32)[:, None]
            per_channel_weight = quantize(weight, dtype, scales)
            channel_error = float(np.mean((per_channel_weight.astype(np.float64) - weight) ** 2))
            results.append({"projection": key, "sample_vectors": x.shape[0], "weight_elements": weight.size,
                            "relative_rmse": float(np.sqrt(mse / max(power, 1e-30))), "mse": mse,
                            "weight_per_tensor_mse": float(np.mean((qw.astype(np.float64) - weight) ** 2)),
                            "weight_per_channel_mse_analysis_only": channel_error})
            print(key, results[-1]["relative_rmse"], flush=True)
    results.sort(key=lambda item: item["relative_rmse"], reverse=True)
    candidate = dict(calibration)
    candidate["keep_bf16"] = [item["projection"] for item in results[:args.keep_top]]
    args.output.mkdir(parents=True)
    (args.output / "sensitivity.json").write_text(json.dumps({"method": "local_final_document_vectors_relative_rmse",
        "limitations": "Independent projections; no propagated quantization error, channel scales are analysis only.",
        "projections": results}, indent=2, allow_nan=False) + "\n")
    (args.output / "calibration.json").write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
