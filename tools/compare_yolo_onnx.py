#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compare actual Feather detections with ONNX Runtime on identical input pixels.

Example:
  python tools/compare_yolo_onnx.py --model models/yolo/yolov8/yolov8n.fth \
      --onnx models/yolo/yolov8/yolov8n.onnx --image assets/bus.jpg \
      --demo build_new/bin/yolo_demo --backend common cuda --output-dir /tmp/yolov8-compare

The demo dumps its actual prepared input as contiguous FP32 NCHW. ORT receives
that dump directly, including Feather's resize, padding, layout conversion and
FP16 rounding when applicable. Reports, raw values and annotated images remain
in --output-dir. No model export or package installation is performed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort


def decode(output, info, confidence, iou_threshold, maximum):
    """Decode the explicitly declared head using NumPy and class-aware NMS."""
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim != 3 or rows.shape[0] != 1:
        raise ValueError(f"Expected batch-one rank-three output, received {rows.shape}")
    rows = rows[0].T if info["output_layout"] == "bcn" else rows[0]
    output_format = info["output_format"]
    classes = info["num_classes"]
    if info["family"] == "yolov5" and classes == 0:
        classes = rows.shape[1] - 5
    expected = 6 if output_format == "xyxy_score_class" else classes + (5 if output_format == "cxcywh_objectness" else 4)
    if rows.shape[1] != expected:
        raise ValueError(f"Output has {rows.shape[1]} channels; metadata requires {expected}")
    valid = np.isfinite(rows).all(axis=1)
    if output_format == "xyxy_score_class":
        boxes, scores, labels = rows[:, :4].copy(), rows[:, 4], rows[:, 5]
        valid &= (labels >= 0) & (labels < classes) & (labels == np.floor(labels))
    else:
        class_offset = 5 if output_format == "cxcywh_objectness" else 4
        class_scores = rows[:, class_offset:]
        labels = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(rows)), labels]
        valid &= scores >= 0
        if output_format == "cxcywh_objectness":
            valid &= rows[:, 4] >= 0
            scores = scores * rows[:, 4]
        valid &= (rows[:, 2] > 0) & (rows[:, 3] > 0)
        boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                                rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    valid &= np.isfinite(scores) & (scores >= confidence) & np.isfinite(boxes).all(axis=1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid].astype(np.int64)
    order = np.argsort(-scores, kind="stable")
    selected, result = [], []
    scale = info["letterbox"]["scale"]
    offset = np.array([info["letterbox"]["pad_x"], info["letterbox"]["pad_y"]] * 2, dtype=np.float64)
    limits = np.array(info["image_size"] * 2, dtype=np.float64)
    # Legacy YOLOv5 performs NMS on clipped coordinates; new heads apply NMS
    # in model coordinates before reversing the letterbox transform.
    if info["family"] == "yolov5":
        boxes = np.clip((boxes - offset) / scale, 0, limits).astype(np.float32)
    for index in order:
        if info["nms_required"] and selected:
            previous = np.asarray(selected, dtype=np.int64)
            previous = previous[labels[previous] == labels[index]]
            if len(previous):
                a, b = boxes[index].astype(np.float64), boxes[previous].astype(np.float64)
                wh = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]))
                overlap = wh[:, 0] * wh[:, 1]
                area_a = np.prod(a[2:] - a[:2])
                area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
                union = area_a + area_b - overlap
                iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
                if np.any(iou > iou_threshold):
                    continue
        selected.append(index)
        box = boxes[index] if info["family"] == "yolov5" else np.clip((boxes[index] - offset) / scale, 0, limits)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result.append({"class_id": int(labels[index]), "score": float(scores[index]), "box": box.tolist()})
        if len(result) == maximum:
            break
    return result


def compare_detections(actual, expected, score_atol, box_atol):
    remaining = set(range(len(expected)))
    mismatches = []
    max_box_error = max_score_error = 0.0
    for detection in actual:
        candidates = [index for index in remaining if expected[index]["class_id"] == detection["class_id"]]
        if not candidates:
            mismatches.append({"unmatched_feather": detection})
            continue
        # Score and box comparisons tolerate a different order for tied scores.
        index = min(candidates, key=lambda i: float(np.max(np.abs(np.asarray(detection["box"]) - expected[i]["box"]))))
        box_error = float(np.max(np.abs(np.asarray(detection["box"]) - expected[index]["box"])))
        score_error = abs(detection["score"] - expected[index]["score"])
        max_box_error, max_score_error = max(max_box_error, box_error), max(max_score_error, score_error)
        remaining.remove(index)
        if box_error > box_atol or score_error > score_atol:
            mismatches.append({"feather": detection, "ort": expected[index], "box_error": box_error, "score_error": score_error})
    mismatches.extend({"unmatched_ort": expected[index]} for index in sorted(remaining))
    return {"passed": not mismatches, "feather_count": len(actual), "ort_count": len(expected),
            "max_box_abs_error": max_box_error, "max_score_abs_error": max_score_error,
            "mismatches": mismatches}


def compare_end_to_end_rows(actual, reference, layout, num_classes, atol, rtol, rank_score_atol):
    """Match every final row, including scores below the display threshold.

    Nearly tied TopK scores may exchange ranks between runtimes. Class IDs
    must still match exactly, every box and score must pass the raw tolerances,
    and the complete score sequence must preserve descending rank order.
    """
    actual_rows = actual[0].T if layout == "bcn" else actual[0]
    reference_rows = reference[0].T if layout == "bcn" else reference[0]
    if actual_rows.shape != reference_rows.shape or actual_rows.ndim != 2 or actual_rows.shape[1] != 6:
        return {"passed": False, "error": "Expected equal batch-one [N,6] final rows"}
    finite = bool(np.isfinite(actual_rows).all() and np.isfinite(reference_rows).all())
    if not finite:
        return {"passed": False, "error": "Final rows contain nonfinite values"}
    valid_classes = bool(np.all(actual_rows[:, 5] == np.floor(actual_rows[:, 5])) and
                         np.all(reference_rows[:, 5] == np.floor(reference_rows[:, 5])) and
                         np.all((actual_rows[:, 5] >= 0) & (actual_rows[:, 5] < num_classes)) and
                         np.all((reference_rows[:, 5] >= 0) & (reference_rows[:, 5] < num_classes)))
    actual_sorted = bool(np.all(np.diff(actual_rows[:, 4].astype(np.float64)) <= rank_score_atol))
    reference_sorted = bool(np.all(np.diff(reference_rows[:, 4].astype(np.float64)) <= rank_score_atol))
    rank_scores_close = bool(np.isclose(actual_rows[:, 4], reference_rows[:, 4], atol=rank_score_atol, rtol=rtol).all())
    compatible = (actual_rows[:, None, 5] == reference_rows[None, :, 5]) & np.isclose(
        actual_rows[:, None, :5], reference_rows[None, :, :5], atol=atol, rtol=rtol).all(axis=2)
    # A perfect bipartite match avoids rejecting a valid permutation when a
    # greedy choice consumes another row's only compatible counterpart.
    adjacency = []
    for index, row in enumerate(compatible):
        candidates = np.flatnonzero(row).tolist()
        candidates.sort(key=lambda other: (other != index,
                        float(np.max(np.abs(actual_rows[index, :5] - reference_rows[other, :5])))))
        adjacency.append(candidates)
    owner = [-1] * len(reference_rows)

    def assign(index, visited):
        for other in adjacency[index]:
            if visited[other]:
                continue
            visited[other] = True
            if owner[other] < 0 or assign(owner[other], visited):
                owner[other] = index
                return True
        return False

    unmatched_actual = []
    for index in range(len(actual_rows)):
        if not assign(index, [False] * len(reference_rows)):
            unmatched_actual.append(index)
    permutation = [-1] * len(actual_rows)
    for other, index in enumerate(owner):
        if index >= 0:
            permutation[index] = other
    complete = not unmatched_actual
    max_matched_error = max_matched_score_error = None
    if complete:
        delta = np.abs(actual_rows[:, :5].astype(np.float64) - reference_rows[permutation, :5].astype(np.float64))
        max_matched_error = float(delta.max()) if delta.size else 0.0
        max_matched_score_error = float(delta[:, 4].max()) if delta.size else 0.0
    return {"passed": complete and valid_classes and actual_sorted and reference_sorted and rank_scores_close,
            "rows_compared": len(actual_rows), "valid_integer_classes": valid_classes,
            "feather_scores_nonincreasing": actual_sorted, "ort_scores_nonincreasing": reference_sorted,
            "rank_scores_close": rank_scores_close, "rank_score_atol": rank_score_atol,
            "complete_row_match": complete, "max_matched_abs_error": max_matched_error,
            "max_matched_score_abs_error": max_matched_score_error,
            "reference_row_for_each_feather_row": permutation, "unmatched_feather_rows": unmatched_actual,
            "unmatched_ort_rows": [index for index, value in enumerate(owner) if value < 0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--demo", type=Path, default=Path("build_new/bin/yolo_demo"))
    parser.add_argument("--backend", nargs="+", choices=("host", "common", "x86", "cuda"), default=["common"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("auto", "nchw", "nhwc"), default="auto")
    parser.add_argument("--ort-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--provenance", type=Path, help="Optional exporter provenance.json; discovered beside ONNX by default")
    parser.add_argument("--reuse", action="store_true", help="Compare existing backend.json/input dumps without rerunning Feather")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--raw-atol", type=float, default=1e-3)
    parser.add_argument("--raw-rtol", type=float, default=1e-3)
    parser.add_argument("--rank-score-atol", type=float, default=1e-6,
                        help="Tiny allowed score-order reversal and absolute rank-score tolerance for final rows")
    parser.add_argument("--score-atol", type=float, default=1e-3)
    parser.add_argument("--box-atol", type=float, default=0.1, help="Detection coordinate tolerance in original image pixels")
    args = parser.parse_args()
    if not (0 <= args.conf_thresh <= 1 and 0 <= args.iou_thresh <= 1 and args.max_detections > 0):
        parser.error("thresholds must be in [0,1] and max-detections must be positive")
    if any(not np.isfinite(t) or t < 0 for t in (args.raw_atol, args.raw_rtol, args.rank_score_atol, args.score_atol, args.box_atol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider = "CPUExecutionProvider" if args.ort_provider == "cpu" else "CUDAExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested ORT provider unavailable: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=[provider])
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"ORT could not initialize the requested provider: {provider}")
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Comparison requires one image input and one decoded detection output")
    model_input = session.get_inputs()[0]
    if model_input.type not in ("tensor(float)", "tensor(float16)"):
        raise ValueError(f"Unsupported ORT image dtype: {model_input.type}")
    provenance_path = args.provenance or args.onnx.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    reports = {}
    for backend in dict.fromkeys(args.backend):
        json_path, input_path = args.output_dir / f"{backend}.json", args.output_dir / f"{backend}.input.bin"
        command = [str(args.demo.resolve()), "--model", str(args.model.resolve()), "--image", str(args.image.resolve()),
                   "--backend", backend, "--layout", args.layout, "--output", str((args.output_dir / f"{backend}.jpg").resolve()),
                   "--json", str(json_path.resolve()), "--dump-input", str(input_path.resolve()),
                   "--conf-thresh", str(args.conf_thresh), "--iou-thresh", str(args.iou_thresh), "--max-detections", str(args.max_detections)]
        if not args.reuse:
            with (args.output_dir / f"{backend}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        info = json.loads(json_path.read_text())
        if provenance:
            for key, value in provenance["model_metadata"].items():
                name = key.removeprefix("yolo.")
                if name in ("family", "num_classes", "output_format", "output_layout", "nms_required") and info[name] != value:
                    raise ValueError(f"Feather metadata disagrees with exporter provenance for {key}")
        image_input = np.fromfile(input_path, dtype="<f4").reshape(info["input_shape"])
        if list(image_input.shape) != model_input.shape:
            raise ValueError(f"Dump NCHW shape {image_input.shape} disagrees with ONNX input {model_input.shape}")
        ort_input = image_input.astype(np.float16 if model_input.type == "tensor(float16)" else np.float32)
        reference = session.run(None, {model_input.name: ort_input})[0]
        actual = np.asarray(info["raw_output"]["data"], dtype=np.float32).reshape(info["raw_output"]["shape"])
        np.save(args.output_dir / f"{backend}.ort.npy", reference)
        if actual.shape != reference.shape:
            raise ValueError(f"Output shape differs: Feather {actual.shape}, ORT {reference.shape}")
        finite = np.isfinite(actual) & np.isfinite(reference)
        delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        close = np.isclose(actual, reference, atol=args.raw_atol, rtol=args.raw_rtol)
        raw = {"passed": bool(close.all() and finite.all()), "shape": list(actual.shape),
               "max_abs_error": float(delta[finite].max()) if finite.any() else None,
               "mean_abs_error": float(delta[finite].mean()) if finite.any() else None,
               "close_fraction": float(close.mean()), "nonfinite_values": int(actual.size - finite.sum())}
        raw["ordered_passed"] = raw["passed"]
        if info["output_format"] == "xyxy_score_class":
            matching = compare_end_to_end_rows(actual, reference, info["output_layout"], info["num_classes"],
                                               args.raw_atol, args.raw_rtol, args.rank_score_atol)
            raw["permutation_aware"] = matching
            raw["passed"] = matching["passed"]
            raw["row_order_differs"] = matching["passed"] and not raw["ordered_passed"]
        reference_detections = decode(reference, info, args.conf_thresh, args.iou_thresh, args.max_detections)
        detections = compare_detections(info["detections"], reference_detections, args.score_atol, args.box_atol)
        report = {"model": str(args.model), "onnx": str(args.onnx), "backend": backend, "ort_provider": provider,
                  "input_shape": list(image_input.shape), "raw": raw, "detections": detections,
                  "passed": raw["passed"] and detections["passed"]}
        (args.output_dir / f"{backend}.comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[backend] = report
        print(json.dumps(report), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(report["passed"] for report in reports.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"YOLO comparison failed: {error}", file=sys.stderr)
        sys.exit(1)
