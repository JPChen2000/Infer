#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible Qwen teacher-forced scoring, calibration and generation timing.

JSONL records contain unique id and either text or messages:[{role,content}].
No token is added for plain text. Messages use the repository's non-thinking
chat template; benchmark appends the assistant generation prefix. The bundled
corpora are authored smoke data, not standardized language benchmarks.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import datetime
from pathlib import Path
import statistics
import subprocess

import numpy as np
from compare_qwen_logits import compare, load_logits
from qwen_fp8_calibration import file_sha256


GENERATION_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def render(record, mode):
    if ("text" in record) == ("messages" in record):
        raise ValueError("each record must contain exactly one of text or messages")
    if "text" in record:
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError("text must be a nonempty string")
        return record["text"]
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be nonempty")
    result = ""
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("unsupported message role")
        role, content = message["role"], message.get("content")
        if (role == "system" and index != 0) or not isinstance(content, str):
            raise ValueError("invalid message content or system position")
        content = content.strip(" \t\n\r\v\f")
        result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if mode == "benchmark":
        if messages[-1]["role"] != "user":
            raise ValueError("benchmark messages must end with a user turn")
        result += GENERATION_PREFIX
    return result


def read_corpus(path):
    records = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {number}")
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError(f"record {number} requires a nonempty string id")
        if record["id"] in seen:
            raise ValueError(f"duplicate record id {record['id']}")
        seen.add(record["id"])
        records.append(record)
    if not records:
        raise ValueError("empty corpus")
    return records


def table(path):
    with Path(path).open(encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def score_summary(rows, documents):
    if not rows:
        raise ValueError("no scored tokens")
    expected = [(d, p, doc["tokens"][p]) for d, doc in enumerate(documents)
                for p in range(1, len(doc["tokens"]))]
    actual = [(int(r["document"]), int(r["position"]), int(r["target"])) for r in rows]
    if actual != expected:
        raise ValueError("score rows do not match token manifest")
    nlls = [float(row["nll"]) for row in rows]
    if any(not math.isfinite(v) or v < -1e-9 for v in nlls):
        raise ValueError("invalid token loss")
    total = math.fsum(nlls)
    mean = total / len(rows)
    if mean > 700:
        raise ValueError("perplexity overflow; model loss exceeds supported report range")
    per_document = []
    for d, document in enumerate(documents):
        values = [float(row["nll"]) for row in rows if int(row["document"]) == d]
        loss = math.fsum(values) / len(values)
        per_document.append({"id": document["id"], "category": document["category"],
                             "scored_tokens": len(values), "mean_nll": loss,
                             "perplexity": math.exp(loss)})
    return {"scored_tokens": len(rows), "total_nll": total, "mean_nll": mean,
            "perplexity": math.exp(mean),
            "target_top1_accuracy": sum(int(r["target"]) == int(r["greedy"]) for r in rows) / len(rows),
            "per_document": per_document}


def benchmark_summary(rows):
    if not rows:
        raise ValueError("empty benchmark result")
    result = []
    for document in sorted({int(row["document"]) for row in rows}):
        samples = [r for r in rows if int(r["document"]) == document]
        cold = [r for r in samples if int(r["run"]) == 0]
        warm = [r for r in samples if int(r["run"]) > 0]
        if len(cold) != 1 or not warm:
            raise ValueError("benchmark requires first-use and warmed measurements")
        result.append({"document": document, "cold_ttft_ms": float(cold[0]["ttft_ms"]),
                       "warm_ttft_median_ms": statistics.median(float(r["ttft_ms"]) for r in warm),
                       "warm_prefill_median_ms": statistics.median(float(r["prefill_ms"]) for r in warm),
                       "warm_decode_ms_per_token": sum(float(r["decode_ms"]) for r in warm) /
                                                   sum(int(r["decode_steps"]) for r in warm),
                       "warm_decode_tokens_per_second": 1000 * sum(int(r["decode_steps"]) for r in warm) /
                                                               sum(float(r["decode_ms"]) for r in warm),
                       "warm_repeats": len(warm)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build_new/bin/qwen_eval"))
    parser.add_argument("--mode", choices=("score", "calibrate", "benchmark"), default="score")
    parser.add_argument("--backend", choices=("cuda", "common", "x86"), default="cuda")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--fp8-weight-cache-mb", type=int, default=256, help="CUDA FP8 fallback BF16 cache budget; 0 disables")
    parser.add_argument("--max-tokens", type=int, help="Explicit per-document prefix limit; truncation is recorded")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", type=Path, help="Matching BF16 score run directory")
    parser.add_argument("--max-relative-ppl-increase", type=float, help="Fail if held-out PPL increases beyond this fraction")
    parser.add_argument("--fp8-format", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--range-margin", type=float, default=1.1)
    args = parser.parse_args()
    if args.fp8_weight_cache_mb < 0 or args.fp8_weight_cache_mb > (2**64 - 1) // (1024 * 1024):
        parser.error("fp8-weight-cache-mb must be nonnegative and fit size_t")
    if args.context < 0 or (args.max_tokens is not None and args.max_tokens < 2):
        parser.error("context must be nonnegative and max-tokens at least two")
    if args.mode != "score" and args.reference:
        parser.error("reference comparisons require score mode")
    if args.max_relative_ppl_increase is not None and (not args.reference or
            not math.isfinite(args.max_relative_ppl_increase) or args.max_relative_ppl_increase < 0):
        parser.error("a finite nonnegative PPL gate requires --reference")
    if not math.isfinite(args.range_margin) or args.range_margin < 1:
        parser.error("range-margin must be finite and at least one")
    from tokenizers import Tokenizer
    tokenizer_path = args.tokenizer / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    records = read_corpus(args.dataset)
    documents = []
    for record in records:
        text = render(record, args.mode)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        count = len(ids)
        if args.max_tokens is not None:
            ids = ids[:args.max_tokens]
        if len(ids) < 2:
            raise ValueError(f"document {record['id']} has fewer than two tokens")
        documents.append({"id": record["id"], "category": record.get("category", "unspecified"),
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "original_tokens": count, "truncated": count != len(ids), "tokens": ids})
    manifest = {"version": 1, "corpus_sha256": canonical_hash(records),
                "tokenizer_sha256": file_sha256(tokenizer_path), "tokens_sha256": canonical_hash(documents),
                "documents": documents, "chat_template": "repository_non_thinking"}
    if args.reference:
        ref_manifest = json.loads((args.reference / "manifest.json").read_text())
        if ref_manifest["tokens_sha256"] != manifest["tokens_sha256"]:
            raise ValueError("reference token manifest differs")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be new or empty; preserve previous measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tokens_path = args.output / "tokens.txt"
    tokens_path.write_text("".join(" ".join(map(str, d["tokens"])) + "\n" for d in documents))
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()), "--input", str(tokens_path.resolve()),
               "--output", str(args.output.resolve()), "--mode", args.mode, "--backend", args.backend,
               "--context", str(args.context), "--new-tokens", str(args.new_tokens), "--repeats", str(args.repeats),
               "--fp8-weight-cache-mb", str(args.fp8_weight_cache_mb)]
    with (args.output / "process.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    runtime = json.loads((args.output / "runtime.json").read_text())
    result = {"version": 1, "mode": args.mode, "model": str(args.model),
              "model_sha256": file_sha256(args.model), "dataset": str(args.dataset),
              "corpus_sha256": manifest["corpus_sha256"], "tokens_sha256": manifest["tokens_sha256"],
              "backend": args.backend, "command": command, "runtime": runtime,
              "binary_sha256": file_sha256(args.binary), "platform": platform.platform(),
              "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Corpus-specific measurement. Bundled data is an authored smoke corpus, not a standard benchmark."}
    if args.backend == "cuda":
        try:
            info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                  text=True, capture_output=True, check=True)
            result["gpu"] = info.stdout.strip().splitlines()
        except (OSError, subprocess.CalledProcessError):
            result["gpu"] = []
    passed = True
    if args.mode == "score":
        rows = table(args.output / "scores.tsv")
        result["quality"] = score_summary(rows, documents)
        if args.reference:
            ref_rows = table(args.reference / "scores.tsv")
            reference = score_summary(ref_rows, documents)
            delta = result["quality"]["mean_nll"] - reference["mean_nll"]
            relative = math.expm1(delta)
            comparisons = []
            for index, document in enumerate(documents):
                actual_path, ref_path = args.output / f"final_{index}.fp32", args.reference / f"final_{index}.fp32"
                comparisons.append({"id": document["id"], **compare(load_logits(actual_path, "fp32"), load_logits(ref_path, "fp32"))})
            result["reference"] = {"directory": str(args.reference), "mean_nll_delta": delta,
                                    "relative_perplexity_increase": relative,
                                    "greedy_agreement": sum(a["greedy"] == b["greedy"] for a, b in zip(rows, ref_rows)) / len(rows),
                                    "final_logits": comparisons}
            if args.max_relative_ppl_increase is not None:
                passed = relative <= args.max_relative_ppl_increase
                result["acceptance"] = {"max_relative_ppl_increase": args.max_relative_ppl_increase, "passed": passed}
    elif args.mode == "calibrate":
        from export_qwen35_model import resolve_shard
        projections = {}
        for row in table(args.output / "ranges.tsv"):
            ranges = projections.setdefault(row["projection"], {})
            side = row["side"]
            if side not in ("input", "output") or side + "_amax" in ranges:
                raise ValueError("invalid/duplicate calibration range")
            ranges[side + "_amax"] = max(float(row["amax"]) * args.range_margin, 1e-30)
            ranges[side + "_count"] = int(row["count"])
            ranges[side + "_rms"] = float(row["rms"])
        config_hash = file_sha256(args.tokenizer / "config.json")
        checkpoint_hash = file_sha256(resolve_shard(args.tokenizer))
        if runtime.get("model_config_sha256") != config_hash or runtime.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("calibration source graph does not match tokenizer/model-dir checkpoint")
        calibration = {"version": 1, "format": args.fp8_format, "method": "amax",
                       "range_margin": args.range_margin, "model_config_sha256": file_sha256(args.tokenizer / "config.json"),
                       "checkpoint_sha256": checkpoint_hash,
                       "corpus_sha256": manifest["corpus_sha256"], "tokens_sha256": manifest["tokens_sha256"],
                       "observed_tokens": runtime["tokens"], "projections": projections, "keep_bf16": []}
        (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
        result["calibration"] = {"projections": len(projections), "path": str(args.output / "calibration.json")}
    else:
        result["performance"] = benchmark_summary(table(args.output / "benchmark.tsv"))
        result["memory_note"] = "Device-wide usage sampled after runs; includes other processes and is not allocator/process peak."
        result["timing_note"] = "Run 0 includes first-use plan preparation. Warm runs retain plans but Reset recreates CUDA captures. Decode intervals are callback gaps, excluding the final unobserved feed."
        for index, document in enumerate(documents):
            ids = [int(t) for t in (args.output / f"generated_{index}.txt").read_text().split()]
            result["performance"][index]["id"] = document["id"]
            result["performance"][index]["prompt_tokens"] = len(document["tokens"])
            result["performance"][index]["generated_tokens"] = ids
            result["performance"][index]["generated_text"] = tokenizer.decode(ids, skip_special_tokens=False)
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("mode", "quality", "calibration", "performance", "acceptance")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
