#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
import json
import math
from pathlib import Path
import tempfile
import unittest

from evaluate_qwen import score_summary, render
from qwen_fp8_calibration import load_calibration, file_sha256


class EvaluationTest(unittest.TestCase):
    def test_perplexity_weights_tokens_and_validates_teacher_forcing_alignment(self):
        docs = [{"id": "short", "category": "a", "tokens": [0, 1]},
                {"id": "long", "category": "b", "tokens": [0, 1, 2, 3]}]
        rows = [{"document": d, "position": p, "target": target, "greedy": target, "nll": loss}
                for d, p, target, loss in [(0, 1, 1, 1), (1, 1, 1, 3), (1, 2, 2, 3), (1, 3, 3, 3)]]
        result = score_summary(rows, docs)
        self.assertEqual(result["mean_nll"], 2.5)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.5))
        rows[-1]["target"] = 2
        with self.assertRaisesRegex(ValueError, "manifest"):
            score_summary(rows, docs)

    def test_chat_rendering_scores_existing_turns_and_benchmarks_only_a_user_suffix(self):
        record = {"messages": [{"role": "user", "content": "  你好\n"}]}
        self.assertEqual(render(record, "score"), "<|im_start|>user\n你好<|im_end|>\n")
        self.assertTrue(render(record, "benchmark").endswith("</think>\n\n"))
        with self.assertRaisesRegex(ValueError, "end with a user"):
            render({"messages": [{"role": "assistant", "content": "answer"}]}, "benchmark")

    def test_calibration_rejects_wrong_model_nonfinite_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            path = root / "calibration.json"
            data = {"version": 1, "format": "e4m3", "model_config_sha256": file_sha256(root / "config.json"),
                    "checkpoint_sha256": "a" * 64, "corpus_sha256": "b" * 64,
                    "projections": {"projection": {"input_amax": 1, "output_amax": 2,
                                                   "input_count": 12, "output_count": 16}}}
            path.write_text(json.dumps(data))
            self.assertEqual(load_calibration(path, root, "e4m3"), data)
            for key, value in [("input_amax", float("nan")), ("output_amax", -1), ("input_count", 0)]:
                invalid = json.loads(json.dumps(data))
                invalid["projections"]["projection"][key] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_calibration(path, root, "e4m3")
            path.write_text(json.dumps(data))
            (root / "config.json").write_text('{"other":1}')
            with self.assertRaisesRegex(ValueError, "config hash"):
                load_calibration(path, root, "e4m3")


if __name__ == "__main__":
    unittest.main()
