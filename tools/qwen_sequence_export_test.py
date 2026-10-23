#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the single Qwen semantic graph."""
import json
import contextlib
import io
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

import export_qwen35_model as atomic
from export_qwen35_optimized import OptimizedQwenBuilder


class TinySource:
    """Small deterministic safetensors-shaped source; no checkpoint needed."""
    def __init__(self):
        self.tensors = {
            "model.language_model.embed_tokens.weight": torch.ones((8, 4), dtype=torch.bfloat16),
            "model.language_model.norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        }
        for layer in range(2):
            prefix = f"model.language_model.layers.{layer}"
            for name in ("input_layernorm", "post_attention_layernorm"):
                self.tensors[f"{prefix}.{name}.weight"] = torch.zeros(4, dtype=torch.bfloat16)
            for name, shape in (("gate_proj", (6, 4)), ("up_proj", (6, 4)), ("down_proj", (4, 6))):
                self.tensors[f"{prefix}.mlp.{name}.weight"] = torch.ones(shape, dtype=torch.bfloat16)
        for name, shape in {"conv1d.weight": (12, 1, 4), "in_proj_qkv.weight": (12, 4),
                            "in_proj_z.weight": (4, 4), "in_proj_a.weight": (2, 4),
                            "in_proj_b.weight": (2, 4), "out_proj.weight": (4, 4),
                            "dt_bias": (2,)}.items():
            self.tensors["model.language_model.layers.0.linear_attn." + name] = torch.ones(shape, dtype=torch.bfloat16)
        for name in ("A_log", "norm.weight"):
            self.tensors["model.language_model.layers.0.linear_attn." + name] = torch.ones(2, dtype=torch.float32)
        for name, shape in {"q_proj": (16, 4), "k_proj": (4, 4), "v_proj": (4, 4),
                            "o_proj": (4, 8), "q_norm": (4,), "k_norm": (4,)}.items():
            self.tensors[f"model.language_model.layers.1.self_attn.{name}.weight"] = torch.ones(shape, dtype=torch.bfloat16)

    def get_slice(self, name):
        tensor = self.tensors[name]
        class Slice:
            def get_shape(self):
                return list(tensor.shape)
            def get_dtype(self):
                return "BF16" if tensor.dtype == torch.bfloat16 else "F32"
        return Slice()

    def get_tensor(self, name):
        return self.tensors[name]


TEXT = {
    "max_position_embeddings": 64, "hidden_size": 4, "intermediate_size": 6,
    "num_hidden_layers": 2, "vocab_size": 8, "head_dim": 4,
    "num_attention_heads": 2, "num_key_value_heads": 1,
    "linear_num_value_heads": 2, "linear_num_key_heads": 2,
    "linear_key_head_dim": 2, "linear_value_head_dim": 2,
    "linear_conv_kernel_dim": 4, "layer_types": ["linear_attention", "full_attention"],
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0}, "rms_norm_eps": 1e-6,
}


class SequenceExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.directory.name)
        (self.model_dir / "config.json").write_text(json.dumps({"text_config": TEXT}))
        self.source = TinySource()

    def tearDown(self):
        self.directory.cleanup()

    def build(self, sequence=1, dtype=None, context=8, **kwargs):
        builder = OptimizedQwenBuilder(self.model_dir, context, sequence=sequence,
                                       prefill_chunk=8, fp8_dtype=dtype, **kwargs)
        builder.source = self.source
        builder.build()
        return builder

    def test_atomic_cli_records_source_fingerprints(self):
        from safetensors.torch import save_file
        from qwen_fp8_calibration import file_sha256
        shard = self.model_dir / "model.safetensors"
        save_file(self.source.tensors, str(shard))
        output = self.model_dir / "atomic.fth"
        with patch("sys.argv", ["export_qwen35_model.py", "--model-dir", str(self.model_dir),
                                "--max-context", "8", "--output-fth", str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            atomic.main()
        data = output.read_bytes()
        for source in (shard, self.model_dir / "config.json"):
            self.assertIn(file_sha256(source).encode(), data)
        self.assertIn(b"qwen.source.checkpoint_sha256", data)
        self.assertIn(b"qwen.source.config_sha256", data)

    def test_calibration_uses_independent_input_output_scales_and_keeps_selected_weight_original(self):
        base = self.build()
        calibration = {"projections": {key: {"input_amax": 0.5, "output_amax": 32.0}
                                       for key in base.projection_keys}}
        kept = "model.language_model.layers.0.linear_attn.out_proj.weight"
        builder = self.build(dtype=atomic.FP8E4M3, fp8_calibration=calibration, fp8_keep_bf16=[kept])
        builder.validate_precision_policy()
        graph = builder.g
        name = "weight_" + kept.replace(".", "_") + "_t"
        self.assertEqual(graph.value_info[name][1], atomic.BF16)
        self.assertNotIn(name + "__fp8e4m3", graph.constants)
        source = "model.language_model.layers.0.mlp.gate_proj.weight"
        weight = "weight_" + source.replace(".", "_") + "_t__fp8e4m3"
        node = next(node for node in graph.nodes if node.op_type == "MatMul" and node.inputs[1] == weight)
        self.assertEqual(graph.quantization_scales[node.inputs[0]], atomic.fp8_scale_for_amax(atomic.FP8E4M3, 0.5))
        self.assertEqual(graph.quantization_scales[node.outputs[0]], atomic.fp8_scale_for_amax(atomic.FP8E4M3, 32))

    def test_calibration_rejects_missing_extra_projection_and_unknown_keep_pattern(self):
        base = self.build()
        ranges = {key: {"input_amax": 1, "output_amax": 2} for key in base.projection_keys}
        missing = dict(ranges)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "missing projection"):
            self.build(dtype=atomic.FP8E4M3, fp8_calibration={"projections": missing})
        builder = self.build(dtype=atomic.FP8E4M3,
                             fp8_calibration={"projections": {**ranges, "unknown": {}}})
        with self.assertRaisesRegex(ValueError, "coverage"):
            builder.validate_precision_policy()
        builder = self.build(dtype=atomic.FP8E4M3, fp8_keep_bf16=["*.typo.weight"])
        with self.assertRaisesRegex(ValueError, "matches no"):
            builder.validate_precision_policy()

    def test_bf16_projection_metadata_identifies_actual_unquantized_values(self):
        builder = self.build()
        graph = builder.g
        for key in builder.projection_keys:
            source = graph.metadata["qwen.projection.input." + key]
            result = graph.metadata["qwen.projection.output." + key]
            self.assertEqual(graph.value_info[source][1], atomic.BF16)
            self.assertEqual(graph.value_info[result][1], atomic.BF16)
            node = next(node for node in graph.nodes if result in node.outputs)
            self.assertEqual(node.op_type, "MatMul")
            self.assertEqual(node.inputs[0], source)

    @staticmethod
    def rebound(graph, values, key, sequence):
        result = list(values)
        pairs = graph.metadata.get(key, [])
        for index, offset in zip(pairs[::2], pairs[1::2]):
            if result[index] != graph.metadata["qwen.sequence.default"] + offset:
                raise AssertionError(f"invalid binding {key}: {result}")
            result[index] = sequence + offset
        return result

    def test_explicit_sequence_symbols_include_scalar_attributes_and_cancel(self):
        t = atomic.SequenceDimension(1)
        self.assertEqual((t + 3) - t, 3)
        self.assertNotIsInstance((t + 3) - t, atomic.SequenceDimension)
        graph = atomic.AtomicGraph()
        graph.input("x", [1, t, 1], atomic.FP32)
        graph.node("Slice", ["x"], "last", [[1, 1, 1]], atomic.FP32,
                   name="last", start=t - 1, end=t, axis=1)
        self.assertEqual(graph.metadata["qwen.sequence.value.x"], [1, 0])
        self.assertEqual(graph.metadata["qwen.sequence.attribute.last.start"], [0, -1])
        self.assertEqual(graph.metadata["qwen.sequence.attribute.last.end"], [0, 0])
        self.assertNotIn("qwen.sequence.value.last", graph.metadata)

    def test_all_lengths_specialize_one_identical_graph_in_bf16_and_fp8(self):
        for dtype in (None, atomic.FP8E4M3, atomic.FP8E5M2):
            canonical = self.build(dtype=dtype)
            graph = canonical.g
            self.assertEqual(graph.metadata["qwen.execution.version"], 2)
            self.assertEqual(graph.metadata["qwen.sequence.maximum"], 8)
            self.assertFalse(any(key.startswith("qwen.prefill.") for key in graph.metadata))
            recurrent = [node for node in graph.nodes if node.op_type == "QwenGatedDeltaSequence"]
            self.assertEqual(len(recurrent), 1)
            self.assertFalse(any(node.op_type == "QwenGatedDelta" for node in graph.nodes))
            for sequence in (2, 3, 4, 7, 8):
                direct = self.build(sequence, dtype=dtype).g
                self.assertEqual(set(graph.value_info), set(direct.value_info))
                self.assertEqual(len(graph.nodes), len(direct.nodes))
                for name, (shape, value_dtype) in graph.value_info.items():
                    rebound = self.rebound(graph, shape, "qwen.sequence.value." + name, sequence)
                    self.assertEqual(rebound, direct.value_info[name][0], msg=name)
                    self.assertEqual(value_dtype, direct.value_info[name][1])
                for name, array in graph.constant_values.items():
                    if name == "qwen_chunk_offsets":
                        self.assertEqual(graph.metadata["qwen.sequence.arange." + name], 1)
                        expected = np.arange(sequence, dtype=np.float32)
                    else:
                        expected = np.asarray(self.rebound(graph, array.reshape(-1),
                                              "qwen.sequence.constant." + name, sequence)).reshape(array.shape)
                    np.testing.assert_array_equal(expected, direct.constant_values[name], err_msg=name)
                for old, new in zip(graph.nodes, direct.nodes):
                    self.assertEqual((old.name, old.op_type, old.inputs, old.outputs),
                                     (new.name, new.op_type, new.inputs, new.outputs))
                    for name, value in old.attributes.items():
                        key = f"qwen.sequence.attribute.{old.name}.{name}"
                        if isinstance(value, (list, tuple)):
                            value = self.rebound(graph, value, key, sequence)
                        elif key in graph.metadata:
                            value = self.rebound(graph, [value], key, sequence)[0]
                        self.assertEqual(value, new.attributes[name], msg=key)
            # The standard writer has precisely one graph (v4 has no variant table).
            output = self.model_dir / f"unified_{dtype}.fth"
            canonical.write_fth(output)
            self.assertEqual(struct.unpack("<8sIIQ", output.read_bytes()[:24])[1], 4)

    def test_context_bindings_remain_independent_at_equal_default_sizes(self):
        small = self.build(sequence=8, context=8).g
        large = self.build(sequence=8, context=16).g
        for name, (shape, _) in small.value_info.items():
            result = list(shape)
            pairs = small.metadata.get("qwen.context.value." + name, [])
            for index, offset in zip(pairs[::2], pairs[1::2]):
                result[index] = 16 + offset
            self.assertEqual(result, large.value_info[name][0], msg=name)
        self.assertEqual(small.metadata["qwen.sequence.value.token_ids"], [1, 0])
        self.assertNotIn("qwen.context.value.token_ids", small.metadata)
        self.assertNotIn("qwen.sequence.value.k_cache_1", small.metadata)

    def test_partial_rotary_passthrough_preserves_the_sequence_symbol(self):
        text = dict(TEXT)
        text["rope_parameters"] = dict(TEXT["rope_parameters"], partial_rotary_factor=0.5)
        (self.model_dir / "config.json").write_text(json.dumps({"text_config": text}))
        canonical, direct = self.build().g, self.build(sequence=3).g
        for name, (shape, _) in canonical.value_info.items():
            self.assertEqual(self.rebound(canonical, shape, "qwen.sequence.value." + name, 3),
                             direct.value_info[name][0], msg=name)
        for name, array in canonical.constant_values.items():
            if name != "qwen_chunk_offsets":
                rebound = self.rebound(canonical, array.reshape(-1), "qwen.sequence.constant." + name, 3)
                np.testing.assert_array_equal(np.asarray(rebound).reshape(array.shape),
                                              direct.constant_values[name], err_msg=name)

    def test_both_exporters_keep_original_bf16_sensitive_weights_and_controls(self):
        sensitive = ("embed_tokens.weight", "linear_attn.conv1d.weight",
                     "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")
        # BF16 1.03125 is not exactly representable in either FP8 format.
        for key, value in self.source.tensors.items():
            if key.endswith(sensitive):
                value.fill_(1.03125)
        for dtype in (atomic.FP8E4M3, atomic.FP8E5M2):
            for builder_type in (atomic.QwenBuilder, OptimizedQwenBuilder):
                builder = builder_type(self.model_dir, 8, fp8_dtype=dtype)
                builder.source = self.source
                graph = builder.build()
                self.assertEqual(graph.metadata["qwen.precision.policy"], atomic.QWEN_PRECISION_POLICY)
                embedding_nodes = [node for node in graph.nodes if node.op_type == "Gather"]
                head = next(node for node in graph.nodes if node.outputs == ["logits"])
                self.assertEqual(len(embedding_nodes), 1)
                embedding = embedding_nodes[0].inputs[0]
                self.assertEqual(head.inputs[1], embedding)
                self.assertEqual(graph.value_info[embedding_nodes[0].outputs[0]][1], atomic.BF16)
                for key in self.source.tensors:
                    if not key.endswith(sensitive):
                        continue
                    prefix = "weight_" + key.replace(".", "_")
                    matching = [name for name in graph.constants if name.startswith(prefix)]
                    self.assertEqual(len(matching), 1, msg=prefix)
                    name = matching[0]
                    self.assertNotIn("__fp8", name)
                    self.assertEqual(graph.value_info[name][1], atomic.BF16)
                    expected = atomic.source_to_numpy(self.source.get_tensor(key))
                    if name.endswith("_t"):
                        expected = np.ascontiguousarray(expected.T)
                    self.assertEqual(graph.external_constant_loaders[name]().tobytes(), expected.tobytes())
                    if "in_proj_a" in name or "in_proj_b" in name:
                        nodes = [node for node in graph.nodes if node.op_type == "MatMul" and node.inputs[1] == name]
                        self.assertEqual(len(nodes), 1)
                        node = nodes[0]
                        self.assertTrue(all(graph.value_info[value][1] == atomic.BF16 for value in node.inputs + node.outputs))
                # Bulk projection kernels retain their existing FP8 contract.
                bulk = [node for node in graph.nodes if node.op_type == "MatMul" and
                        graph.value_info[node.outputs[0]][1] == dtype]
                self.assertTrue(bulk)
                for node in bulk:
                    self.assertTrue(all(graph.value_info[value][1] == dtype for value in node.inputs + node.outputs))


if __name__ == "__main__":
    unittest.main()
