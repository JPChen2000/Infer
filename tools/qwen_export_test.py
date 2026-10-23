#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused regression tests for Qwen atomic ONNX graph construction."""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_qwen35_model", ROOT / "tools" / "export_qwen35_model.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPORTER)


class AtomicGraphTest(unittest.TestCase):
    def test_context_bindings_do_not_rebind_equal_static_dimensions(self):
        dimension = getattr(EXPORTER, "ContextDimension", None)
        self.assertIsNotNone(dimension, "context expressions must be annotated explicitly")
        context = dimension(4)
        graph = EXPORTER.AtomicGraph()
        value = graph.input("value", [1, 4, context], EXPORTER.BF16)
        graph.input("cache", [1, context - 1, 4], EXPORTER.BF16)
        graph.reshape(value, [1, context, 4], "reshape")
        graph.node("Reshape", [value], "attr_shape", [[1, context, 4]], EXPORTER.BF16,
                   name="attribute_reshape", shape=[1, context, 4])
        self.assertEqual(graph.metadata["qwen.context.value.value"], [2, 0])
        self.assertEqual(graph.metadata["qwen.context.value.cache"], [1, -1])
        shape_names = [name for name in graph.constant_names if name.startswith("reshape_shape_")]
        self.assertEqual(len(shape_names), 1)
        self.assertEqual(graph.metadata["qwen.context.constant." + shape_names[0]], [1, 0])
        self.assertEqual(graph.metadata["qwen.context.attribute.attribute_reshape.shape"], [1, 0])
        self.assertEqual(graph.value_info["value"][0], [1, 4, 4])
        self.assertEqual(graph.constant_values[shape_names[0]].tolist(), [1, 4, 4])

    def test_fp8_export_contract_uses_fth_v4_and_standard_onnx_types(self):
        self.assertEqual(EXPORTER.FTH_FORMAT_VERSION, 4)
        self.assertEqual(EXPORTER.FTH_DTYPE_BYTES.get(12), 1)
        self.assertEqual(EXPORTER.FTH_DTYPE_BYTES.get(13), 1)
        self.assertEqual(EXPORTER.FTH_TO_ONNX_DTYPE.get(12), 17)
        self.assertEqual(EXPORTER.FTH_TO_ONNX_DTYPE.get(13), 19)

    def test_fth_tensor_quantization_tail_matches_cpp_field_widths(self):
        descriptor = EXPORTER.FeatherTensorDesc(name="x", dims=[2], data_type=EXPORTER.FP8E5M2,
                                                quantization_enabled=True, quantization_scale=0.25)
        encoded = EXPORTER.fth_encode_tensor(descriptor)
        expected = (struct.pack("<Q", 1) + b"x" + struct.pack("<QqiiBfiqqiQQ", 1, 2,
                    EXPORTER.FP8E5M2, EXPORTER.FTH_LAYOUT_ND, 1, 0.25, 0, -1, 0, 0, 0, 0))
        self.assertEqual(encoded, expected)

    def test_fp8_weight_encoder_uses_standard_bytes_and_power_of_two_scales(self):
        encode = getattr(EXPORTER, "fp8_encode", lambda *_: np.asarray([], dtype=np.uint8))
        scale_for_amax = getattr(EXPORTER, "fp8_scale_for_amax", lambda *_: 0.0)

        self.assertEqual(scale_for_amax(EXPORTER.FP8E4M3, 224.0), 0.5)
        self.assertEqual(scale_for_amax(EXPORTER.FP8E5M2, 28672.0), 0.5)
        self.assertEqual(
            encode(np.asarray([1.0, -1.0, 448.0], dtype=np.float32), EXPORTER.FP8E4M3, 1.0).tolist(),
            [0x38, 0xB8, 0x7E],
        )
        self.assertEqual(
            encode(np.asarray([1.0, -1.0, 57344.0], dtype=np.float32), EXPORTER.FP8E5M2, 1.0).tolist(),
            [0x3C, 0xBC, 0x7B],
        )

    def test_fp8_cast_records_per_tensor_scale_for_fth_v4(self):
        graph = EXPORTER.AtomicGraph()
        value = graph.input("value", [2], EXPORTER.BF16)
        try:
            fp8_value = graph.cast(value, [2], EXPORTER.FP8E4M3, "to_fp8", quantization_scale=0.25)
        except TypeError:
            fp8_value = None
        self.assertIsNotNone(fp8_value)
        self.assertEqual(graph.value_info[fp8_value][1], EXPORTER.FP8E4M3)
        self.assertEqual(graph.quantization_scales[fp8_value], 0.25)

        restored = graph.cast(fp8_value, [2], EXPORTER.BF16, "to_bf16")
        graph.output(restored, [2], EXPORTER.BF16)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fp8_cast.fth"
            graph.write_fth(output, "fp8_cast")
            _, version, _, _ = struct.unpack("<8sIIQ", output.read_bytes()[:24])
            self.assertEqual(version, 4)

    def test_direct_fth_export_does_not_import_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atomic.fth"
            script = f"""
import builtins
import importlib.util

original_import = builtins.__import__
def blocked_onnx(name, *args, **kwargs):
    if name == 'onnx' or name.startswith('onnx.'):
        raise ImportError('onnx is intentionally unavailable')
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_onnx
spec = importlib.util.spec_from_file_location('direct_exporter', r'{ROOT / "tools" / "export_qwen35_model.py"}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
graph = module.AtomicGraph()
input_name = graph.input('value', [1], module.BF16)
output_name = graph.unary('Neg', input_name, [1], module.BF16, 'neg')
graph.output(output_name, [1], module.BF16)
graph.write_fth(r'{output}', 'atomic')
"""
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(output.read_bytes()[:8], b"FTHMODL\x00")

    def test_expand_uses_static_int64_shape_input(self):
        graph = EXPORTER.AtomicGraph()
        value = graph.input("value", [1, 2, 1, 4, 8], EXPORTER.BF16)
        expanded = graph.expand(value, [1, 2, 4, 4, 8], "repeat_kv")
        graph.output(expanded, [1, 2, 4, 4, 8], EXPORTER.BF16)

        model = graph.make_model(max_context=4)
        onnx.checker.check_model(model)

        self.assertEqual(len(model.graph.node), 1)
        node = model.graph.node[0]
        self.assertEqual(node.op_type, "Expand")
        self.assertEqual(len(node.input), 2)
        self.assertEqual(node.input[0], "value")
        self.assertEqual(node.input[1], "repeat_kv_shape_1")

    def test_bfloat16_source_conversion_preserves_payload(self):
        import torch

        source = torch.tensor([1.0, -2.5, 0.333], dtype=torch.bfloat16)
        converted = EXPORTER.source_to_numpy(source)

        self.assertEqual(str(converted.dtype), "bfloat16")
        self.assertEqual(converted.tobytes(), source.view(torch.uint16).numpy().tobytes())
        self.assertEqual(converted.dtype, ml_dtypes.bfloat16)

    def test_tiny_decode_graph_uses_explicit_named_state_contract(self):
        import torch

        text = {
            "max_position_embeddings": 64,
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 2,
            "vocab_size": 8,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            "rms_norm_eps": 1e-6,
        }
        tensors = {"model.language_model.embed_tokens.weight": torch.ones((8, 4), dtype=torch.bfloat16),
                   "model.language_model.norm.weight": torch.zeros(4, dtype=torch.bfloat16)}
        for layer in range(2):
            prefix = f"model.language_model.layers.{layer}"
            tensors[f"{prefix}.input_layernorm.weight"] = torch.zeros(4, dtype=torch.bfloat16)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.zeros(4, dtype=torch.bfloat16)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.ones((6, 4), dtype=torch.bfloat16)
            tensors[f"{prefix}.mlp.up_proj.weight"] = torch.ones((6, 4), dtype=torch.bfloat16)
            tensors[f"{prefix}.mlp.down_proj.weight"] = torch.ones((4, 6), dtype=torch.bfloat16)
        linear = "model.language_model.layers.0.linear_attn"
        tensors.update({
            f"{linear}.A_log": torch.zeros(2, dtype=torch.float32),
            f"{linear}.conv1d.weight": torch.ones((12, 1, 4), dtype=torch.bfloat16),
            f"{linear}.dt_bias": torch.zeros(2, dtype=torch.bfloat16),
            f"{linear}.in_proj_qkv.weight": torch.ones((12, 4), dtype=torch.bfloat16),
            f"{linear}.in_proj_z.weight": torch.ones((4, 4), dtype=torch.bfloat16),
            f"{linear}.in_proj_a.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            f"{linear}.in_proj_b.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            f"{linear}.norm.weight": torch.ones(2, dtype=torch.float32),
            f"{linear}.out_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
        })
        full = "model.language_model.layers.1.self_attn"
        tensors.update({
            f"{full}.q_proj.weight": torch.ones((16, 4), dtype=torch.bfloat16),
            f"{full}.k_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
            f"{full}.v_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
            f"{full}.o_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
            f"{full}.q_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
            f"{full}.k_norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        })

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps({"text_config": text}))
            shard = model_dir / "model.safetensors"
            save_file(tensors, shard)
            builder = EXPORTER.QwenBuilder(model_dir, max_context=4)
            with EXPORTER.safe_open(shard, framework="pt", device="cpu") as source:
                builder.source = source
                builder.build()
                for node in builder.g.nodes:
                    if (node.op_type == "MatMul" and
                            node.inputs[1] in builder.g.external_constant_loaders and
                            builder.g.value_info[node.outputs[0]][1] == EXPORTER.BF16):
                        self.assertTrue(all(builder.g.value_info[name][1] == EXPORTER.BF16
                                            for name in node.inputs), msg=node.name)
                model = builder.make_onnx_model()
                fth_path = model_dir / "tiny_qwen.fth"
                builder.write_fth(fth_path)
                self.assertGreater(fth_path.stat().st_size, 0)
                self.assertEqual(fth_path.read_bytes()[:8], b"FTHMODL\x00")

                # Specializing every explicit binding must produce the same graph
                # as exporting directly at the new context, even when hidden/head
                # dimensions happen to equal the original context size.
                larger = EXPORTER.QwenBuilder(model_dir, max_context=8)
                larger.source = source
                larger.build()
                graph = builder.g
                self.assertEqual(graph.metadata["qwen.context.maximum"], 64)
                self.assertEqual(set(graph.value_info), set(larger.g.value_info))
                def rebound(values, key):
                    result = list(values)
                    bindings = graph.metadata.get(key, [])
                    for index, offset in zip(bindings[::2], bindings[1::2]):
                        self.assertEqual(result[index], 4 + offset, msg=key)
                        result[index] = 8 + offset
                    return result
                for name, (shape, dtype) in graph.value_info.items():
                    expected_shape, expected_dtype = larger.g.value_info[name]
                    self.assertEqual(rebound(shape, "qwen.context.value." + name),
                                     expected_shape, msg=name)
                    self.assertEqual(dtype, expected_dtype)
                for name, array in graph.constant_values.items():
                    key = "qwen.context.constant." + name
                    actual = np.asarray(rebound(array.reshape(-1), key)).reshape(array.shape)
                    np.testing.assert_array_equal(actual, larger.g.constant_values[name], err_msg=name)
                for original, direct in zip(graph.nodes, larger.g.nodes):
                    self.assertEqual(original.name, direct.name)
                    for name, value in EXPORTER.feather_attributes(original.op_type, original.attributes).items():
                        expected = EXPORTER.feather_attributes(direct.op_type, direct.attributes)[name]
                        if isinstance(value, (list, tuple)):
                            value = rebound(value, f"qwen.context.attribute.{original.name}.{name}")
                        self.assertEqual(value, expected)

            for fp8_dtype in (EXPORTER.FP8E4M3, EXPORTER.FP8E5M2):
                try:
                    fp8_builder = EXPORTER.QwenBuilder(model_dir, max_context=4, fp8_dtype=fp8_dtype)
                except TypeError:
                    fp8_builder = None
                self.assertIsNotNone(fp8_builder)
                with EXPORTER.safe_open(shard, framework="pt", device="cpu") as source:
                    fp8_builder.source = source
                    fp8_graph = fp8_builder.build()
                    fp8_fth_path = model_dir / f"tiny_qwen_{fp8_dtype}.fth"
                    fp8_builder.write_fth(fp8_fth_path)

                self.assertEqual(fp8_graph.value_info["recurrent_state_0"][1], EXPORTER.FP32)
                self.assertEqual(fp8_graph.value_info["attention_mask"][1], EXPORTER.BF16)
                self.assertEqual(fp8_graph.value_info["logits"][1], EXPORTER.BF16)
                self.assertGreater(fp8_builder.fp8_activation_scale, 0.0)
                self.assertGreater(
                    sum(node.op_type == "Gather" and fp8_graph.value_info[node.outputs[0]][1] == EXPORTER.BF16
                        for node in fp8_graph.nodes),
                    0,
                )
                for op_type in ("MatMul",):
                    fp8_nodes = [node for node in fp8_graph.nodes
                                 if node.op_type == op_type and fp8_graph.value_info[node.outputs[0]][1] == fp8_dtype]
                    self.assertGreater(len(fp8_nodes), 0, msg=f"missing FP8 {op_type}")
                    for node in fp8_nodes:
                        self.assertTrue(all(fp8_graph.value_info[input_name][1] == fp8_dtype
                                            for input_name in node.inputs))
                        self.assertEqual(
                            fp8_graph.quantization_scales[node.outputs[0]],
                            fp8_builder.fp8_activation_scale,
                        )
                fp8_constants = [name for name in fp8_graph.constants
                                 if fp8_graph.value_info[name][1] == fp8_dtype]
                self.assertGreater(len(fp8_constants), 0)
                self.assertTrue(all("__fp8" in name for name in fp8_constants))
                self.assertTrue(all(fp8_graph.quantization_scales[name] > 0.0
                                    for name in fp8_constants))
                self.assertGreater(
                    sum(node.op_type == "Cast" and node.attributes["to"] == EXPORTER.FTH_TO_ONNX_DTYPE[fp8_dtype]
                        for node in fp8_graph.nodes),
                    0,
                )
                self.assertGreater(
                    sum(node.op_type == "Cast" and node.attributes["to"] == EXPORTER.FTH_TO_ONNX_DTYPE[EXPORTER.BF16]
                        for node in fp8_graph.nodes),
                    0,
                )
                for stem in ("layer_0_linear_mlp_product", "layer_1_full_mlp_product"):
                    products = [node for node in fp8_graph.nodes
                                if node.op_type == "Mul" and any(
                                    output.startswith(f"{stem}_") for output in node.outputs)]
                    self.assertEqual(len(products), 1, msg=f"missing BF16 {stem}")
                    product = products[0]
                    self.assertEqual(fp8_graph.value_info[product.outputs[0]][1], EXPORTER.BF16)
                    self.assertTrue(all(fp8_graph.value_info[input_name][1] == EXPORTER.BF16
                                        for input_name in product.inputs))
                # Repeated FP8 residual rounding erases small updates across layers.
                residuals = [node for node in fp8_graph.nodes if node.op_type == "Add" and any(
                    "residual_" in output for output in node.outputs)]
                self.assertEqual(len(residuals), 4)
                for residual in residuals:
                    self.assertEqual(fp8_graph.value_info[residual.outputs[0]][1], EXPORTER.BF16)
                    self.assertTrue(all(fp8_graph.value_info[name][1] == EXPORTER.BF16
                                        for name in residual.inputs))
                convolutions = [node for node in fp8_graph.nodes if node.op_type == "Conv"]
                self.assertEqual(len(convolutions), 1)
                self.assertEqual(fp8_graph.value_info[convolutions[0].outputs[0]][1], EXPORTER.BF16)
                self.assertTrue(all(fp8_graph.value_info[name][1] == EXPORTER.BF16
                                    for name in convolutions[0].inputs))
                # FP8 token scores can tie distinct candidates before greedy argmax.
                logits = [node for node in fp8_graph.nodes if node.outputs == ["logits"]]
                self.assertEqual(len(logits), 1)
                self.assertEqual(logits[0].op_type, "Gemm")
                self.assertTrue(all(fp8_graph.value_info[name][1] == EXPORTER.BF16
                                    for name in logits[0].inputs))
                _, version, _, _ = struct.unpack("<8sIIQ", fp8_fth_path.read_bytes()[:24])
                self.assertEqual(version, 4)

        onnx.checker.check_model(model)
        self.assertEqual(
            [output.name for output in model.graph.output],
            ["next_conv_state_0", "next_recurrent_state_0", "next_k_cache_1", "next_v_cache_1", "logits"],
        )
        full_attention_gate_activation_nodes = [
            node
            for node in model.graph.node
            if any(output.startswith("layer_1_full_gate_sigmoid_") for output in node.output)
        ]
        self.assertEqual(len(full_attention_gate_activation_nodes), 1)
        self.assertEqual(full_attention_gate_activation_nodes[0].op_type, "Sigmoid")
        self.assertFalse(any("layer_1_full_gate_silu" in output
                             for node in model.graph.node for output in node.output))

        # Qwen packs query and output-gate values per attention head as
        # [heads, 2 * head_dim]. Splitting the flattened projection would mix
        # query values from later heads with gate values from earlier heads.
        q_gate_splits = [
            node for node in model.graph.node
            if any(output.startswith("layer_1_full_q_gate_split_") for output in node.output)
        ]
        self.assertEqual(len(q_gate_splits), 1)
        q_gate_split = q_gate_splits[0]
        split_axis = next(attribute for attribute in q_gate_split.attribute if attribute.name == "axis")
        self.assertEqual(onnx.helper.get_attribute_value(split_axis), 3)
        value_shapes = {
            value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            for value in model.graph.value_info
        }
        self.assertEqual([value_shapes[output] for output in q_gate_split.output],
                         [[1, 1, 2, 4], [1, 1, 2, 4]])

        inputs = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
                  for value in model.graph.input}
        self.assertEqual(inputs["k_cache_1"], [1, 1, 3, 4])
        self.assertEqual(inputs["v_cache_1"], [1, 1, 3, 4])
        self.assertGreaterEqual(sum(node.op_type == "Expand" for node in model.graph.node), 2)
        self.assertGreaterEqual(sum(node.op_type == "Concat" for node in model.graph.node), 5)



if __name__ == "__main__":
    unittest.main()
