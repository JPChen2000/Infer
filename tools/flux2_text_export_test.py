# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mandatory tiny Qwen3 math references, independent of transformers installation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_graph import TensorStore, BF16


def fixture(layers=3):
    torch.manual_seed(917)
    config = dict(
        model_type="qwen3",
        hidden_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=layers,
        intermediate_size=12,
        vocab_size=19,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    weights = {
        "model.embed_tokens.weight": torch.randn(19, 8) * 0.3,
        "model.norm.weight": torch.rand(8) + 0.5,
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        for name, shape in [
            ("input_layernorm", (8,)),
            ("post_attention_layernorm", (8,)),
            ("self_attn.q_norm", (4,)),
            ("self_attn.k_norm", (4,)),
        ]:
            weights[p + name + ".weight"] = torch.rand(shape) + 0.5
        for name, shape in [
            ("self_attn.q_proj", (8, 8)),
            ("self_attn.k_proj", (4, 8)),
            ("self_attn.v_proj", (4, 8)),
            ("self_attn.o_proj", (8, 8)),
            ("mlp.gate_proj", (12, 8)),
            ("mlp.up_proj", (12, 8)),
            ("mlp.down_proj", (8, 12)),
        ]:
            weights[p + name + ".weight"] = torch.randn(shape) * 0.18
    return config, weights


def reference(config, w, ids, mask, selected):
    dtype = w["model.embed_tokens.weight"].dtype

    def norm(x, weight):
        z = x.float()
        return (
            z * torch.rsqrt(z.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
        ).to(dtype) * weight

    x = F.embedding(ids, w["model.embed_tokens.weight"])
    states = {0: x}
    length = ids.shape[1]
    d = config["head_dim"]
    angle = (
        torch.arange(length).float()[:, None]
        * (config["rope_theta"] ** (-torch.arange(0, d, 2).float() / d))[None]
    )
    # Qwen3RotaryEmbedding casts the trigonometric buffers to hidden dtype
    # before apply_rotary_pos_emb multiplies/adds in that dtype.
    cos = torch.cat((angle.cos(), angle.cos()), -1)[None, None].to(dtype)
    sin = torch.cat((angle.sin(), angle.sin()), -1)[None, None].to(dtype)

    def rope(y):
        return y * cos + torch.cat((-y[..., d // 2 :], y[..., : d // 2]), -1) * sin

    for i in range(max(selected)):
        p = f"model.layers.{i}."
        h = norm(x, w[p + "input_layernorm.weight"])
        q, k, v = [
            F.linear(h, w[p + "self_attn." + kind + "_proj.weight"])
            .reshape(1, length, heads, d)
            .transpose(1, 2)
            for kind, heads in [
                ("q", config["num_attention_heads"]),
                ("k", config["num_key_value_heads"]),
                ("v", config["num_key_value_heads"]),
            ]
        ]
        q = rope(norm(q, w[p + "self_attn.q_norm.weight"]))
        k = rope(norm(k, w[p + "self_attn.k_norm.weight"]))
        k = k.repeat_interleave(
            config["num_attention_heads"] // config["num_key_value_heads"], 1
        )
        v = v.repeat_interleave(
            config["num_attention_heads"] // config["num_key_value_heads"], 1
        )
        scores = q.float() @ k.float().transpose(-1, -2) / d**0.5 + mask
        scores = scores.masked_fill(
            torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf
        )
        probs = scores.softmax(-1).nan_to_num()
        attn = (probs @ v.float()).to(dtype).transpose(1, 2).reshape(1, length, -1)
        x = x + F.linear(attn, w[p + "self_attn.o_proj.weight"])
        h = norm(x, w[p + "post_attention_layernorm.weight"])
        x = x + F.linear(
            F.silu(F.linear(h, w[p + "mlp.gate_proj.weight"]))
            * F.linear(h, w[p + "mlp.up_proj.weight"]),
            w[p + "mlp.down_proj.weight"],
        )
        states[i + 1] = x
    if config["num_hidden_layers"] in selected:
        states[config["num_hidden_layers"]] = norm(x, w["model.norm.weight"])
    return torch.cat([states[k] for k in selected], -1)


class Qwen3TextExportTest(unittest.TestCase):
    def test_native_reference_all_states_and_padding(self):
        from flux2_text_export import build_text_encoder

        probe = Path(
            os.environ.get(
                "FLUX2_COMPONENT_PROBE",
                str(
                    Path(__file__).resolve().parents[1]
                    / "build_new/bin/flux2_component_probe"
                ),
            )
        )
        self.assertTrue(probe.is_file(), f"mandatory native probe missing: {probe}")
        config, weights = fixture()
        ids = np.array([[2, 4, 8, 1]], np.int64)
        for selected in [(0, 1, 2, 3), (2, 1), (3,)]:
            for padding in [[], [0, 1], [2, 3]]:
                mask = np.zeros((1, 1, 1, 4), np.float32)
                mask[..., padding] = -np.inf
                graph = build_text_encoder(
                    TensorStore(weights), config, 4, hidden_layers=selected
                )
                ref = reference(
                    config,
                    weights,
                    torch.from_numpy(ids),
                    torch.from_numpy(mask),
                    selected,
                ).numpy()
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    graph.write(root / "text.fth", "tiny_qwen3")
                    ids.tofile(root / "ids.bin")
                    mask.tofile(root / "mask.bin")
                    for device in ["common", "cuda"]:
                        if device == "cuda" and os.environ.get("FEATHER_TEST_CUDA", "1") == "0":
                            continue
                        result = subprocess.run(
                            [
                                str(probe),
                                str(root / "text.fth"),
                                str(root),
                                device,
                                f"input_ids={root}/ids.bin",
                                f"attention_mask={root}/mask.bin",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        actual = np.fromfile(
                            root / "prompt_embeds.bin", np.float32
                        ).reshape(ref.shape)
                        np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-5)

    def test_bf16_native_reference(self):
        import ml_dtypes
        from flux2_text_export import build_text_encoder

        probe = Path(
            os.environ.get(
                "FLUX2_COMPONENT_PROBE",
                str(
                    Path(__file__).resolve().parents[1]
                    / "build_new/bin/flux2_component_probe"
                ),
            )
        )
        self.assertTrue(probe.is_file())
        config, weights = fixture(2)
        weights = {k: v.bfloat16() for k, v in weights.items()}
        ids = np.array([[3, 1, 5, 2]], np.int64)
        mask = np.array([[[[0, 0, -np.inf, -np.inf]]]], np.float32)
        selected = (0, 1, 2)
        graph = build_text_encoder(
            TensorStore(weights), config, 4, dtype=BF16, hidden_layers=selected
        )
        pre_rope = next(
            n.outputs[0] for n in graph.nodes
            if n.op_type == "Mul" and any("self_attn.q_norm.weight" in x for x in n.inputs)
        )
        post_rope = next(
            n.inputs[0] for n in graph.nodes if n.op_type == "ScaledDotProductAttention"
        )
        graph.output(pre_rope)
        graph.output(post_rope)
        self.assertEqual(graph.value_info["prompt_embeds"], ([1, 4, 24], BF16))
        ref = (
            reference(
                config, weights, torch.from_numpy(ids), torch.from_numpy(mask), selected
            )
            .float()
            .numpy()
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph.write(root / "text.fth", "tiny_qwen3_bf16")
            ids.tofile(root / "ids.bin")
            mask.tofile(root / "mask.bin")
            for device in ["common", "cuda"]:
                if device == "cuda" and os.environ.get("FEATHER_TEST_CUDA", "1") == "0":
                    continue
                result = subprocess.run(
                    [
                        str(probe),
                        str(root / "text.fth"),
                        str(root),
                        device,
                        f"input_ids={root}/ids.bin",
                        f"attention_mask={root}/mask.bin",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                actual = (
                    np.fromfile(root / "prompt_embeds.bin", ml_dtypes.bfloat16)
                    .astype(np.float32)
                    .reshape(ref.shape)
                )
                np.testing.assert_allclose(actual, ref, atol=0.04, rtol=0.04)
                q = torch.from_numpy(
                    np.fromfile(root / (pre_rope + ".bin"), ml_dtypes.bfloat16)
                    .astype(np.float32).reshape(1, 2, 4, 4)
                ).bfloat16()
                angle = torch.arange(4).float()[:, None] * (
                    config["rope_theta"] ** (-torch.arange(0, 4, 2).float() / 4)
                )[None]
                cos = torch.cat((angle.cos(), angle.cos()), -1)[None, None].bfloat16()
                sin = torch.cat((angle.sin(), angle.sin()), -1)[None, None].bfloat16()
                expected_rope = (
                    q * cos + torch.cat((-q[..., 2:], q[..., :2]), -1) * sin
                ).float().numpy()
                actual_rope = np.fromfile(
                    root / (post_rope + ".bin"), ml_dtypes.bfloat16
                ).astype(np.float32).reshape(expected_rope.shape)
                np.testing.assert_array_equal(actual_rope, expected_rope)

    def test_early_stop_is_lazy_and_embedding_state(self):
        from flux2_text_export import build_text_encoder

        config, weights = fixture()
        # Neither later blocks nor final norm are required for an early state.
        limited = {
            key: value
            for key, value in weights.items()
            if key.startswith("model.layers.0.") or key == "model.embed_tokens.weight"
        }
        graph = build_text_encoder(
            TensorStore(limited), config, 4, hidden_layers=(1, 0, 1)
        )
        self.assertEqual(graph.value_info["prompt_embeds"][0], [1, 4, 24])
        self.assertEqual(
            sum(n.op_type == "ScaledDotProductAttention" for n in graph.nodes), 1
        )
        graph = build_text_encoder(
            TensorStore(
                {"model.embed_tokens.weight": weights["model.embed_tokens.weight"]}
            ),
            config,
            4,
            hidden_layers=(0,),
        )
        self.assertFalse(
            any(n.op_type == "ScaledDotProductAttention" for n in graph.nodes)
        )

    def test_rejects_bad_config_and_weights(self):
        from flux2_text_export import build_text_encoder

        config, weights = fixture(2)
        for change in [
            {"model_type": "qwen2"},
            {"rope_scaling": {"type": "linear", "factor": 2}},
            {"head_dim": 3},
            {"num_key_value_heads": 3},
            {"hidden_act": "gelu"},
            {"partial_rotary_factor": 0.5},
            {"use_sliding_window": True},
            {"max_position_embeddings": 2},
        ]:
            with self.assertRaises(ValueError):
                build_text_encoder(
                    TensorStore(weights), config | change, 4, hidden_layers=(1, 2)
                )
        with self.assertRaises(ValueError):
            build_text_encoder(TensorStore(weights), config, 4, hidden_layers=(3,))
        weights["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(9, 8)
        with self.assertRaises(ValueError):
            build_text_encoder(TensorStore(weights), config, 4, hidden_layers=(1,))


if __name__ == "__main__":
    unittest.main()
