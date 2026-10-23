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

"""Fixed-shape unquantized HF Qwen3 hidden-state exporter for FLUX.2."""

import math
import numpy as np
from flux2_graph import Graph, FP32, INT64


def build_text_encoder(
    store, config, text_length, dtype=FP32, hidden_layers=(9, 18, 27)
):
    """Concatenate HF hidden_states[k], preserving requested order (including 0).

    Positions are fixed 0..L-1, including padding, matching Qwen3's default
    position_ids. The attention_mask is additive FP32 [1,1,1,L]. Only selected
    final-layer states receive model.norm. Weights remain lazy until write().
    """
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        raise ValueError("Qwen3 config must be a mapping")
    if config.get("model_type", "qwen3") != "qwen3":
        raise ValueError("only model_type=qwen3 is supported")
    if config.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embedding is unsupported")
    if config.get("rope_scaling"):
        raise ValueError("Qwen3 rope_scaling is unsupported")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen3 hidden_act must be silu")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise ValueError("Qwen3 biased projections are unsupported")
    if config.get("use_sliding_window", False) or any(
        kind != "full_attention" for kind in config.get("layer_types", []) or []
    ):
        raise ValueError("Qwen3 sliding-window attention is unsupported")

    def positive(name, default=None):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    hidden = positive("hidden_size")
    heads = positive("num_attention_heads")
    kv_heads = positive("num_key_value_heads", heads)
    depth = positive("head_dim", hidden // heads)
    total = positive("num_hidden_layers")
    intermediate = positive("intermediate_size")
    vocab = positive("vocab_size")
    if heads % kv_heads or depth % 2:
        raise ValueError("invalid Qwen3 GQA heads or odd rotary head_dim")
    if (
        isinstance(text_length, bool)
        or not isinstance(text_length, int)
        or text_length <= 0
    ):
        raise ValueError("text_length must be a positive integer")
    if "max_position_embeddings" in config and text_length > positive(
        "max_position_embeddings"
    ):
        raise ValueError("text_length exceeds max_position_embeddings")
    if hidden_layers is None:
        raise ValueError("hidden_layers must be a nonempty sequence")
    selected = tuple(hidden_layers)
    if not selected or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > total
        for k in selected
    ):
        raise ValueError("hidden_layers must contain indices in [0,num_hidden_layers]")
    epsilon = float(config.get("rms_norm_eps", 1e-6))
    theta = float(config.get("rope_theta", 1000000.0))
    if (
        not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("rms_norm_eps and rope_theta must be finite and positive")
    expected = {"model.embed_tokens.weight": [vocab, hidden]}
    for i in range(max(selected)):
        p = f"model.layers.{i}."
        for name, shape in [
            ("input_layernorm", [hidden]),
            ("post_attention_layernorm", [hidden]),
            ("self_attn.q_norm", [depth]),
            ("self_attn.k_norm", [depth]),
            ("self_attn.q_proj", [heads * depth, hidden]),
            ("self_attn.k_proj", [kv_heads * depth, hidden]),
            ("self_attn.v_proj", [kv_heads * depth, hidden]),
            ("self_attn.o_proj", [hidden, heads * depth]),
            ("mlp.gate_proj", [intermediate, hidden]),
            ("mlp.up_proj", [intermediate, hidden]),
            ("mlp.down_proj", [hidden, intermediate]),
        ]:
            expected[p + name + ".weight"] = shape
    if total in selected:
        expected["model.norm.weight"] = [hidden]
    for key, shape in expected.items():
        if list(store.shape(key)) != shape:
            raise ValueError(f"{key}: expected {shape}, got {store.shape(key)}")
        if key.removesuffix(".weight") + ".bias" in store:
            raise ValueError(f"unexpected Qwen3 projection/norm bias: {key}")
    g = Graph(store, dtype)
    ids = g.input("input_ids", [1, text_length], INT64)
    mask = g.input("attention_mask", [1, 1, 1, text_length], FP32)
    x = g.node(
        "Gather",
        [g.weight("model.embed_tokens.weight"), ids],
        g.fresh("embedding"),
        [[1, text_length, hidden]],
        dtype,
        axis=0,
    )
    states = {0: x}
    angle = (
        np.arange(text_length, dtype=np.float32)[:, None]
        * np.power(
            np.float32(theta), -np.arange(0, depth, 2, dtype=np.float32) / depth
        )[None]
    )
    cos = g.constant_float(
        np.concatenate((np.cos(angle), np.cos(angle)), -1)[None, None], "rope_cos", dtype
    )
    sin = g.constant_float(
        np.concatenate((np.sin(angle), np.sin(angle)), -1)[None, None], "rope_sin", dtype
    )
    minus_one = g.constant_float(-1, "negative", dtype)

    def cast(value, target, stem):
        return g.cast(value, g.shape(value), target, stem)

    def norm(value, key):
        # HF computes variance in FP32, casts normalized x, then applies the
        # affine weight in the original storage dtype (a distinct BF16 round).
        normalized = g.normalize(
            cast(value, FP32, "norm_float"), epsilon=epsilon, layer=False
        )
        return g.binary_op(
            "Mul", cast(normalized, dtype, "norm_cast"), g.weight(key), "norm_affine"
        )

    def rope(value):
        # HF casts cos/sin to hidden dtype first, then rounds each multiply
        # and addition in that dtype. FP32 arithmetic changes BF16 queries.
        y = value
        first = g.slice_axis(y, 0, depth // 2, -1)
        second = g.slice_axis(y, depth // 2, depth, -1)
        rotated = g.concat(
            [g.binary_op("Mul", second, minus_one), first],
            -1,
            g.shape(y),
            dtype,
            "rotate_half",
        )
        return g.binary_op(
            "Add", g.binary_op("Mul", y, cos), g.binary_op("Mul", rotated, sin)
        )

    def silu(value):
        value = cast(value, FP32, "silu_float")
        return cast(
            g.binary_op("Mul", value, g.unary_op("Sigmoid", value), "silu"),
            dtype,
            "silu_cast",
        )

    for i in range(max(selected)):
        p = f"model.layers.{i}."
        h = norm(x, p + "input_layernorm.weight")
        qkv = []
        for kind, nh in [("q", heads), ("k", kv_heads), ("v", kv_heads)]:
            y = g.linear(h, p + "self_attn." + kind + "_proj.weight")
            y = g.reshape(y, [1, text_length, nh, depth], kind + "_heads")
            y = g.transpose(
                y, [0, 2, 1, 3], [1, nh, text_length, depth], kind + "_transpose"
            )
            if kind != "v":
                y = rope(norm(y, p + "self_attn." + kind + "_norm.weight"))
            qkv.append(y)
        h = g.attention(*qkv, mask=mask, causal=True)
        h = g.transpose(
            h, [0, 2, 1, 3], [1, text_length, heads, depth], "attention_tokens"
        )
        h = g.reshape(h, [1, text_length, heads * depth], "attention_flat")
        x = g.binary_op(
            "Add", x, g.linear(h, p + "self_attn.o_proj.weight"), "attention_residual"
        )
        h = norm(x, p + "post_attention_layernorm.weight")
        h = g.binary_op(
            "Mul",
            silu(g.linear(h, p + "mlp.gate_proj.weight")),
            g.linear(h, p + "mlp.up_proj.weight"),
            "gated_mlp",
        )
        x = g.binary_op(
            "Add", x, g.linear(h, p + "mlp.down_proj.weight"), "mlp_residual"
        )
        states[i + 1] = x
    if total in selected:
        states[total] = norm(x, "model.norm.weight")
    result = (
        states[selected[0]]
        if len(selected) == 1
        else g.concat(
            [states[k] for k in selected],
            -1,
            [1, text_length, hidden * len(selected)],
            dtype,
            "selected_states",
        )
    )
    g.node(
        "Identity",
        [result],
        "prompt_embeds",
        [[1, text_length, hidden * len(selected)]],
        dtype,
    )
    g.output("prompt_embeds")
    return g
