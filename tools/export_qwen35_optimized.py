#!/usr/bin/env python3
# Copyright 2026 JPChen2000 and the Feather contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export one sequence-specializable Qwen3.5 graph with causal GQA caches.

The legacy atomic/debug exporter stays in export_qwen35_model.py. These FTH
graphs use internal fused operators and are not portable ONNX exports.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from qwen_fp8_calibration import load_calibration

from export_qwen35_model import (
    QwenBuilder, SequenceDimension, BF16, FP32, INT64, FP8E4M3, FP8E5M2,
    DEFAULT_MODEL_DIR, QWEN_PRECISION_POLICY, resolve_shard, fp8_format_name,
)


class OptimizedQwenBuilder(QwenBuilder):
    def __init__(self, *args, sequence=1, prefill_chunk=None, weights_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence = SequenceDimension(sequence)
        self.prefill_chunk = int(prefill_chunk if prefill_chunk is not None else self.max_context)
        if not 1 <= self.sequence <= self.prefill_chunk <= self.max_context:
            raise ValueError("prefill chunk must fit the context capacity")
        if self.text["linear_num_key_heads"] != self.linear_heads:
            raise NotImplementedError("this hybrid export requires equal linear key/value head counts")
        if int(self.text["linear_conv_kernel_dim"]) != 4:
            raise NotImplementedError("this hybrid export requires convolution width four")
        self.g.metadata.update({"qwen.execution.version": 2,
                                "qwen.execution.sequence": int(self.sequence),
                                "qwen.sequence.version": 1,
                                "qwen.sequence.default": int(self.sequence),
                                "qwen.sequence.maximum": self.prefill_chunk,
                                "qwen.cache.layout": "position",
                                "qwen.weights.id": weights_id})

    def slice(self, x, start, end, axis, stem):
        shape, dtype = self.g.value_info[x]
        result = list(shape)
        result[axis] = end - start
        return self.g.node("Slice", [x], self.g.fresh(stem), [result], dtype,
                           start=start, end=end, axis=axis)

    def rms_norm(self, x, weight_key, stem):
        x = self.cast(x, BF16, f"{stem}_input")
        return self.g.node("QwenRmsNorm", [x, self.weight(weight_key), self.g.scalar("qwen_rms_eps", self.eps)],
                           self.g.fresh(stem), [self.g.value_info[x][0]], BF16, weight_offset=1.0)

    def mlp(self, hidden, layer, stem):
        residual = hidden
        prefix = f"model.language_model.layers.{layer}"
        hidden = self.rms_norm(hidden, f"{prefix}.post_attention_layernorm.weight", f"{stem}_post_norm")
        gate = self.silu(self.projection(hidden, f"{prefix}.mlp.gate_proj.weight", self.intermediate, f"{stem}_mlp_gate"), f"{stem}_mlp_silu")
        up = self.projection(hidden, f"{prefix}.mlp.up_proj.weight", self.intermediate, f"{stem}_mlp_up")
        product = self.g.binary("Mul", gate, up, [1, self.sequence, self.intermediate], BF16, f"{stem}_mlp_product")
        down = self.projection(product, f"{prefix}.mlp.down_proj.weight", self.hidden, f"{stem}_mlp_down")
        return self.g.binary("Add", residual, down, [1, self.sequence, self.hidden], BF16, f"{stem}_mlp_residual")

    def rope(self, x, position, stem):
        t = self.sequence
        p = self.cast(position, FP32, f"{stem}_position_float")
        offsets = self.g.const("qwen_chunk_offsets", np.arange(int(t), dtype=np.float32))
        self.g.register(offsets, [t], FP32)
        self.g.metadata["qwen.sequence.arange." + offsets] = 1
        p = self.g.binary("Add", p, offsets, [t], FP32, f"{stem}_positions")
        p = self.g.reshape(p, [t, 1], f"{stem}_position_column")
        inv = self.g.const("qwen_rope_inv_freq", np.asarray(
            1.0 / (self.rope_theta ** (np.arange(0, self.rope_dim, 2, dtype=np.float32) / self.rope_dim)), dtype=np.float32))
        freq = self.g.binary("Mul", p, inv, [t, self.rope_dim // 2], FP32, f"{stem}_freq")
        freq = self.g.concat([freq, freq], 1, [t, self.rope_dim], FP32, f"{stem}_freq_concat")
        freq = self.g.reshape(freq, [1, 1, t, self.rope_dim], f"{stem}_freq_reshape")
        cos = self.cast(self.g.unary("Cos", freq, [1, 1, t, self.rope_dim], FP32, f"{stem}_cos"), BF16, f"{stem}_cos_bf16")
        sin = self.cast(self.g.unary("Sin", freq, [1, 1, t, self.rope_dim], FP32, f"{stem}_sin"), BF16, f"{stem}_sin_bf16")
        shape = self.g.value_info[x][0]
        rot_shape, pass_shape = list(shape), list(shape)
        rot_shape[-1] = self.rope_dim
        pass_shape[-1] -= self.rope_dim
        if pass_shape[-1]:
            x_rot, x_pass = self.g.split(x, [self.rope_dim, pass_shape[-1]], 3, [rot_shape, pass_shape], f"{stem}_split")
        else:
            x_rot, x_pass = x, None
        half_shape = [*shape[:-1], self.rope_dim // 2]
        first, second = self.g.split(x_rot, [self.rope_dim // 2] * 2, 3, [half_shape, half_shape], f"{stem}_halves")
        negative = self.g.unary("Neg", second, half_shape, BF16, f"{stem}_neg")
        rotated = self.g.concat([negative, first], 3, rot_shape, BF16, f"{stem}_rotate")
        a = self.g.binary("Mul", x_rot, cos, rot_shape, BF16, f"{stem}_cos_mul")
        b = self.g.binary("Mul", rotated, sin, rot_shape, BF16, f"{stem}_sin_mul")
        result = self.g.binary("Add", a, b, rot_shape, BF16, f"{stem}_sum")
        return self.g.concat([result, x_pass], 3, shape, BF16, f"{stem}_output") if x_pass else result

    def build_full_layer(self, layer, hidden, position, attention_mask):
        t, h, d, kv = self.sequence, self.full_heads, self.head_dim, self.kv_heads
        stem = f"layer_{layer}_full"
        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        hidden = self.rms_norm(hidden, f"{prefix}.input_layernorm.weight", f"{stem}_input_norm")
        q_gate = self.projection(hidden, f"{prefix}.self_attn.q_proj.weight", h * d * 2, f"{stem}_q_gate")
        q_gate = self.g.reshape(q_gate, [1, t, h, d * 2], f"{stem}_q_gate_shape")
        q, gate = self.g.split(q_gate, [d, d], 3, [[1, t, h, d]] * 2, f"{stem}_q_gate_split")
        q = self.g.transpose(q, [0, 2, 1, 3], [1, h, t, d], f"{stem}_q_heads")
        gate = self.g.reshape(gate, [1, t, h * d], f"{stem}_gate_flat")
        k = self.projection(hidden, f"{prefix}.self_attn.k_proj.weight", kv * d, f"{stem}_k_proj")
        v = self.projection(hidden, f"{prefix}.self_attn.v_proj.weight", kv * d, f"{stem}_v_proj")
        k = self.g.reshape(k, [1, t, kv, d], f"{stem}_k_shape")
        v = self.g.reshape(v, [1, t, kv, d], f"{stem}_v_shape")
        k = self.g.transpose(k, [0, 2, 1, 3], [1, kv, t, d], f"{stem}_k_heads")
        v = self.g.transpose(v, [0, 2, 1, 3], [1, kv, t, d], f"{stem}_v_heads")
        q = self.rope(self.rms_norm(q, f"{prefix}.self_attn.q_norm.weight", f"{stem}_q_norm"), position, f"{stem}_q_rope")
        k = self.rope(self.rms_norm(k, f"{prefix}.self_attn.k_norm.weight", f"{stem}_k_norm"), position, f"{stem}_k_rope")
        attended = self.g.node("CachedAttention", [q, k, v, f"k_cache_{layer}", f"v_cache_{layer}", position],
                               self.g.fresh(f"{stem}_attention"), [[1, h, t, d]], BF16,
                               scale=float(1.0 / d ** 0.5), causal=1, round_intermediates=1)
        attended = self.g.transpose(attended, [0, 2, 1, 3], [1, t, h, d], f"{stem}_attended_heads")
        attended = self.g.reshape(attended, [1, t, h * d], f"{stem}_attended_flat")
        gate = self.g.unary("Sigmoid", gate, [1, t, h * d], BF16, f"{stem}_gate_sigmoid")
        attended = self.g.binary("Mul", attended, gate, [1, t, h * d], BF16, f"{stem}_gate_mul")
        attended = self.projection(attended, f"{prefix}.self_attn.o_proj.weight", self.hidden, f"{stem}_out_proj")
        hidden = self.g.binary("Add", residual, attended, [1, t, self.hidden], BF16, f"{stem}_residual")
        return self.mlp(hidden, layer, stem)

    def build_linear_chunk(self, layer, hidden):
        t, h, dk, dv = self.sequence, self.linear_heads, self.linear_key_dim, self.linear_value_dim
        c = self.conv_dim
        stem = f"layer_{layer}_linear"
        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        hidden = self.rms_norm(hidden, f"{prefix}.input_layernorm.weight", f"{stem}_input_norm")
        mixed = self.projection(hidden, f"{prefix}.linear_attn.in_proj_qkv.weight", c, f"{stem}_qkv_proj")
        mixed = self.g.transpose(mixed, [0, 2, 1], [1, c, t], f"{stem}_qkv_channels")
        joined = self.g.concat([f"conv_state_{layer}", mixed], 2, [1, c, t + 3], BF16, f"{stem}_conv_join")
        next_conv = self.slice(joined, t, t + 3, 2, f"{stem}_next_conv")
        conv_input = self.g.reshape(joined, [1, c, 1, t + 3], f"{stem}_conv_input")
        weight = self.weight(f"{prefix}.linear_attn.conv1d.weight", force_dtype=BF16)
        weight = self.g.reshape(weight, [c, 1, 1, 4], f"{stem}_conv_weight")
        weight = self.cast(weight, BF16, f"{stem}_conv_weight_bf16")
        conv = self.g.node("Conv", [conv_input, weight], self.g.fresh(f"{stem}_conv"), [[1, c, 1, t]], BF16,
                           strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=c)
        conv = self.silu(self.g.reshape(conv, [1, c, t], f"{stem}_conv_flat"), f"{stem}_conv_silu")
        conv = self.g.transpose(conv, [0, 2, 1], [1, t, c], f"{stem}_qkv_tokens")
        q, k, v = self.g.split(conv, [h * dk, h * dk, h * dv], 2,
                              [[1, t, h * dk], [1, t, h * dk], [1, t, h * dv]], f"{stem}_qkv_split")
        q = self.l2_norm(self.g.reshape(q, [1, t, h, dk], f"{stem}_q_shape"), f"{stem}_q_l2")
        k = self.l2_norm(self.g.reshape(k, [1, t, h, dk], f"{stem}_k_shape"), f"{stem}_k_l2")
        q = self.g.binary("Mul", q, self.g.scalar("qwen_linear_query_scale", 1.0 / dk ** 0.5), [1, t, h, dk], FP32, f"{stem}_q_scale")
        v = self.cast(self.g.reshape(v, [1, t, h, dv], f"{stem}_v_shape"), FP32, f"{stem}_v_fp32")
        z = self.projection(hidden, f"{prefix}.linear_attn.in_proj_z.weight", h * dv, f"{stem}_z_proj")
        z = self.g.reshape(z, [1, t, h, dv], f"{stem}_z_shape")
        b = self.projection(hidden, f"{prefix}.linear_attn.in_proj_b.weight", h, f"{stem}_b_proj", preserve_bf16=True)
        a = self.projection(hidden, f"{prefix}.linear_attn.in_proj_a.weight", h, f"{stem}_a_proj", preserve_bf16=True)
        b = self.cast(self.g.reshape(b, [1, t, h], f"{stem}_b_shape"), FP32, f"{stem}_b_fp32")
        a = self.cast(self.g.reshape(a, [1, t, h], f"{stem}_a_shape"), FP32, f"{stem}_a_fp32")
        beta = self.g.unary("Sigmoid", b, [1, t, h], FP32, f"{stem}_beta")
        dt = self.cast(self.weight(f"{prefix}.linear_attn.dt_bias"), FP32, f"{stem}_dt_fp32")
        a = self.g.binary("Add", a, dt, [1, t, h], FP32, f"{stem}_a_dt")
        a = self.g.unary("Softplus", a, [1, t, h], FP32, f"{stem}_softplus")
        alog = self.cast(self.weight(f"{prefix}.linear_attn.A_log"), FP32, f"{stem}_alog_fp32")
        aexp = self.g.unary("Exp", alog, [h], FP32, f"{stem}_aexp")
        negative = self.g.unary("Neg", aexp, [h], FP32, f"{stem}_negative")
        decay = self.g.binary("Mul", a, negative, [1, t, h], FP32, f"{stem}_decay")
        decay = self.g.unary("Exp", decay, [1, t, h], FP32, f"{stem}_decay_exp")
        state, core = self.g.node("QwenGatedDeltaSequence",
                                  [f"recurrent_state_{layer}", k, v, beta, decay, q],
                                  [self.g.fresh(f"{stem}_state"), self.g.fresh(f"{stem}_core")],
                                  [[1, h, dk, dv], [1, t, h, dv]], FP32)
        # Preserve the checkpoint's BF16 recurrent-to-norm boundary.
        core = self.cast(self.cast(core, BF16, f"{stem}_core_bf16"), FP32, f"{stem}_core_fp32")
        square = self.g.binary("Mul", core, core, [1, t, h, dv], FP32, f"{stem}_norm_square")
        mean = self.g.reduce("ReduceMean", square, [-1], True, [1, t, h, 1], FP32, f"{stem}_norm_mean")
        mean = self.g.binary("Add", mean, self.g.scalar("qwen_rms_eps_gated", self.eps), [1, t, h, 1], FP32, f"{stem}_norm_eps")
        root = self.g.unary("Sqrt", mean, [1, t, h, 1], FP32, f"{stem}_norm_root")
        norm = self.g.binary("Div", core, root, [1, t, h, dv], FP32, f"{stem}_norm_div")
        norm = self.g.binary("Mul", norm, self.weight(f"{prefix}.linear_attn.norm.weight"), [1, t, h, dv], FP32, f"{stem}_norm_scale")
        z = self.silu(self.cast(z, FP32, f"{stem}_z_fp32"), f"{stem}_gate_silu")
        gated = self.g.binary("Mul", norm, z, [1, t, h, dv], FP32, f"{stem}_gated")
        gated = self.g.reshape(gated, [1, t, h * dv], f"{stem}_gated_flat")
        projected = self.projection(gated, f"{prefix}.linear_attn.out_proj.weight", self.hidden, f"{stem}_out_proj")
        hidden = self.g.binary("Add", residual, projected, [1, t, self.hidden], BF16, f"{stem}_residual")
        return self.mlp(hidden, layer, stem), next_conv, state

    def build(self):
        g, t = self.g, self.sequence
        tokens = g.input("token_ids", [1, t], INT64)
        position = g.input("position_id", [1], INT64)
        # Retained for context specialization and compatibility, not consumed
        # by the fused causal prefix attention kernel.
        mask = g.input("attention_mask", [1, 1, 1, self.max_context], BF16)
        for layer, kind in enumerate(self.text["layer_types"]):
            if kind == "linear_attention":
                g.input(f"conv_state_{layer}", [1, self.conv_dim, 3], BF16)
                g.input(f"recurrent_state_{layer}", [1, self.linear_heads, self.linear_key_dim, self.linear_value_dim], FP32)
            else:
                g.input(f"k_cache_{layer}", [1, self.kv_heads, self.max_context, self.head_dim], BF16)
                g.input(f"v_cache_{layer}", [1, self.kv_heads, self.max_context, self.head_dim], BF16)
        embedding = self.weight("model.language_model.embed_tokens.weight", force_dtype=BF16)
        hidden = g.node("Gather", [embedding, tokens], g.fresh("embedding"), [[1, t, self.hidden]],
                        BF16, axis=0)
        hidden = self.cast(hidden, BF16, "embedding_bf16")
        for layer, kind in enumerate(self.text["layer_types"]):
            if kind == "linear_attention":
                hidden, conv, state = self.build_linear_chunk(layer, hidden)
                conv = g.node("Identity", [conv], f"next_conv_state_{layer}", [[1, self.conv_dim, 3]], BF16)
                state = g.node("Identity", [state], f"next_recurrent_state_{layer}",
                               [[1, self.linear_heads, self.linear_key_dim, self.linear_value_dim]], FP32)
                g.output(conv); g.output(state)
            else:
                hidden = self.build_full_layer(layer, hidden, position, mask)
        hidden = self.rms_norm(hidden, "model.language_model.norm.weight", "final_norm")
        hidden = self.slice(hidden, t - 1, t, 1, "last_prompt_hidden")
        hidden = g.reshape(hidden, [1, self.hidden], "lm_head_input")
        weight = embedding
        logits = g.node("Gemm", [hidden, weight], "logits", [[1, self.vocab]], BF16,
                        alpha=1.0, beta=1.0, transA=0, transB=1)
        g.output(logits)
        return g

    def write_fth(self, path):
        self.validate_precision_policy()
        precision = fp8_format_name(self.fp8_dtype) if self.fp8_dtype else "bf16"
        self.g.write_fth(path, f"qwen3.5-0.8b_unified_{precision}_cached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=16)
    parser.add_argument("--fp8-format", choices=("e4m3", "e5m2"))
    parser.add_argument("--fp8-activation-amax", type=float, default=96.0)
    parser.add_argument("--fp8-calibration", type=Path)
    parser.add_argument("--fp8-keep-bf16", action="append", default=[], metavar="CHECKPOINT_KEY_GLOB")
    parser.add_argument("--output-fth", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.prefill_chunk <= args.max_context:
        parser.error("prefill chunk must be between 1 and context capacity")
    shard = resolve_shard(args.model_dir)
    digest = hashlib.sha256()
    with shard.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    fp8 = {"e4m3": FP8E4M3, "e5m2": FP8E5M2}.get(args.fp8_format)
    checkpoint_sha256 = digest.hexdigest()
    calibration = load_calibration(args.fp8_calibration, args.model_dir, args.fp8_format)
    if calibration and calibration["checkpoint_sha256"] != checkpoint_sha256:
        parser.error("calibration checkpoint hash does not match")
    digest.update((args.model_dir / "config.json").read_bytes())
    digest.update(json.dumps({"execution_version": 2, "fp8_format": args.fp8_format,
                              "precision_policy": QWEN_PRECISION_POLICY,
                              "fp8_activation_amax": args.fp8_activation_amax,
                              "calibration": calibration,
                              "keep_bf16": sorted(set(args.fp8_keep_bf16))}, sort_keys=True).encode())
    args.output_fth.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(shard, framework="pt", device="cpu") as source:
        builder = OptimizedQwenBuilder(args.model_dir, args.max_context, fp8_dtype=fp8,
                                       fp8_activation_amax=args.fp8_activation_amax,
                                       fp8_calibration=calibration, fp8_keep_bf16=args.fp8_keep_bf16,
                                       sequence=1, prefill_chunk=args.prefill_chunk,
                                       weights_id=digest.hexdigest())
        builder.source = source
        builder.build()
        builder.g.metadata["qwen.source.checkpoint_sha256"] = checkpoint_sha256
        builder.g.metadata["qwen.source.config_sha256"] = hashlib.sha256(
            (args.model_dir / "config.json").read_bytes()).hexdigest()
        builder.write_fth(args.output_fth)


if __name__ == "__main__":
    main()
