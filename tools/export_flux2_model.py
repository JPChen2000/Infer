#!/usr/bin/env python3
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

"""Export official FLUX.2 Klein 4B native checkpoints to three Feather graphs.

Python is needed only for conversion. The resulting text encoder, flow model
and VAE decoder run in C++ without torch or diffusers.
"""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from flux2_graph import Graph, TensorStore, FP32, BF16


@dataclass(frozen=True)
class DenoiserConfig:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: tuple = (32, 32, 32, 32)
    theta: float = 2000.0
    mlp_ratio: float = 3.0


def component_metadata(graph, component, height, width, text_length, context_dim, latent_channels):
    graph.metadata.update({'flux2.schema': 1, 'flux2.component': component, 'flux2.variant': 'klein-4b',
                           'flux2.height': height, 'flux2.width': width, 'flux2.text_length': text_length,
                           'flux2.context_dim': context_dim, 'flux2.latent_channels': latent_channels})


def build_denoiser(store, height, width, text_length, dtype=FP32, config=None):
    """Lower BFL double/single-stream blocks to fixed-shape native operators."""
    c = config or DenoiserConfig()
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError('resolution must be positive multiples of 16')
    if (text_length <= 0 or c.hidden_size <= 0 or c.num_heads <= 0 or c.hidden_size % c.num_heads
            or c.depth < 0 or c.depth_single_blocks < 0 or c.in_channels <= 0 or c.context_in_dim <= 0
            or len(c.axes_dim) != 4 or any(d <= 0 or d % 2 for d in c.axes_dim)
            or sum(c.axes_dim) != c.hidden_size // c.num_heads or c.mlp_ratio <= 0
            or not math.isfinite(c.mlp_ratio) or not math.isfinite(c.theta) or c.theta <= 0):
        raise ValueError('invalid FLUX.2 denoiser configuration')
    h, heads, dim = c.hidden_size, c.num_heads, c.hidden_size // c.num_heads
    intermediate = int(h * c.mlp_ratio)
    image_length = height // 16 * (width // 16)
    total = text_length + image_length

    def expect(key, shape):
        if store.shape(key) != list(shape):
            raise ValueError(f'{key}: expected shape {list(shape)}, got {store.shape(key)}')

    expect('img_in.weight', (h, c.in_channels))
    expect('txt_in.weight', (h, c.context_in_dim))
    expect('time_in.in_layer.weight', (h, 256))
    expect('time_in.out_layer.weight', (h, h))
    for name, multiplier in [('double_stream_modulation_img', 6), ('double_stream_modulation_txt', 6),
                             ('single_stream_modulation', 3)]:
        expect(name + '.lin.weight', (multiplier * h, h))
    for i in range(c.depth):
        for stream in ('img', 'txt'):
            prefix = f'double_blocks.{i}.{stream}'
            expect(prefix + '_attn.qkv.weight', (3 * h, h))
            expect(prefix + '_attn.proj.weight', (h, h))
            expect(prefix + '_mlp.0.weight', (2 * intermediate, h))
            expect(prefix + '_mlp.2.weight', (h, intermediate))
            for kind in ('query', 'key'):
                expect(prefix + f'_attn.norm.{kind}_norm.scale', (dim,))
    for i in range(c.depth_single_blocks):
        prefix = f'single_blocks.{i}'
        expect(prefix + '.linear1.weight', (3 * h + 2 * intermediate, h))
        expect(prefix + '.linear2.weight', (h, h + intermediate))
        for kind in ('query', 'key'):
            expect(prefix + f'.norm.{kind}_norm.scale', (dim,))
    expect('final_layer.adaLN_modulation.1.weight', (2 * h, h))
    expect('final_layer.linear.weight', (c.in_channels, h))

    g = Graph(store, dtype)
    image = g.input('latents', [1, image_length, c.in_channels], dtype)
    text = g.input('prompt_embeds', [1, text_length, c.context_in_dim], dtype)
    timestep = g.input('timestep', [1], FP32)
    one = g.constant_float([1], 'one')

    def chunks(x, sizes):
        result, offset = [], 0
        for size in sizes:
            result.append(g.slice_axis(x, offset, offset + size, -1))
            offset += size
        return result

    def modulation(prefix, count):
        value = g.linear(g.unary_op('SiLU', vector), prefix + '.lin.weight')
        return [g.reshape(x, [1, 1, h], 'modulation') for x in chunks(value, [h] * count)]

    def modulated(x, shift, scale):
        return g.binary_op('Add', g.binary_op('Mul', g.normalize(x), g.binary_op('Add', one, scale)), shift)

    def residual(x, value, gate):
        return g.binary_op('Add', x, g.binary_op('Mul', gate, value))

    def into_heads(x):
        length = g.shape(x)[1]
        x = g.reshape(x, [1, length, heads, dim], 'split_heads')
        return g.transpose(x, [0, 2, 1, 3], [1, heads, length, dim], 'heads')

    def rms(x, key):
        # BFL normalizes in FP32, casts back, then applies the learned scale.
        return g.binary_op('Mul', g.normalize(x, layer=False), g.weight(key))

    text_ids = np.zeros((text_length, 4), np.float32)
    text_ids[:, 3] = np.arange(text_length)
    image_ids = np.zeros((image_length, 4), np.float32)
    image_ids[:, 1] = np.repeat(np.arange(height // 16), width // 16)
    image_ids[:, 2] = np.tile(np.arange(width // 16), height // 16)
    ids = np.concatenate((text_ids, image_ids))
    angles = np.concatenate([ids[:, axis:axis + 1] * np.power(c.theta, -np.arange(0, axis_dim, 2, dtype=np.float32) / axis_dim)
                             for axis, axis_dim in enumerate(c.axes_dim)], -1).astype(np.float32)
    cosine = g.const('rope_cos', np.cos(angles)[None, None, :, :, None])
    sine = g.const('rope_sin', np.sin(angles)[None, None, :, :, None])

    def rope(x):
        original_shape = g.shape(x)
        pairs = g.reshape(g.cast(x, original_shape, FP32, 'rope_fp32'), [1, heads, total, dim // 2, 2], 'rope_pairs')
        a, b = chunks(pairs, [1, 1])
        real = g.binary_op('Sub', g.binary_op('Mul', a, cosine), g.binary_op('Mul', b, sine))
        imag = g.binary_op('Add', g.binary_op('Mul', a, sine), g.binary_op('Mul', b, cosine))
        result = g.concat([real, imag], 4, [1, heads, total, dim // 2, 2], FP32, 'rope_interleave')
        result = g.reshape(result, original_shape, 'rope_merge')
        return g.cast(result, original_shape, dtype, 'rope_dtype')

    def attend(q, k, v):
        result = g.attention(rope(q), rope(k), v)
        result = g.transpose(result, [0, 2, 1, 3], [1, total, heads, dim], 'attended')
        return g.reshape(result, [1, total, h], 'merge_heads')

    def qkv(x, prefix):
        q, k, v = [into_heads(x) for x in chunks(g.linear(x, prefix + '.qkv.weight'), [h] * 3)]
        return rms(q, prefix + '.norm.query_norm.scale'), rms(k, prefix + '.norm.key_norm.scale'), v

    def gated_mlp(x, prefix):
        gate, value = chunks(g.linear(x, prefix + '.0.weight'), [intermediate] * 2)
        return g.linear(g.binary_op('Mul', g.unary_op('SiLU', gate), value), prefix + '.2.weight')

    # BFL scales in the model dtype before computing FP32 frequencies. In
    # BF16, e.g. .625 * 1000 rounds to 624; folding 1000 into the frequencies
    # changes the model's time conditioning.
    time = g.cast(timestep, [1], dtype, 'timestep_dtype')
    time = g.binary_op('Mul', time, g.constant_float([1000], 'time_scale'))
    time = g.cast(time, [1], FP32, 'scaled_timestep_fp32')
    time_freq = np.exp(-math.log(10000) * np.arange(128, dtype=np.float32) / 128)[None]
    phase = g.binary_op('Mul', g.reshape(time, [1, 1], 'timestep_scalar'), g.const('time_frequency', time_freq))
    embedding = g.concat([g.unary_op('Cos', phase), g.unary_op('Sin', phase)], 1, [1, 256], FP32, 'time_embedding')
    embedding = g.cast(embedding, [1, 256], dtype, 'time_dtype')
    vector = g.linear(g.unary_op('SiLU', g.linear(embedding, 'time_in.in_layer.weight')), 'time_in.out_layer.weight')
    mod_img, mod_txt = modulation('double_stream_modulation_img', 6), modulation('double_stream_modulation_txt', 6)
    mod_single = modulation('single_stream_modulation', 3)
    image, text = g.linear(image, 'img_in.weight'), g.linear(text, 'txt_in.weight')
    for i in range(c.depth):
        prefix = f'double_blocks.{i}'
        iq, ik, iv = qkv(modulated(image, *mod_img[:2]), prefix + '.img_attn')
        tq, tk, tv = qkv(modulated(text, *mod_txt[:2]), prefix + '.txt_attn')
        merged = [g.concat([a, b], 2, [1, heads, total, dim], dtype, 'joint_qkv')
                  for a, b in ((tq, iq), (tk, ik), (tv, iv))]
        attention = attend(*merged)
        ta = g.slice_axis(attention, 0, text_length, 1)
        ia = g.slice_axis(attention, text_length, total, 1)
        image = residual(image, g.linear(ia, prefix + '.img_attn.proj.weight'), mod_img[2])
        text = residual(text, g.linear(ta, prefix + '.txt_attn.proj.weight'), mod_txt[2])
        image = residual(image, gated_mlp(modulated(image, *mod_img[3:5]), prefix + '.img_mlp'), mod_img[5])
        text = residual(text, gated_mlp(modulated(text, *mod_txt[3:5]), prefix + '.txt_mlp'), mod_txt[5])
    x = g.concat([text, image], 1, [1, total, h], dtype, 'join_streams')
    for i in range(c.depth_single_blocks):
        prefix = f'single_blocks.{i}'
        projected = g.linear(modulated(x, *mod_single[:2]), prefix + '.linear1.weight')
        q, k, v, gate, value = chunks(projected, [h, h, h, intermediate, intermediate])
        q = rms(into_heads(q), prefix + '.norm.query_norm.scale')
        k = rms(into_heads(k), prefix + '.norm.key_norm.scale')
        attn = attend(q, k, into_heads(v))
        mlp = g.binary_op('Mul', g.unary_op('SiLU', gate), value)
        result = g.concat([attn, mlp], 2, [1, total, h + intermediate], dtype, 'attention_mlp')
        x = residual(x, g.linear(result, prefix + '.linear2.weight'), mod_single[2])
    x = g.slice_axis(x, text_length, total, 1)
    shift, scale = chunks(g.linear(g.unary_op('SiLU', vector), 'final_layer.adaLN_modulation.1.weight'), [h, h])
    shift, scale = [g.reshape(v, [1, 1, h], 'final_modulation') for v in (shift, scale)]
    output = g.linear(modulated(x, shift, scale), 'final_layer.linear.weight')
    g.node('Identity', [output], 'velocity', [g.shape(output)], dtype)
    g.output('velocity')
    component_metadata(g, 'denoiser', height, width, text_length, c.context_in_dim, c.in_channels)
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--denoiser', required=True, type=Path, help='BFL flux-2-klein-4b.safetensors (not diffusers transformer shards)')
    parser.add_argument('--vae', required=True, type=Path, help='BFL ae.safetensors')
    parser.add_argument('--text-encoder-dir', required=True, type=Path, help='unquantized HF Qwen3-4B snapshot including tokenizer')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--text-length', type=int, default=512)
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    args = parser.parse_args()
    from flux2_text_export import build_text_encoder
    from flux2_vae_export import build_vae_decoder
    dtype = BF16 if args.dtype == 'bf16' else FP32
    config = json.loads((args.text_encoder_dir / 'config.json').read_text())
    tokenizer_config = json.loads((args.text_encoder_dir / 'tokenizer_config.json').read_text())
    for name in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        if not (args.text_encoder_dir / name).is_file():
            parser.error(f'text encoder snapshot must include {name}')
    if config.get('hidden_size') != 2560:
        parser.error('Klein 4B requires the Qwen3-4B text encoder (hidden_size=2560)')
    padding_side = tokenizer_config.get('padding_side', 'right')
    if padding_side not in ('left', 'right'):
        parser.error('unsupported tokenizer padding_side')
    token_data = json.loads((args.text_encoder_dir / 'tokenizer.json').read_text())
    vocabulary = token_data.get('model', {}).get('vocab', {})
    vocabulary.update({v['content']: v['id'] for v in token_data.get('added_tokens', [])})
    pad_token = tokenizer_config.get('pad_token', '<|endoftext|>')
    if isinstance(pad_token, dict):  # HF AddedToken serialization
        pad_token = pad_token.get('content')
    if not isinstance(pad_token, str):
        parser.error('tokenizer pad_token must be a string or AddedToken object')
    pad_id = vocabulary.get(pad_token)
    if not isinstance(pad_id, int) or pad_id < 0 or pad_id >= config['vocab_size']:
        parser.error('tokenizer pad token does not belong to text encoder vocabulary')
    # Build and check all descriptors before writing any model to the destination.
    graphs = {
        'text_encoder': build_text_encoder(TensorStore(args.text_encoder_dir), config, args.text_length, dtype),
        'denoiser': build_denoiser(TensorStore(args.denoiser), args.height, args.width, args.text_length, dtype),
        'vae_decoder': build_vae_decoder(TensorStore(args.vae), args.height, args.width, dtype),
    }
    for component, graph in graphs.items():
        component_metadata(graph, component, args.height, args.width, args.text_length, 7680, 128)
    graphs['text_encoder'].metadata.update({'flux2.pad_token_id': pad_id, 'flux2.padding_side': padding_side})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.flux2-export-', dir=args.output_dir) as temporary:
        staging = Path(temporary)
        for component, graph in graphs.items():
            print(f'exporting {component}: {len(graph.nodes)} nodes', flush=True)
            graph.write(staging / (component + '.fth'), 'flux2_klein_4b_' + component)
        for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json'):
            source = args.text_encoder_dir / name
            if source.is_file():
                shutil.copy2(source, staging / name)
        for source in staging.iterdir():
            source.replace(args.output_dir / source.name)
    print(f'exported FLUX.2 Klein 4B to {args.output_dir}')


if __name__ == '__main__':
    main()
