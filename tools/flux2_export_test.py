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

"""Native FTH numerical tests against an independent tiny torch FLUX.2."""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from flux2_graph import FP32, BF16, TensorStore
from export_flux2_model import DenoiserConfig, build_denoiser


def tiny_denoiser_fixture():
    config = DenoiserConfig(in_channels=8, context_in_dim=24, hidden_size=16, num_heads=2,
                            depth=2, depth_single_blocks=2, axes_dim=(2, 2, 2, 2), mlp_ratio=2.0)
    rng = np.random.default_rng(1729)
    arrays = {}

    def weight(key, *shape):
        arrays[key] = (rng.standard_normal(shape) * 0.12).astype(np.float32)

    def norm(prefix):
        weight(prefix + '.query_norm.scale', 8)
        weight(prefix + '.key_norm.scale', 8)
        arrays[prefix + '.query_norm.scale'] += 1
        arrays[prefix + '.key_norm.scale'] += 1

    weight('img_in.weight', 16, 8)
    weight('txt_in.weight', 16, 24)
    weight('time_in.in_layer.weight', 16, 256)
    weight('time_in.out_layer.weight', 16, 16)
    for name, size in [('double_stream_modulation_img', 96), ('double_stream_modulation_txt', 96),
                       ('single_stream_modulation', 48)]:
        weight(name + '.lin.weight', size, 16)
    for i in range(config.depth):
        for stream in ('img', 'txt'):
            prefix = f'double_blocks.{i}.{stream}'
            weight(prefix + '_attn.qkv.weight', 48, 16)
            norm(prefix + '_attn.norm')
            weight(prefix + '_attn.proj.weight', 16, 16)
            weight(prefix + '_mlp.0.weight', 64, 16)
            weight(prefix + '_mlp.2.weight', 16, 32)
    for i in range(config.depth_single_blocks):
        prefix = f'single_blocks.{i}'
        norm(prefix + '.norm')
        weight(prefix + '.linear1.weight', 112, 16)
        weight(prefix + '.linear2.weight', 16, 48)
    weight('final_layer.adaLN_modulation.1.weight', 32, 16)
    weight('final_layer.linear.weight', 8, 16)
    inputs = {'latents': rng.standard_normal((1, 2, 8)).astype(np.float32),
              'prompt_embeds': rng.standard_normal((1, 4, 24)).astype(np.float32),
              'timestep': np.asarray([0.625], np.float32)}
    return config, TensorStore(arrays), inputs


def reference_denoiser(config, store, inputs, dtype=torch.float32):
    weights = {k: torch.tensor(store.load(k), dtype=dtype) for k in store.keys()}
    image = torch.tensor(inputs['latents'], dtype=dtype)
    text = torch.tensor(inputs['prompt_embeds'], dtype=dtype)
    t = torch.tensor(inputs['timestep'], dtype=dtype)
    h, heads, dim = config.hidden_size, config.num_heads, config.hidden_size // config.num_heads

    def linear(x, key):
        return F.linear(x, weights[key + '.weight'])

    def normalize(x):
        return F.layer_norm(x, (h,), eps=1e-6)

    def rms(x, key):
        return (x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(dtype) * weights[key]

    def heads_of(x):
        return x.reshape(1, -1, heads, dim).transpose(1, 2)

    image_ids = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
    text_ids = torch.zeros(4, 4)
    text_ids[:, 3] = torch.arange(4)
    ids = torch.cat([text_ids, image_ids])
    frequencies = torch.cat([ids[:, i:i+1] * config.theta ** (-torch.arange(0, d, 2) / d)
                             for i, d in enumerate(config.axes_dim)], dim=-1)[None, None]

    def rotate(x):
        pairs = x.float().reshape(1, heads, -1, dim // 2, 2)
        a, b = pairs.unbind(-1)
        return torch.stack((a * frequencies.cos() - b * frequencies.sin(),
                            a * frequencies.sin() + b * frequencies.cos()), -1).flatten(-2).to(dtype)

    def attend(q, k, v):
        return F.scaled_dot_product_attention(rotate(q), rotate(k), v).transpose(1, 2).reshape(1, -1, h)

    def qkv(x, prefix):
        q, k, v = linear(x, prefix + '.qkv').chunk(3, -1)
        return (rms(heads_of(q), prefix + '.norm.query_norm.scale'),
                rms(heads_of(k), prefix + '.norm.key_norm.scale'), heads_of(v))

    def mlp(x, prefix):
        gate, value = linear(x, prefix + '.0').chunk(2, -1)
        return linear(F.silu(gate) * value, prefix + '.2')

    frequencies_t = torch.exp(-math.log(10000) * torch.arange(128) / 128)
    args = 1000 * t[:, None] * frequencies_t[None]
    vector = linear(F.silu(linear(torch.cat((args.cos(), args.sin()), -1).to(dtype), 'time_in.in_layer')),
                    'time_in.out_layer')
    mod_img = linear(F.silu(vector), 'double_stream_modulation_img.lin')[:, None].chunk(6, -1)
    mod_txt = linear(F.silu(vector), 'double_stream_modulation_txt.lin')[:, None].chunk(6, -1)
    mod_single = linear(F.silu(vector), 'single_stream_modulation.lin')[:, None].chunk(3, -1)
    image, text = linear(image, 'img_in'), linear(text, 'txt_in')
    for i in range(config.depth):
        iq, ik, iv = qkv((1 + mod_img[1]) * normalize(image) + mod_img[0], f'double_blocks.{i}.img_attn')
        tq, tk, tv = qkv((1 + mod_txt[1]) * normalize(text) + mod_txt[0], f'double_blocks.{i}.txt_attn')
        attention = attend(torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2))
        ta, ia = attention[:, :4], attention[:, 4:]
        image = image + mod_img[2] * linear(ia, f'double_blocks.{i}.img_attn.proj')
        text = text + mod_txt[2] * linear(ta, f'double_blocks.{i}.txt_attn.proj')
        image = image + mod_img[5] * mlp((1 + mod_img[4]) * normalize(image) + mod_img[3], f'double_blocks.{i}.img_mlp')
        text = text + mod_txt[5] * mlp((1 + mod_txt[4]) * normalize(text) + mod_txt[3], f'double_blocks.{i}.txt_mlp')
    x = torch.cat((text, image), 1)
    for i in range(config.depth_single_blocks):
        prefix = f'single_blocks.{i}'
        projected = linear((1 + mod_single[1]) * normalize(x) + mod_single[0], prefix + '.linear1')
        q, k, v, gate, value = projected.split([h, h, h, 2*h, 2*h], -1)
        q = rms(heads_of(q), prefix + '.norm.query_norm.scale')
        k = rms(heads_of(k), prefix + '.norm.key_norm.scale')
        result = torch.cat((attend(q, k, heads_of(v)), F.silu(gate) * value), -1)
        x = x + mod_single[2] * linear(result, prefix + '.linear2')
    shift, scale = linear(F.silu(vector), 'final_layer.adaLN_modulation.1')[:, None].chunk(2, -1)
    return linear((1 + scale) * normalize(x[:, 4:]) + shift, 'final_layer.linear').float().detach().numpy()


class Flux2ExportTest(unittest.TestCase):
    def test_native_full_denoiser_matches_torch(self):
        config, store, inputs = tiny_denoiser_fixture()
        probe = Path(os.environ.get('FEATHER_FLUX2_PROBE', Path(__file__).resolve().parents[1] / 'build_new/bin/flux2_component_probe'))
        self.assertTrue(probe.is_file(), 'build flux2_component_probe before running numerical exporter tests')
        for backend in ('common', 'cuda'):
            if backend == 'cuda' and os.environ.get('FEATHER_TEST_CUDA', '1') == '0':
                continue
            for dtype in (FP32, BF16):
                with self.subTest(backend=backend, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                    expected = reference_denoiser(config, store, inputs,
                                                  torch.float32 if dtype == FP32 else torch.bfloat16)
                    graph = build_denoiser(store, 16, 32, 4, dtype=dtype, config=config)
                    root = Path(directory)
                    graph.write(root / 'model.fth', 'flux2_tiny_denoiser')
                    args = []
                    for name, array in inputs.items():
                        if graph.dtype(name) == BF16:
                            import ml_dtypes
                            array = array.astype(ml_dtypes.bfloat16)
                        array.tofile(root / (name + '.bin'))
                        args.append(f'{name}={root / (name + ".bin")}')
                    run = subprocess.run([str(probe), str(root / 'model.fth'), str(root / 'out'), backend, *args],
                                         capture_output=True, text=True, timeout=90)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    raw = np.fromfile(root / 'out/velocity.bin', dtype=np.float32 if dtype == FP32 else np.uint16)
                    if dtype == BF16:
                        import ml_dtypes
                        raw = raw.view(ml_dtypes.bfloat16).astype(np.float32)
                    np.testing.assert_allclose(raw.reshape(expected.shape), expected,
                                               atol=3e-5 if dtype == FP32 else 0.008, rtol=3e-4 if dtype == FP32 else 0.03)

    def test_bad_resolution_and_weights_fail_before_serialization(self):
        config, store, _ = tiny_denoiser_fixture()
        with self.assertRaisesRegex(ValueError, 'resolution'):
            build_denoiser(store, 15, 32, 4, config=config)
        store.arrays['img_in.weight'] = np.zeros((16, 7), np.float32)
        store.shapes['img_in.weight'] = [16, 7]
        with self.assertRaisesRegex(ValueError, 'img_in'):
            build_denoiser(store, 16, 32, 4, config=config)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
