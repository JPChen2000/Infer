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

"""Tiny independent torch reference for the native BFL FLUX.2 VAE decoder."""
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


class TensorStore:
    def __init__(self, tensors):
        self.tensors = tensors

    def keys(self):
        return self.tensors.keys()

    def __contains__(self, key):
        return key in self.tensors

    def shape(self, key):
        return tuple(self.tensors[key].shape)

    def load(self, key):
        return self.tensors[key]


class Residual(nn.Module):
    def __init__(self, cin, cout, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, eps=1e-6)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout, eps=1e-6)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        if cin != cout:
            self.nin_shortcut = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + (self.nin_shortcut(x) if hasattr(self, 'nin_shortcut') else x)


class Attention(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = [layer(h).flatten(2).transpose(1, 2)[:, None] for layer in (self.q, self.k, self.v)]
        # Explicit math is independent of the runtime attention implementation.
        scores = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
        out = (scores.softmax(-1) @ v)[:, 0].transpose(1, 2).reshape(x.shape)
        return x + self.proj_out(out)


class TinyDecoder(nn.Module):
    def __init__(self, ch=4, ch_mult=(1, 2, 2, 2), num_res_blocks=1, z_channels=2, groups=2):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        channels = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, channels, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = Residual(channels, channels, groups)
        self.mid.attn_1 = Attention(channels, groups)
        self.mid.block_2 = Residual(channels, channels, groups)
        self.up = nn.ModuleList()
        for level in reversed(range(len(ch_mult))):
            stage = nn.Module()
            blocks = []
            for _ in range(num_res_blocks + 1):
                blocks.append(Residual(channels, ch * ch_mult[level], groups))
                channels = ch * ch_mult[level]
            stage.block = nn.ModuleList(blocks)
            if level:
                stage.upsample = nn.Module()
                stage.upsample.conv = nn.Conv2d(channels, channels, 3, padding=1)
            self.up.insert(0, stage)
        self.norm_out = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                h = block(h)
            if level:
                h = self.up[level].upsample.conv(F.interpolate(h, scale_factor=2, mode='nearest'))
        return self.conv_out(F.silu(self.norm_out(h)))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = TinyDecoder()
        self.bn = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.bn.running_mean.copy_(torch.linspace(-0.4, 0.3, 8))
        self.bn.running_var.copy_(torch.linspace(0.2, 1.6, 8))

    def forward(self, tokens, height=16, width=32):
        b, _, packed = tokens.shape
        ih, iw = height // 16, width // 16
        z = tokens.transpose(1, 2).reshape(b, packed, ih, iw)
        z = z * (self.bn.running_var[None, :, None, None] + 1e-4).sqrt()
        z = z + self.bn.running_mean[None, :, None, None]
        z = z.reshape(b, packed // 4, 2, 2, ih, iw).permute(0, 1, 4, 2, 5, 3)
        return self.decoder(z.reshape(b, packed // 4, ih * 2, iw * 2).to(next(self.decoder.parameters()).dtype))


def tiny_fixture():
    torch.manual_seed(281)
    model = TinyVAE().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                module.weight.uniform_(0.6, 1.4)
                module.bias.uniform_(-0.2, 0.2)
    tokens = torch.randn(1, 2, 8)
    return model, tokens, TensorStore(model.state_dict())


class VAEDecoderExportTest(unittest.TestCase):
    def build(self, store, **kwargs):
        from flux2_vae_export import build_vae_decoder
        return build_vae_decoder(store, 16, 32, ch=4, ch_mult=(1, 2, 2, 2),
                                 num_res_blocks=1, z_channels=2, groups=2, **kwargs)

    def test_export_contains_native_decoder_and_output_contract(self):
        model, tokens, store = tiny_fixture()
        graph = self.build(store)
        self.assertEqual(graph.value_info['latents'][0], [1, 2, 8])
        self.assertEqual(graph.value_info['image'][0], [1, 3, 16, 32])
        self.assertEqual(graph.outputs, ['image'])
        self.assertTrue(any(n.op_type == 'ScaledDotProductAttention' for n in graph.nodes))
        self.assertTrue(any(n.op_type == 'LayerNormalization' for n in graph.nodes))
        self.assertFalse(any(n.op_type == 'GroupNormalization' for n in graph.nodes))

    def test_bad_checkpoint_shape_rejected(self):
        _, _, store = tiny_fixture()
        store.tensors['decoder.post_quant_conv.weight'] = torch.zeros(2, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, 'post_quant_conv'):
            self.build(store)

    def test_invalid_bn_statistics_rejected(self):
        _, _, store = tiny_fixture()
        store.tensors['bn.running_var'][3] = -0.1
        with self.assertRaisesRegex(ValueError, 'running_var'):
            self.build(store)

    def test_bf16_decoder_retains_fp32_output(self):
        from flux2_graph import BF16, FP32
        _, _, store = tiny_fixture()
        graph = self.build(store, dtype=BF16)
        self.assertEqual(graph.value_info['latents'][1], BF16)
        self.assertEqual(graph.value_info['image'][1], FP32)
        conv = next(node for node in graph.nodes if node.op_type == 'Conv2D')
        self.assertEqual(graph.value_info[conv.inputs[0]][1], BF16)

    def test_native_common_and_cuda_reference(self):
        from flux2_graph import BF16, FP32
        import ml_dtypes
        probe = Path(os.environ.get('FEATHER_FLUX2_PROBE',
                     str(Path(__file__).resolve().parents[1] / 'build_new/bin/flux2_component_probe')))
        self.assertTrue(probe.is_file(), 'build flux2_component_probe before running numerical exporter tests')
        model, tokens, store = tiny_fixture()
        for dtype in (FP32, BF16):
            graph = self.build(store, dtype=dtype)
            if dtype == BF16:
                model.decoder.bfloat16()
                reference_tokens = tokens.bfloat16().float()
            else:
                reference_tokens = tokens
            expected = model(reference_tokens).float().detach().numpy()
            with tempfile.TemporaryDirectory(prefix='flux2-vae-') as directory:
                root = Path(directory)
                graph.write(root / 'vae.fth', 'flux2_vae_decoder')
                raw = np.asarray(tokens.numpy(), dtype=ml_dtypes.bfloat16 if dtype == BF16 else np.float32)
                raw.tofile(root / 'latents.bin')
                for backend in ('common', 'cuda'):
                    if backend == 'cuda' and os.environ.get('FEATHER_TEST_CUDA', '1') == '0':
                        continue
                    with self.subTest(dtype=dtype, backend=backend):
                        result = subprocess.run([str(probe), str(root / 'vae.fth'),
                                                 str(root / backend), backend,
                                                 'latents=' + str(root / 'latents.bin')],
                                                text=True, capture_output=True, timeout=90)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        actual = np.fromfile(root / backend / 'image.bin', dtype=np.float32).reshape(expected.shape)
                        np.testing.assert_allclose(actual, expected, rtol=0.15 if dtype == BF16 else 3e-4,
                                                   atol=0.08 if dtype == BF16 else 3e-5)

    def test_full_torch_reference(self):
        _, tokens, store = tiny_fixture()
        graph = self.build(store)
        model = TinyVAE().eval()
        model.load_state_dict(store.tensors)
        actual = evaluate_graph(graph, tokens)
        torch.testing.assert_close(actual, model(tokens), rtol=2e-4, atol=2e-5)


def evaluate_graph(graph, tokens):
    values = {'latents': tokens}
    for name in graph.constant_names:
        if name in graph.constant_values:
            a = graph.constant_values[name]
        else:
            a = graph.external_constant_loaders[name]()
        values[name] = torch.from_numpy(np.asarray(a).copy())
    for node in graph.nodes:
        args = [values[name] if name else None for name in node.inputs]
        attr = node.attributes
        op = node.op_type
        if op == 'Reshape':
            y = args[0].reshape(args[1].tolist())
        elif op == 'Transpose':
            y = args[0].permute(attr['perm']).contiguous()
        elif op == 'Add':
            y = args[0] + args[1]
        elif op == 'Mul':
            y = args[0] * args[1]
        elif op == 'Sigmoid':
            y = args[0].sigmoid()
        elif op == 'Cast':
            y = args[0].float()
        elif op == 'Identity':
            y = args[0]
        elif op in ('Conv', 'Conv2D'):
            y = F.conv2d(args[0], args[1], args[2] if len(args) > 2 else None,
                         stride=attr.get('strides', [attr.get('stride_h', 1), attr.get('stride_w', 1)]),
                         padding=attr.get('pads', [attr.get('pad_h', 0)])[0])
        elif op == 'LayerNormalization':
            axis = attr.get('axis', -1)
            shape = args[0].shape[axis:]
            weight = args[1].expand(shape) if len(args) > 1 else None
            bias = args[2].expand(shape) if len(args) > 2 else None
            y = F.layer_norm(args[0], shape, weight, bias, attr.get('epsilon', 1e-5))
        elif op == 'ScaledDotProductAttention':
            y = F.scaled_dot_product_attention(*args[:3])
        elif op == 'Resize':
            y = F.interpolate(args[0], scale_factor=2, mode='nearest')
        else:
            raise AssertionError(f'No torch interpreter for {op}: {attr}')
        values[node.outputs[0]] = y
    return values['image']


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
