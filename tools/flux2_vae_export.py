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

"""Export the official BFL FLUX.2 decoder using native Feather operations."""
import numpy as np

from flux2_graph import BF16, FP32, Graph


def _array(store, key):
    value = store.load(key)
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _validate(store, height, width, ch, ch_mult, num_res_blocks, z_channels, groups):
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError('VAE height and width must be positive multiples of 16')
    if len(ch_mult) != 4 or ch <= 0 or z_channels <= 0 or groups <= 0 or num_res_blocks < 0:
        raise ValueError('VAE requires four positive channel multipliers and positive channel/group counts')
    if any(m <= 0 or (ch * m) % groups for m in ch_mult):
        raise ValueError('VAE channels must be positive and divisible by groups')

    def expect(key, shape):
        if key not in store:
            raise ValueError(f'missing VAE checkpoint tensor {key}')
        actual = tuple(store.shape(key))
        if actual != tuple(shape):
            raise ValueError(f'{key}: expected shape {tuple(shape)}, got {actual}')

    def conv(prefix, cin, cout, kernel=3):
        expect(prefix + '.weight', (cout, cin, kernel, kernel))
        expect(prefix + '.bias', (cout,))

    def norm(prefix, channels):
        expect(prefix + '.weight', (channels,))
        expect(prefix + '.bias', (channels,))

    def residual(prefix, cin, cout):
        norm(prefix + '.norm1', cin)
        conv(prefix + '.conv1', cin, cout)
        norm(prefix + '.norm2', cout)
        conv(prefix + '.conv2', cout, cout)
        if cin != cout:
            conv(prefix + '.nin_shortcut', cin, cout, 1)

    expect('bn.running_mean', (4 * z_channels,))
    expect('bn.running_var', (4 * z_channels,))
    conv('decoder.post_quant_conv', z_channels, z_channels, 1)
    channels = ch * ch_mult[-1]
    conv('decoder.conv_in', z_channels, channels)
    residual('decoder.mid.block_1', channels, channels)
    norm('decoder.mid.attn_1.norm', channels)
    for name in ('q', 'k', 'v', 'proj_out'):
        conv('decoder.mid.attn_1.' + name, channels, channels, 1)
    residual('decoder.mid.block_2', channels, channels)
    for level in reversed(range(len(ch_mult))):
        cout = ch * ch_mult[level]
        for block in range(num_res_blocks + 1):
            residual(f'decoder.up.{level}.block.{block}', channels, cout)
            channels = cout
        if level:
            conv(f'decoder.up.{level}.upsample.conv', channels, channels)
    norm('decoder.norm_out', channels)
    conv('decoder.conv_out', channels, 3)
    variance = _array(store, 'bn.running_var')
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError('bn.running_var must be finite and nonnegative')
    if not np.all(np.isfinite(_array(store, 'bn.running_mean'))):
        raise ValueError('bn.running_mean must be finite')


def build_vae_decoder(store, height, width, dtype=FP32, ch=128,
                      ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=32, groups=32):
    """Build a fixed-resolution decoder accepting packed [B, H*W/256, 4Z].

    Checkpoint prefixes match ``flux2.autoencoder.AutoEncoder.state_dict``.
    GroupNorm is lowered to a spatial/channel LayerNormalization per group.
    """
    _validate(store, height, width, ch, ch_mult, num_res_blocks, z_channels, groups)
    if dtype not in (FP32, BF16):
        raise ValueError('FLUX.2 VAE supports FP32 or BF16 export')
    g = Graph(store, dtype)

    def swish(x):
        return g.binary_op('Mul', x, g.unary_op('Sigmoid', x, 'swish_sigmoid'), 'swish')

    def norm(x, prefix):
        b, channels, h, w = g.shape(x)
        # Each group includes its contiguous C/G channels and every spatial element.
        flattened = g.reshape(x, [b, groups, channels // groups * h * w], 'group_norm_flat')
        normalized = g.normalize(flattened, epsilon=1e-6, layer=True, axis=-1)
        normalized = g.reshape(normalized, [b, channels, h, w], 'group_norm_spatial')
        weight = g.weight(prefix + '.weight', reshape=[1, channels, 1, 1])
        bias = g.weight(prefix + '.bias', reshape=[1, channels, 1, 1])
        return g.binary_op('Add', g.binary_op('Mul', normalized, weight, 'group_norm_scale'), bias,
                           'group_norm_bias')

    def residual(x, prefix):
        h = g.conv2d(swish(norm(x, prefix + '.norm1')), prefix + '.conv1')
        h = g.conv2d(swish(norm(h, prefix + '.norm2')), prefix + '.conv2')
        if g.shape(x)[1] != g.shape(h)[1]:
            x = g.conv2d(x, prefix + '.nin_shortcut', padding=0)
        return g.binary_op('Add', x, h, 'residual')

    def attention(x, prefix):
        h = norm(x, prefix + '.norm')
        b, channels, ih, iw = g.shape(h)
        qkv = []
        for key in ('q', 'k', 'v'):
            value = g.conv2d(h, prefix + '.' + key, padding=0)
            value = g.reshape(value, [b, 1, channels, ih * iw], key + '_flat')
            value = g.transpose(value, [0, 1, 3, 2], [b, 1, ih * iw, channels], key + '_tokens')
            qkv.append(value)
        h = g.attention(*qkv)
        h = g.transpose(h, [0, 1, 3, 2], [b, 1, channels, ih * iw], 'attention_channels')
        h = g.reshape(h, [b, channels, ih, iw], 'attention_spatial')
        h = g.conv2d(h, prefix + '.proj_out', padding=0)
        return g.binary_op('Add', x, h, 'attention_residual')

    ih, iw = height // 16, width // 16
    packed = 4 * z_channels
    x = g.input('latents', [1, ih * iw, packed], dtype)
    # The native BN buffers are FP32; inverse normalization precedes unpacking.
    x = g.cast(x, g.shape(x), FP32, 'bn_input_fp32')
    mean = g.const('vae_bn_mean', _array(store, 'bn.running_mean').reshape(1, 1, packed))
    scale = g.const('vae_bn_scale', np.sqrt(_array(store, 'bn.running_var') + 1e-4).reshape(1, 1, packed))
    x = g.binary_op('Add', g.binary_op('Mul', x, scale, 'inverse_bn_scale'), mean, 'inverse_bn_mean')
    x = g.reshape(x, [1, ih, iw, z_channels, 2, 2], 'packed_latents')
    x = g.transpose(x, [0, 3, 1, 4, 2, 5], [1, z_channels, ih, 2, iw, 2], 'unpacked_latents')
    x = g.reshape(x, [1, z_channels, ih * 2, iw * 2], 'latent_image')
    x = g.cast(x, g.shape(x), dtype, 'decoder_input_dtype')
    x = g.conv2d(x, 'decoder.post_quant_conv', padding=0)
    x = g.conv2d(x, 'decoder.conv_in')
    x = residual(x, 'decoder.mid.block_1')
    x = attention(x, 'decoder.mid.attn_1')
    x = residual(x, 'decoder.mid.block_2')
    for level in reversed(range(len(ch_mult))):
        for block in range(num_res_blocks + 1):
            x = residual(x, f'decoder.up.{level}.block.{block}')
        if level:
            b, channels, h, w = g.shape(x)
            scales = g.const(g.fresh('upsample_scales'), np.asarray([1, 1, 2, 2], dtype=np.float32))
            x = g.node('Resize', [x, '', scales], g.fresh('nearest_upsample'),
                       [[b, channels, h * 2, w * 2]], g.dtype(x), mode='nearest',
                       coordinate_transformation_mode='asymmetric', nearest_mode='floor')
            x = g.conv2d(x, f'decoder.up.{level}.upsample.conv')
    x = g.conv2d(swish(norm(x, 'decoder.norm_out')), 'decoder.conv_out')
    x = g.cast(x, g.shape(x), FP32, 'image_fp32')
    g.node('Identity', [x], 'image', [[1, 3, height, width]], FP32)
    g.output('image')
    g.metadata['flux2.component'] = 'vae_decoder'
    return g
