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

"""Fixed-shape graph helpers and lazy native safetensors access for FLUX.2."""
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from export_qwen35_model import AtomicGraph, FP32, BF16, INT64 as INT64, write_feather_model


class TensorStore:
    """Read one source tensor at a time; never load a complete checkpoint."""
    def __init__(self, source):
        self.arrays = source if isinstance(source, dict) else None
        self.files = {}
        self.shapes = {}
        if self.arrays is not None:
            self.shapes = {name: list(value.shape) for name, value in source.items()}
            return
        source = Path(source)
        paths = sorted(source.glob('*.safetensors')) if source.is_dir() else [source]
        if not paths:
            raise ValueError(f'no safetensors checkpoint in {source}')
        for path in paths:
            with safe_open(path, framework='pt', device='cpu') as handle:
                for name in handle.keys():
                    if name in self.files:
                        raise ValueError(f'duplicate checkpoint key: {name}')
                    dtype = handle.get_slice(name).get_dtype()
                    if dtype not in ('F32', 'F16', 'BF16', 'I64', 'I32'):
                        raise ValueError(f'{name}: unsupported source dtype {dtype}; use unquantized weights')
                    self.files[name] = path
                    self.shapes[name] = handle.get_slice(name).get_shape()

    def __contains__(self, key):
        return key in self.shapes

    def keys(self):
        return self.shapes.keys()

    def shape(self, key):
        if key not in self.shapes:
            raise ValueError(f'missing checkpoint tensor: {key}')
        return list(self.shapes[key])

    def load(self, key):
        self.shape(key)
        if self.arrays is not None:
            value = self.arrays[key]
        else:
            with safe_open(self.files[key], framework='pt', device='cpu') as handle:
                value = handle.get_tensor(key)
        if hasattr(value, 'detach'):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)


class Graph(AtomicGraph):
    def __init__(self, store, dtype=FP32):
        super().__init__()
        if dtype not in (FP32, BF16):
            raise ValueError('FLUX.2 export supports fp32 and bf16')
        self.store = store
        self.float_dtype = dtype

    def shape(self, value):
        return list(self.value_info[value][0])

    def dtype(self, value):
        return self.value_info[value][1]

    def weight(self, key, transpose=False, reshape=None, force_dtype=None):
        shape = self.store.shape(key)
        if transpose:
            if len(shape) != 2:
                raise ValueError(f'transposed weight must be a matrix: {key}')
            shape = shape[::-1]
        if reshape is not None:
            if int(np.prod(shape)) != int(np.prod(reshape)):
                raise ValueError(f'invalid weight reshape: {key}')
            shape = list(reshape)
        dtype = force_dtype or self.float_dtype
        name = 'weight.' + key + ('.T' if transpose else '')
        name += '.' + str(dtype) + ('.' + 'x'.join(map(str, shape)) if reshape is not None else '')

        def load():
            value = self.store.load(key)
            if transpose:
                value = value.T
            value = value.reshape(shape)
            return np.ascontiguousarray(value, dtype=ml_dtypes.bfloat16 if dtype == BF16 else np.float32)

        return self.external_const(name, shape, dtype, load)

    def unary_op(self, op, x, stem=''):
        return self.unary(op, x, self.shape(x), self.dtype(x), stem or op.lower())

    def binary_op(self, op, a, b, stem=''):
        shape = list(np.broadcast_shapes(tuple(self.shape(a)), tuple(self.shape(b))))
        if self.dtype(a) != self.dtype(b):
            raise ValueError(f'{op}: mixed dtypes require explicit Cast')
        return self.binary(op, a, b, shape, self.dtype(a), stem or op.lower())

    def constant_float(self, value, stem='constant', dtype=None):
        dtype = dtype or self.float_dtype
        return self.const(self.fresh(stem), np.asarray(value, dtype=ml_dtypes.bfloat16 if dtype == BF16 else np.float32))

    def linear(self, x, key, stem='', bias=False):
        shape, weight_shape = self.shape(x), self.store.shape(key)
        if len(weight_shape) != 2 or shape[-1] != weight_shape[1]:
            raise ValueError(f'linear {key}: input {shape}, weight {weight_shape}')
        w = self.weight(key, transpose=True)
        y = self.node('MatMul', [x, w], self.fresh(stem or key), [shape[:-1] + [weight_shape[0]]], self.dtype(x))
        if bias:
            bias_key = key.removesuffix('.weight') + '.bias'
            if self.store.shape(bias_key) != [weight_shape[0]]:
                raise ValueError(f'invalid linear bias: {bias_key}')
            y = self.binary_op('Add', y, self.weight(bias_key), stem)
        return y

    def normalize(self, x, epsilon=1e-6, weight_key=None, bias_key=None, layer=True, axis=-1):
        shape = self.shape(x)
        w = self.weight(weight_key, force_dtype=FP32) if weight_key else self.constant_float([1.0], 'norm_one', FP32)
        args = [x, w]
        if bias_key:
            args.append(self.weight(bias_key, force_dtype=FP32))
        return self.node('LayerNormalization' if layer else 'RMSNormalization', args,
                         self.fresh('norm'), [shape], self.dtype(x), axis=axis, epsilon=float(epsilon))

    def slice_axis(self, x, start, end, axis, stem=''):
        shape = self.shape(x)
        axis %= len(shape)
        if not 0 <= start < end <= shape[axis]:
            raise ValueError(f'invalid slice {start}:{end} along {axis} in {shape}')
        shape[axis] = end - start
        return self.node('Slice', [x], self.fresh(stem or 'slice'), [shape], self.dtype(x),
                         axis=axis, start=int(start), end=int(end))

    def attention(self, q, k, v, mask=None, causal=False):
        shape = self.shape(q)[:-1] + [self.shape(v)[-1]]
        args = [q, k, v] + ([] if mask is None else [mask])
        return self.node('ScaledDotProductAttention', args, self.fresh('attention'), [shape], self.dtype(q), causal=int(causal))

    def conv2d(self, x, prefix, padding=1, stride=1):
        shape = self.shape(x)
        wshape = self.store.shape(prefix + '.weight')
        if len(shape) != 4 or len(wshape) != 4 or shape[1] != wshape[1]:
            raise ValueError(f'invalid convolution {prefix}: {shape}, {wshape}')
        result = [shape[0], wshape[0]] + [(shape[i + 2] + 2 * padding - wshape[i + 2]) // stride + 1 for i in range(2)]
        inputs = [x, self.weight(prefix + '.weight')]
        if prefix + '.bias' in self.store:
            inputs.append(self.weight(prefix + '.bias'))
        y = self.node('Conv2D', inputs, self.fresh(prefix), [result], self.dtype(x),
                      stride_h=stride, stride_w=stride, pad_h=padding, pad_w=padding, group=1)
        # Native convolution descriptors use NCHW, unlike token tensors (ND).
        return y

    def write(self, path, name):
        values, nodes, payloads = self.fth_components()
        convolution_values = set()
        for node in self.nodes:
            if node.op_type in ('Conv2D', 'Resize'):
                convolution_values.update(node.inputs[:1] + node.outputs)
                if node.op_type == 'Conv2D':
                    convolution_values.add(node.inputs[1])
        # Layout is descriptive; all rank-four spatial values in decoder graphs
        # are NCHW. Attention/normalization transposes keep ND.
        for value in values:
            if value.tensor.name in convolution_values:
                value.tensor.layout = 0  # DataLayout::NCHW
        write_feather_model(Path(path), name, name, self.inputs, self.outputs, values, nodes, payloads, self.metadata)
