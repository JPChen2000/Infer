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

import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from flux2_graph import Graph, TensorStore, FP32, BF16


class Flux2GraphTest(unittest.TestCase):
    def test_lazy_sharded_weights_preserve_shapes_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({'proj.weight': np.arange(12, dtype=np.float32).reshape(3, 4)}, str(root / 'a.safetensors'))
            store = TensorStore(root)
            graph = Graph(store, BF16)
            x = graph.input('x', [1, 2, 4], BF16)
            y = graph.linear(x, 'proj.weight')
            self.assertEqual(graph.shape(y), [1, 2, 3])
            self.assertEqual(graph.dtype(y), BF16)
            weight = next(n.inputs[1] for n in graph.nodes if n.op_type == 'MatMul')
            self.assertEqual(graph.shape(weight), [4, 3])
            np.testing.assert_array_equal(graph.external_constant_loaders[weight]().astype(np.float32),
                                          np.arange(12, dtype=np.float32).reshape(3, 4).T)

    def test_incompatible_linear_and_broadcast_fail_before_export(self):
        graph = Graph(TensorStore({'w': np.zeros((3, 4), np.float32)}), FP32)
        x = graph.input('x', [1, 2, 5], FP32)
        with self.assertRaisesRegex(ValueError, 'linear'):
            graph.linear(x, 'w')
        y = graph.input('y', [1, 3, 5], FP32)
        with self.assertRaises(ValueError):
            graph.binary_op('Add', x, y)

    def test_duplicate_safetensors_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('a', 'b'):
                save_file({'w': np.ones((2,), np.float32)}, str(Path(directory) / (name + '.safetensors')))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                TensorStore(directory)

    def test_affine_free_norm_uses_broadcast_scalar(self):
        graph = Graph(TensorStore({}), FP32)
        x = graph.input('x', [1, 32, 65536], FP32)
        graph.normalize(x)
        norm = graph.nodes[-1]
        self.assertEqual(graph.shape(norm.inputs[1]), [1])


if __name__ == '__main__':
    unittest.main()
