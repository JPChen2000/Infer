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

"""Mandatory native tiny Qwen3 -> FLUX.2 -> VAE pipeline acceptance.

These deterministic synthetic networks exercise real architecture operators;
PNG checks establish execution, not real-checkpoint visual quality.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

from export_flux2_model import build_denoiser, component_metadata
from flux2_export_test import tiny_denoiser_fixture
from flux2_graph import BF16, FP32, TensorStore
from flux2_text_export import build_text_encoder
from flux2_text_export_test import fixture as text_fixture
from flux2_vae_export import build_vae_decoder
from flux2_vae_export_test import tiny_fixture as vae_fixture


class Flux2PipelineTest(unittest.TestCase):
    def test_native_full_pipeline_png_determinism_and_failures(self):
        repo = Path(__file__).resolve().parents[1]
        demo = Path(os.environ.get('FEATHER_FLUX2_DEMO', repo / 'build_new/bin/flux2_demo'))
        probe = Path(os.environ.get('FLUX2_COMPONENT_PROBE', repo / 'build_new/bin/flux2_component_probe'))
        self.assertTrue(demo.is_file(), f'mandatory native demo missing: {demo}')
        self.assertTrue(probe.is_file(), f'mandatory native component probe missing: {probe}')
        length = 64  # Preserve the complete non-thinking template and either tiny prompt.
        config, weights = text_fixture()
        denoiser_config, denoiser_store, _ = tiny_denoiser_fixture()
        _, _, vae_store = vae_fixture()
        for dtype in (FP32, BF16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                graphs = {
                    'text_encoder': build_text_encoder(TensorStore(weights), config, length,
                                                       dtype=dtype, hidden_layers=(1, 2, 3)),
                    'denoiser': build_denoiser(denoiser_store, 16, 32, length,
                                               dtype=dtype, config=denoiser_config),
                    'vae_decoder': build_vae_decoder(vae_store, 16, 32, dtype=dtype, ch=4,
                                                     ch_mult=(1, 2, 2, 2), num_res_blocks=1,
                                                     z_channels=2, groups=2),
                }
                for component, graph in graphs.items():
                    component_metadata(graph, component, 16, 32, length, 24, 8)
                    if component == 'text_encoder':
                        graph.metadata.update({'flux2.pad_token_id': 0, 'flux2.padding_side': 'right'})
                    graph.write(root / (component + '.fth'), 'tiny_full_' + component)
                # All literal bytes in the chat wrapper plus a/b are represented.
                # The single BPE merge yields exactly the fixture's vocabulary size.
                specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
                vocab = {token: i for i, token in enumerate(specials)}
                for token in sorted(set('userassistant<think></think>b')) + ['Ċ', 'ab']:
                    vocab[token] = len(vocab)
                self.assertEqual(len(vocab), 19)
                (root / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
                (root / 'merges.txt').write_text('#version: 0.2\na b\n', encoding='utf-8')
                (root / 'tokenizer.json').write_text(json.dumps({'added_tokens': [
                    {'id': vocab[token], 'content': token} for token in specials]}), encoding='utf-8')
                # Independently form the exact byte tokens; compare native Qwen3
                # embeddings for distinct prompts if PNG quantization hides changes.
                def ids_for(prompt):
                    parts = ['<|im_start|>', 'user\n' + prompt, '<|im_end|>', '\n',
                             '<|im_start|>', 'assistant\n<think>\n\n</think>\n\n']
                    ids = []
                    for part in parts:
                        if part in specials:
                            ids.append(vocab[part])
                        else:
                            ids.extend(vocab['Ċ' if c == '\n' else c] for c in part)
                    self.assertLess(len(ids), length)
                    padded = np.zeros((1, length), np.int64)
                    padded[0, :len(ids)] = ids
                    mask = np.full((1, 1, 1, length), -np.inf, np.float32)
                    mask[..., :len(ids)] = 0
                    return padded, mask

                def invoke(prompt, backend, output):
                    return subprocess.run([str(demo), '--model-dir', str(root), '--prompt', prompt,
                                           '--backend', backend, '--seed', '1729', '--steps', '4',
                                           '--output', str(output)], capture_output=True, text=True, timeout=120)

                for backend in ('common', 'cuda'):
                    if backend == 'cuda' and os.environ.get('FEATHER_TEST_CUDA', '1') == '0':
                        continue
                    with self.subTest(backend=backend):
                        files = [root / f'{backend}-{i}.png' for i in range(3)]
                        for prompt, output in zip(('a', 'a', 'b'), files):
                            run = invoke(prompt, backend, output)
                            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                            data = output.read_bytes()
                            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
                            self.assertEqual(struct.unpack('>II', data[16:24]), (32, 16))
                        self.assertEqual(files[0].read_bytes(), files[1].read_bytes(),
                                         'same seed/prompt must reproduce identical PNG bytes')
                        if files[0].read_bytes() == files[2].read_bytes():
                            embeddings = []
                            for prompt in ('a', 'b'):
                                ids, mask = ids_for(prompt)
                                ids.tofile(root / 'ids.bin')
                                mask.tofile(root / 'mask.bin')
                                out = root / f'{backend}-{prompt}-text'
                                run = subprocess.run([str(probe), str(root / 'text_encoder.fth'),
                                                      str(out), backend, f'input_ids={root / "ids.bin"}',
                                                      f'attention_mask={root / "mask.bin"}'],
                                                     capture_output=True, text=True, timeout=120)
                                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                                embeddings.append((out / 'prompt_embeds.bin').read_bytes())
                            self.assertNotEqual(*embeddings, 'changed prompt must change native embeddings')
                        sentinel = root / 'existing.png'
                        sentinel.write_bytes(b'keep existing output')
                        saved = root / 'vae_decoder.fth'
                        saved.rename(root / 'vae_decoder.saved')
                        try:
                            run = invoke('a', backend, sentinel)
                            self.assertNotEqual(run.returncode, 0)
                            self.assertEqual(sentinel.read_bytes(), b'keep existing output')
                            missing = root / 'should-not-exist.png'
                            run = invoke('a', backend, missing)
                            self.assertNotEqual(run.returncode, 0)
                            self.assertFalse(missing.exists())
                        finally:
                            (root / 'vae_decoder.saved').rename(saved)


if __name__ == '__main__':
    unittest.main()
