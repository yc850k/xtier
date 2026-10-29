"""The GSQ checkpoints: the Maya pack of a GGUF DeepSeek V4.1 (the routed experts in llama.cpp's IQ quants).

The pack directory holds the geometry (``ds4.txt``), the tensor table (``tensors.txt``: every tensor's name, dtype,
file, offset, bytes and shape) and the tokenizer. The expert tensors live in the GGUF files themselves: the engine
reads the rows where they are (its IQ decode), so no weights are copied and the host memory holds only the page
cache of the files the experts are read from.
"""

import ctypes
import mmap
import os
from pathlib import Path

import numpy as np

# the pack's dtype names -> the ggml type ids (the engine's IQ decoders)
TYPES = {
    'IQ2_XXS': 16, 'IQ2_XS': 17, 'IQ1_S': 19, 'IQ1_M': 29,
    'IQ2_S': 22, 'IQ3_XXS': 18, 'IQ3_S': 21, 'IQ4_XS': 23,
    'Q2_K': 10, 'Q3_K': 11, 'Q4_K': 12, 'Q5_K': 13, 'Q6_K': 14,
}


class GsqPack:
    """One Maya pack: the geometry, the tensor table and the mapped GGUF files."""

    def __init__(self, path):
        self.path = Path(path)
        self.geometry = self._geometry()
        self.tensors, self.files = self._tensors()
        self._mapped = {}
        self.experts = int(self.geometry.get('n_routed_experts', 0))
        self.hidden = int(self.geometry.get('hidden_size', 0))
        self.inter = int(self.geometry.get('moe_intermediate_size', 0))
        self.layers = int(self.geometry.get('num_hidden_layers', 0))
        if not self.experts or not self.hidden or not self.inter:
            raise ValueError(f'xtier: {self.path}/ds4.txt has no geometry (n_routed_experts, hidden_size, '
                             f'moe_intermediate_size)')

    def _geometry(self):
        """ds4.txt: the model's geometry under the checkpoint's names."""
        out = {}
        for line in (self.path / 'ds4.txt').read_text().splitlines():
            if ' ' not in line:
                continue
            key, _, value = line.partition(' ')
            out[key] = value
        return out

    def _tensors(self):
        """tensors.txt: the tensors (name -> dtype, file, offset, bytes, dims) and the files (index -> name)."""
        tensors, files = {}, {}
        for line in (self.path / 'tensors.txt').read_text().splitlines():
            if line.startswith('file '):
                parts = line.split()
                files[int(parts[1])] = parts[2]
            elif line.startswith('t '):
                parts = line.split()
                tensors[parts[1]] = (parts[2], int(parts[3]), int(parts[4]), int(parts[5]),
                                      [int(d) for d in parts[6:]])
        return tensors, files

    def _file(self, index):
        """The mapped GGUF file with this index."""
        if index in self._mapped:
            return self._mapped[index]
        name = self.files.get(index)
        if name is None:
            raise ValueError(f'xtier: the pack has no file {index}')
        root = Path(self.geometry.get('model_dir', self.path.parent))
        path = Path(name) if os.path.isabs(name) else root / name
        handle = open(path, 'rb')
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._mapped[index] = (handle, mapped)
        return self._mapped[index]

    def _address(self, index, offset):
        """The address of a byte range of a mapped file."""
        _, mapped = self._file(index)
        base = np.frombuffer(mapped, dtype=np.uint8).ctypes.data
        return base + offset

    def expert(self, role, layer, expert):
        """One expert's rows of a role ('w1' the gate, 'w3' the up, 'w2' the down): the address and the type id."""
        name = f'layers.{layer}.ffn.experts.{expert}.{role}.weight'
        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f'xtier: the pack has no {name}')
        dtype, file, offset, size, dims = entry
        if dtype not in TYPES:
            raise ValueError(f'xtier: {name} is {dtype}, which the engine cannot read')
        return self._address(file, offset), TYPES[dtype], dims

    def layer_sources(self, layer):
        """A layer's experts as the engine takes them: the gate, the up and the down (the base addresses of the
        experts' tensors) and the type of every expert."""
        gate, gate_type, gate_dims = self.expert('w1', layer, 0)
        up, _, _ = self.expert('w3', layer, 0)
        down, _, _ = self.expert('w2', layer, 0)
        types = bytearray()
        for expert in range(self.experts):
            types.append(self.expert('w1', layer, expert)[1])
        if gate_dims and (gate_dims[-1] != self.hidden or gate_dims[-2] != self.inter):
            raise ValueError(f'xtier: the gate of layer {layer} is {gate_dims}, not [experts, {self.inter}, '
                             f'{self.hidden}]')
        return gate, up, down, bytes(types)

    def close(self):
        for handle, mapped in self._mapped.values():
            mapped.close()
            handle.close()
        self._mapped.clear()


def pack_path(model_path):
    """The pack of a checkpoint: XTIER_GSQ_PACK, else <model>/maya-ds4-gsq."""
    env = os.environ.get('XTIER_GSQ_PACK', '').strip()
    if env:
        return Path(env)
    return Path(model_path) / 'maya-ds4-gsq'
