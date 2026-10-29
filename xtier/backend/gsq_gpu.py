"""The GSQ (IQ-quantized GGUF) experts on the GPU: the IQ blocks decoded in Triton, for the hot expert cache.

The CPU engine reads the experts' IQ blocks where they are (the checkpoint file, xtier/backend/gsq.py) and decodes
them with llama.cpp's semantics (csrc/engine/gsq.cpp). The hot expert cache puts the hot experts' bytes in the VRAM
and computes them there, so the GPU needs the same decode: this module is its Triton port, reading the same bytes
and producing the same weights (the tables are the engine's, csrc/engine/gsq_tables.h).

A block is 256 values: the bytes of one row of the expert's tensor (the rows of H values for the gate and the up,
the rows of I values for the down). The decode of a block is a @triton.jit function over the [BN, 256] values of the
BN rows at a given byte offset, so the GEMM kernels can decode the block they need. The u16 fields are read through
a second view of the same bytes (B16): the offsets are even (the block sizes are), so the rows stay aligned.
"""

import re
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

# The engine's tables (csrc/engine/gsq_tables.h): the grids of the magnitudes and the sign masks.
_TABLE_NAMES = ('kmask_iq2xs', 'ksigns_iq2xs', 'iq2xxs_grid', 'iq2xs_grid', 'iq1s_grid')
_tables = None


def tables(device):
    """The IQ decode's tables on the device: iq1s (int8) and iq2xs / iq2xxs (the bytes of the uint64 grids) and the
    sign masks (uint8)."""
    global _tables
    if _tables is None:
        text = (Path(__file__).resolve().parents[2] / 'csrc' / 'engine' / 'gsq_tables.h').read_text()
        out = {}
        for name in _TABLE_NAMES:
            m = re.search(r'inline constexpr \w+ ' + name + r'\[\d+\] = \{(.*?)\};', text, re.S)
            if m is None:
                raise RuntimeError(f'xtier: the engine tables have no {name} (csrc/engine/gsq_tables.h)')
            values = [int(v.strip(), 0) for v in m.group(1).replace('\n', ' ').split(',') if v.strip()]
            out[name] = np.array(values, dtype=np.uint64 if 'grid' in name else np.uint8)
        _tables = dict(
            iq1s=torch.from_numpy(out['iq1s_grid'].view(np.uint8)).view(torch.int8).to(device),
            iq2xs=torch.from_numpy(out['iq2xs_grid'].view(np.uint8)).to(device),
            iq2xxs=torch.from_numpy(out['iq2xxs_grid'].view(np.uint8)).to(device),
            ksigns=torch.from_numpy(out['ksigns_iq2xs']).to(device),
        )
    return _tables


@triton.jit
def _base(ROW0, ROW_STRIDE, BLK, BLOCK_BYTES, BN: tl.constexpr):
    """The byte offset of every row's block: the rows are ROW_STRIDE bytes apart, the block BLK of the row at
    BLOCK_BYTES bytes each."""
    row = tl.arange(0, BN).to(tl.int64)
    return ROW0 + row * ROW_STRIDE + BLK * BLOCK_BYTES


@triton.jit
def _iq2_xxs(B8, B16, BASE, G3, KS, BN: tl.constexpr):
    """IQ2_XXS (66 bytes a block): the fp16 d, then 8 groups of 32 values (two u32 each). A group's magnitudes come
    from the iq2xxs grid (the byte at 8*index + j) and its signs from the sign masks."""
    v = tl.arange(0, 256)
    ib = (v // 32).to(tl.int64)          # the group of 32 values
    ln = ((v % 32) // 8).to(tl.int64)    # the lane of 8 values in the group
    j = (v % 8).to(tl.int64)             # the value in the lane
    # the group's u32s: aux32[0] at +2, aux32[1] at +6 (u16 loads: the addresses are even)
    i0 = BASE[:, None] // 2 + 1 + 4 * ib[None, :]
    a0 = tl.load(B16 + i0).to(tl.int32) | (tl.load(B16 + i0 + 1).to(tl.int32) << 16)
    a1 = tl.load(B16 + i0 + 2).to(tl.int32) | (tl.load(B16 + i0 + 3).to(tl.int32) << 16)
    # (the u16 loads come from a uint16 view: they are zero-extended)
    d = tl.load(B16 + BASE[:, None] // 2).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    db = d * (0.5 + (a1.to(tl.uint32) >> 28).to(tl.float32)) * 0.25
    # the group's four grid indices: the bytes of aux32[0]
    idx = (a0 >> (8 * ln[None, :])) & 0xFF
    mag = tl.load(G3 + idx.to(tl.int64) * 8 + j[None, :]).to(tl.float32)
    signs = tl.load(KS + ((a1 >> (7 * ln[None, :])) & 127)).to(tl.int32)
    sign = tl.where(((signs >> j[None, :]) & 1) != 0, -1.0, 1.0)
    return db * mag * sign


@triton.jit
def _iq2_xs(B8, B16, BASE, G2, KS, BN: tl.constexpr):
    """IQ2_XS (74 bytes a block): the fp16 d, 8 groups of 32 values (four u16 codes each) and 8 scale bytes."""
    v = tl.arange(0, 256)
    ib = (v // 32).to(tl.int64)
    ln = ((v % 32) // 8).to(tl.int64)
    j = (v % 8).to(tl.int64)
    d = tl.load(B16 + BASE[:, None] // 2).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    sb = tl.load(B8 + BASE[:, None] + 66 + ib[None, :]).to(tl.int32)
    db0 = d * (0.5 + (sb & 0xF).to(tl.float32)) * 0.25
    db1 = d * (0.5 + (sb >> 4).to(tl.float32)) * 0.25
    db = tl.where(ln[None, :] >= 2, db1, db0)
    code = tl.load(B16 + BASE[:, None] // 2 + 1 + 4 * ib[None, :] + ln[None, :]).to(tl.int32)
    mag = tl.load(G2 + (code & 511).to(tl.int64) * 8 + j[None, :]).to(tl.float32)
    signs = tl.load(KS + (code >> 9)).to(tl.int32)
    sign = tl.where(((signs >> j[None, :]) & 1) != 0, -1.0, 1.0)
    return db * mag * sign


@triton.jit
def _iq1_s(B8, B16, BASE, G1, BN: tl.constexpr):
    """IQ1_S (50 bytes a block): the fp16 d, 32 code bytes and 16 high bits (u16); the magnitudes come from the
    iq1s grid and the sign is the delta (+-0.125) of the group's high bits."""
    v = tl.arange(0, 256)
    ib = (v // 32).to(tl.int64)
    ln = ((v % 32) // 8).to(tl.int64)
    j = (v % 8).to(tl.int64)
    d = tl.load(B16 + BASE[:, None] // 2).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    qh = tl.load(B16 + BASE[:, None] // 2 + 17 + ib[None, :]).to(tl.int32)
    dl = d * (2 * ((qh >> 12) & 7) + 1).to(tl.float32)
    delta = tl.where((qh & 0x8000) != 0, -0.125, 0.125)
    qs = tl.load(B8 + BASE[:, None] + 2 + 4 * ib[None, :] + ln[None, :]).to(tl.int32)
    idx = qs | (((qh >> (3 * ln[None, :])) & 7) << 8)
    mag = tl.load(G1 + idx.to(tl.int64) * 8 + j[None, :]).to(tl.float32)
    return dl * (mag + delta)


@triton.jit
def _iq1_m(B8, B16, BASE, G1, BN: tl.constexpr):
    """IQ1_M (56 bytes a block): 32 code bytes, 16 high bits (bytes) and the 4 scale u16s (the block's d is assembled
    from their high nibbles, the groups' scales from their low ones)."""
    v = tl.arange(0, 256)
    ib = (v // 32).to(tl.int64)
    ln = ((v % 32) // 8).to(tl.int64)
    j = (v % 8).to(tl.int64)
    s = BASE[:, None] // 2 + 24
    s0 = tl.load(B16 + s).to(tl.int32)
    s1 = tl.load(B16 + s + 1).to(tl.int32)
    s2 = tl.load(B16 + s + 2).to(tl.int32)
    s3 = tl.load(B16 + s + 3).to(tl.int32)
    bits = ((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000)).to(tl.uint16)
    d = bits.to(tl.float16, bitcast=True).to(tl.float32)
    sc = tl.load(B16 + BASE[:, None] // 2 + 24 + (ib // 2)[None, :]).to(tl.int32)
    dl1 = d * (2 * ((sc >> (6 * (ib % 2)[None, :] + 0)) & 7) + 1).to(tl.float32)
    dl2 = d * (2 * ((sc >> (6 * (ib % 2)[None, :] + 3)) & 7) + 1).to(tl.float32)
    # the high bits of the group: two bytes
    qh0 = tl.load(B8 + BASE[:, None] + 32 + 2 * ib[None, :]).to(tl.int32)
    qh1 = tl.load(B8 + BASE[:, None] + 33 + 2 * ib[None, :]).to(tl.int32)
    qs = tl.load(B8 + BASE[:, None] + 4 * ib[None, :] + ln[None, :]).to(tl.int32)
    sel = tl.where(ln[None, :] < 2, qh0, qh1)
    shift = tl.where(ln[None, :] % 2 == 0, 8, 4)
    idx = qs | ((sel << shift) & 0x700)
    mag = tl.load(G1 + idx.to(tl.int64) * 8 + j[None, :]).to(tl.float32)
    bit = tl.where(ln[None, :] % 2 == 0, 0x08, 0x80)
    delta = tl.where((sel & bit) != 0, -0.125, 0.125)
    dl = tl.where(ln[None, :] < 2, dl1, dl2)
    return dl * (mag + delta)


@triton.jit
def _iq_vals(TYPE, B8, B16, BASE, G1, G2, G3, KS, BN: tl.constexpr):
    """The [BN, 256] FP32 values of the block at each row's BASE: the engine's IQ decode, per type."""
    if TYPE == 16:
        vals = _iq2_xxs(B8, B16, BASE, G3, KS, BN)
    elif TYPE == 17:
        vals = _iq2_xs(B8, B16, BASE, G2, KS, BN)
    elif TYPE == 19:
        vals = _iq1_s(B8, B16, BASE, G1, BN)
    else:
        vals = _iq1_m(B8, B16, BASE, G1, BN)
    return vals


@triton.jit
def decode_rows(TYPE, B8, B16, ROW0, ROW_STRIDE, BLK, BLOCK_BYTES, OUT, G1, G2, G3, KS, BN: tl.constexpr):
    """Decode the block BLK of every row (ROW_STRIDE bytes apart, the first at ROW0) into OUT[row, 0:256] (FP32)."""
    base = _base(ROW0, ROW_STRIDE, BLK, BLOCK_BYTES, BN)
    vals = _iq_vals(TYPE, B8, B16, base, G1, G2, G3, KS, BN)
    row = tl.arange(0, BN)
    tl.store(OUT + row[:, None] * 256 + tl.arange(0, 256)[None, :], vals)
