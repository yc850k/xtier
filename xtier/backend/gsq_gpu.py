"""The GSQ (IQ-quantized GGUF) experts on the GPU: the IQ blocks decoded in Triton, for the hot expert cache.

The CPU engine reads the experts' IQ blocks where they are (the checkpoint file, xtier/backend/gsq.py) and decodes
them with llama.cpp's semantics (csrc/engine/gsq.cpp). The hot expert cache puts the hot experts' bytes in the VRAM
and computes them there, so the GPU needs the same decode: this module is its Triton port, reading the same bytes
and producing the same weights (the tables are the engine's, csrc/engine/gsq_tables.h).

A block is 256 values: the bytes of one row of the expert's tensor (the rows of H values for the gate and the up,
the rows of I values for the down). The decode of a block is a @triton.jit function over the [BN, 256] values of the
BN rows at a given byte offset, so the GEMM kernels can decode the block they need. The u16 fields are read through
a second view of the same bytes (B16): the offsets are even (the block sizes are), so the rows stay aligned.

The cache's GEMM kernels are here too (gate_up, down): they decode the blocks of the expert they need and multiply
them by the activations, like the FP4 pool's kernels (xtier/cache/pool.py) do with the packed FP4 weights. The
decode's arithmetic is FP32 (the block scales are FP16 and the index arithmetic must not round); the dot takes the
decoded values in FP16 (as the FP4 pool's does) with an FP32 accumulator.
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
    from the iq2xxs grid (the byte at 8*index + j) and its signs from the sign masks.

    The values are decoded in three dimensions (the row, the group of eight values, the value in the group): the
    group's codes and scales are the same for the eight values of a group, so they are loaded once for the group
    instead of once for every value (the loads are what the kernel spends its time on: seven per value against
    one and a half here).
    """
    g = tl.arange(0, 32)                    # the groups of eight values (the four lanes of the eight groups of 32)
    ib = (g // 4).to(tl.int64)              # the group of 32 values
    ln = (g % 4).to(tl.int64)               # the lane of 8 values in the group
    j = tl.arange(0, 8)                     # the value in the lane
    # the group's u32s: aux32[0] at +2, aux32[1] at +6 (u16 loads: the addresses are even)
    i0 = BASE[:, None] // 2 + 1 + 4 * ib[None, :]
    a0 = tl.load(B16 + i0).to(tl.int32) | (tl.load(B16 + i0 + 1).to(tl.int32) << 16)
    a1 = tl.load(B16 + i0 + 2).to(tl.int32) | (tl.load(B16 + i0 + 3).to(tl.int32) << 16)
    # (the u16 loads come from a uint16 view: they are zero-extended)
    d = tl.load(B16 + BASE[:, None] // 2).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    db = d * (0.5 + (a1.to(tl.uint32) >> 28).to(tl.float32)) * 0.25
    # the group's four grid indices: the bytes of aux32[0]
    idx = (a0 >> (8 * ln[None, :])) & 0xFF
    mag = tl.load(G3 + idx.to(tl.int64)[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
    signs = tl.load(KS + ((a1 >> (7 * ln[None, :])) & 127)).to(tl.int32)
    sign = tl.where(((signs[:, :, None] >> j[None, None, :]) & 1) != 0, -1.0, 1.0)
    return tl.reshape(db[:, :, None] * mag * sign, (BN, 256))


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


@triton.jit
def _block_bytes(TYPE):
    """The bytes of one 256-value block of a type (csrc/engine/gsq.h: the IQ2_XXS, IQ2_XS, IQ1_S and IQ1_M blocks)."""
    return tl.where(TYPE == 16, 66, tl.where(TYPE == 17, 74, tl.where(TYPE == 19, 50, 56)))


@triton.jit
def gate_up(X, SLOT_LIST, SLOT_EXP, ARENA, ARENA16, TYPES, OFFS, MID, GATE, M, H: tl.constexpr, I: tl.constexpr,
            LIMIT, STRIDE, G1, G2, G3, KS, BM: tl.constexpr, BN: tl.constexpr,
            ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0, DT: tl.constexpr = tl.float16,
            HOT: tl.constexpr = 0, ARENA2=None, ARENA16_2=None, ROLE: tl.constexpr = 0,
            IOFF: tl.constexpr = 0, IP: tl.constexpr = 0):
    """The gate (ROLE 0) or the up (ROLE 1) of a cached expert: its IQ blocks decoded and multiplied by the tokens.

    Program (route, tile): the rows n of the expert's gate or up (I rows of H values, a run of 256-value blocks a
    row) for the tokens of the call. The two roles are **two launches**: the decoded tiles of one role are
    [BN, 256] FP32 (16 KB a tile), and both roles live at once do not fit in the registers (the kernel spilled
    522 of them and ran six times slower). The gate's launch writes its FP32 values to GATE and the up's launch
    reads them, so the intermediate is the one of a single launch: SiLU(gate) * up in FP32.

    ARENA/ARENA16: the slots' bytes (the uint8 and the uint16 view of them). TYPES: the type of every expert per role
    (the gate, the up, the down), OFFS: where an expert's three tensors start in its slot, STRIDE: the bytes of a
    slot. The blocks are decoded in FP32 (the scales are FP16 and the index arithmetic must not round) and the dot
    takes them in FP16 (the tensor cores of the card), with an FP32 accumulator.

    The tensor-parallel split: the rank computes the rows IOFF..IOFF+IP of the intermediate (IP its own width, the
    expert's whole intermediate is I). The rows of the gate and the up are the intermediate's, so the rank's rows
    are the ones at IOFF; the intermediate it keeps (MID, GATE) is IP wide. The ranks' partial outputs are added by
    SGLang, so each rank computes its own slice of every expert.
    """
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    # the hot experts live in the pool's arena, the ones uploaded for this call in the scratch arena
    arena = ARENA
    arena16 = ARENA16
    row_off = slot.to(tl.int64)
    if HOT > 0 and slot >= HOT:
        arena = ARENA2
        arena16 = ARENA16_2
        row_off = row_off - HOT
    base = row_off * STRIDE
    ty = tl.load(TYPES + e * 3 + ROLE)
    o = tl.load(OFFS + e * 3 + ROLE).to(tl.int64)
    b = _block_bytes(ty).to(tl.int64)
    r = (H // 256) * b     # the bytes of a row
    tok = tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)      # the rank's own rows of the intermediate
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(H // 256):
        ke = kb * 256 + tl.arange(0, 256)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0).to(tl.float16)
        w = _iq_vals(ty, arena, arena16, base + o + n.to(tl.int64) * r + kb * b, G1, G2, G3, KS, BN).to(tl.float16)
        acc = tl.dot(xe, tl.trans(w), acc)
    if ROLE == 0:
        # the gate: its values (FP32) for the up's launch
        tl.store(GATE + (row * BM + tok[:, None]) * IP + n[None, :], acc, mask=tok[:, None] < M)
    else:
        g = tl.load(GATE + (row * BM + tok[:, None]) * IP + n[None, :], mask=tok[:, None] < M, other=0.0)
        if LIMIT > 0:
            g = tl.minimum(g, LIMIT)
            acc = tl.minimum(tl.maximum(acc, -LIMIT), LIMIT)
        if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
            val = g / (1.0 + tl.exp(-ALPHA * g)) * (acc + BETA)
        else:
            val = g / (1.0 + tl.exp(-g)) * acc
        tl.store(MID + (row * BM + tok[:, None]) * IP + n[None, :], val.to(DT), mask=tok[:, None] < M)


@triton.jit
def down(MID, SLOT_LIST, SLOT_EXP, ARENA, ARENA16, TYPES, OFFS, PART, M, H: tl.constexpr, I: tl.constexpr,
         STRIDE, G1, G2, G3, KS, BM: tl.constexpr, BN: tl.constexpr, DT: tl.constexpr = tl.float16,
         HOT: tl.constexpr = 0, ARENA2=None, ARENA16_2=None, IOFF: tl.constexpr = 0, IP: tl.constexpr = 0):
    """The down of a cached expert: its IQ blocks decoded and multiplied by the intermediate.

    Program (route, tile): the rows h of the expert's down (H rows of I values, a run of 256-value blocks a row) for
    the tokens of the call. The partial output goes to PART[route, token, h] (FP32).

    The tensor-parallel split: the rank owns the columns IOFF..IOFF+IP of the intermediate, so it decodes the
    blocks of the down that overlap them (the block that straddles the boundary is decoded whole and its columns
    outside the rank's range are dropped: the intermediate's are zero, so they add nothing to the dot). The
    intermediate the rank keeps (MID) is IP wide.
    """
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    arena = ARENA
    arena16 = ARENA16
    row_off = slot.to(tl.int64)
    if HOT > 0 and slot >= HOT:
        arena = ARENA2
        arena16 = ARENA16_2
        row_off = row_off - HOT
    base = row_off * STRIDE
    td = tl.load(TYPES + e * 3 + 2)
    od = tl.load(OFFS + e * 3 + 2).to(tl.int64)
    bd = _block_bytes(td).to(tl.int64)
    rd = (I // 256) * bd     # the bytes of a down row
    tok = tl.arange(0, BM)
    h = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # the blocks of the down that hold the rank's columns of the intermediate
    for kb in range(IOFF // 256, (IOFF + IP - 1) // 256 + 1):
        ke = kb * 256 + tl.arange(0, 256) - IOFF    # the rank's own columns of the block
        km = (ke >= 0) & (ke < IP)
        ae = tl.load(MID + (row * BM + tok[:, None]) * IP + ke[None, :],
                     mask=km[None, :] & (tok[:, None] < M), other=0.0).to(tl.float16)
        w = _iq_vals(td, arena, arena16, base + od + h.to(tl.int64) * rd + kb * bd, G1, G2, G3, KS, BN).to(tl.float16)
        acc = tl.dot(ae, tl.trans(w), acc)
    tl.store(PART + (row * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


@triton.jit
def gate_up_tiles(X, ORDER, TILE_E, TILE_S, TILE_START, TILE_END, ARENA, ARENA16, TYPES, OFFS, MID, GATE,
                  K: tl.constexpr, H: tl.constexpr, I: tl.constexpr, LIMIT, STRIDE, G1, G2, G3, KS,
                  BM: tl.constexpr, BN: tl.constexpr, ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0,
                  DT: tl.constexpr = tl.float16, ROLE: tl.constexpr = 0, IOFF: tl.constexpr = 0,
                  IP: tl.constexpr = 0):
    """The gate (ROLE 0) or the up (ROLE 1) of a cached expert for the routes of one tile (a batched prefill).

    Program (tile, row tile): the rows n of the expert's gate or up (I rows of H values, a run of 256-value blocks a
    row) for the routes of the tile. The two roles are **two launches** (the decoded tiles of both roles do not fit
    in the registers at once: the kernel spills hundreds of them and runs several times slower). The gate's launch
    writes its FP32 values to GATE and the up's launch reads them, so the intermediate is the one of a single launch:
    SiLU(gate) * up in FP32.
    """
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
    slot = tl.load(TILE_S + tile).to(tl.int64)
    base = slot * STRIDE
    arena = ARENA
    arena16 = ARENA16
    ty = tl.load(TYPES + e * 3 + ROLE)
    o = tl.load(OFFS + e * 3 + ROLE).to(tl.int64)
    b = _block_bytes(ty).to(tl.int64)
    r = (H // 256) * b     # the bytes of a row
    rows = tl.load(TILE_START + tile) + tl.arange(0, BM)
    rm = rows < tl.load(TILE_END + tile)
    tok = (tl.load(ORDER + rows, rm, 0) // K).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)      # the rank's own rows of the intermediate
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(H // 256):
        ke = kb * 256 + tl.arange(0, 256)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=rm[:, None], other=0.0).to(tl.float16)
        w = _iq_vals(ty, arena, arena16, base + o + n.to(tl.int64) * r + kb * b, G1, G2, G3, KS, BN).to(tl.float16)
        acc = tl.dot(xe, tl.trans(w), acc)
    if ROLE == 0:
        # the gate: its values (FP32) for the up's launch
        tl.store(GATE + rows[:, None].to(tl.int64) * IP + n[None, :], acc, mask=rm[:, None])
    else:
        g = tl.load(GATE + rows[:, None].to(tl.int64) * IP + n[None, :], mask=rm[:, None], other=0.0)
        if LIMIT > 0:
            g = tl.minimum(g, LIMIT)
            acc = tl.minimum(tl.maximum(acc, -LIMIT), LIMIT)
        if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
            val = g / (1.0 + tl.exp(-ALPHA * g)) * (acc + BETA)
        else:
            val = g / (1.0 + tl.exp(-g)) * acc
        tl.store(MID + rows[:, None].to(tl.int64) * IP + n[None, :], val.to(DT), mask=rm[:, None])


@triton.jit
def down_tiles(MID, ORDER, TILE_E, TILE_S, TILE_START, TILE_END, ARENA, ARENA16, TYPES, OFFS, ROUTES,
               H: tl.constexpr, I: tl.constexpr, STRIDE, G1, G2, G3, KS, BM: tl.constexpr, BN: tl.constexpr,
               DT: tl.constexpr = tl.float16, IOFF: tl.constexpr = 0, IP: tl.constexpr = 0):
    """The down of a cached expert for the routes of one tile (a batched prefill).

    Program (tile, row tile): the rows h of the expert's down (H rows of I values, a run of 256-value blocks a row)
    for the routes of the tile. The partial output goes to ROUTES[route, h] (FP32), the per-token sum of the routes
    is the caller's.

    The tensor-parallel split: the rank owns the columns IOFF..IOFF+IP of the intermediate (see the down kernel).
    """
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)
    slot = tl.load(TILE_S + tile).to(tl.int64)
    base = slot * STRIDE
    arena = ARENA
    arena16 = ARENA16
    td = tl.load(TYPES + e * 3 + 2)
    od = tl.load(OFFS + e * 3 + 2).to(tl.int64)
    bd = _block_bytes(td).to(tl.int64)
    rd = (I // 256) * bd     # the bytes of a down row
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    rm = r < tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, rm, 0).to(tl.int64)
    h = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # the blocks of the down that hold the rank's columns of the intermediate
    for kb in range(IOFF // 256, (IOFF + IP - 1) // 256 + 1):
        ke = kb * 256 + tl.arange(0, 256) - IOFF    # the rank's own columns of the block
        km = (ke >= 0) & (ke < IP)
        ae = tl.load(MID + r[:, None].to(tl.int64) * IP + ke[None, :],
                     mask=km[None, :] & rm[:, None], other=0.0).to(tl.float16)
        w = _iq_vals(td, arena, arena16, base + od + h.to(tl.int64) * rd + kb * bd, G1, G2, G3, KS, BN).to(tl.float16)
        acc = tl.dot(ae, tl.trans(w), acc)
    tl.store(ROUTES + route[:, None] * H + h[None, :], acc, mask=rm[:, None])
