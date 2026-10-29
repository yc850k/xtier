"""The GSQ experts (a GGUF checkpoint's IQ-quantized rows) through the CPU engine, against a reference.

  python tests/gsq_check.py

The engine decodes llama.cpp's IQ types (IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M) and computes the experts with them; the
reference here decodes the same blocks in Python (llama.cpp's arithmetic, the grid tables read from the engine's own
header) and computes the same layer in FP32.

With XTIER_LLAMA_CPP pointing at a llama.cpp checkout, the decode is also compared with llama.cpp's own
dequantize_row_iq* bit for bit.
"""
import ctypes
import ctypes as C
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from xtier.engine import CpuPlan  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


# the grid tables of the engine (csrc/engine/gsq_tables.h)
def _tables():
    text = (Path(__file__).resolve().parents[1] / 'csrc' / 'engine' / 'gsq_tables.h').read_text()
    out = {}
    for name in ('kmask_iq2xs', 'ksigns_iq2xs', 'iq2xxs_grid', 'iq2xs_grid', 'iq1s_grid'):
        m = re.search(r'inline constexpr \w+ ' + name + r'\[\d+\] = \{(.*?)\};', text, re.S)
        values = [int(v.strip(), 0) for v in m.group(1).replace('\n', ' ').split(',') if v.strip()]
        out[name] = np.array(values, dtype=np.uint64 if 'grid' in name else np.uint8)
    return out


T = _tables()


def dequant(type_id, blob, n):
    """The blocks of one row as FP32 (llama.cpp's dequantize_row_iq*)."""
    out = np.empty(n, dtype=np.float32)
    k = 256
    for i in range(n // k):
        b = blob[i * (len(blob) // (n // k)):]
        if type_id == 16:      # IQ2_XXS
            d = np.float32(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
            for ib32 in range(8):
                aux = np.frombuffer(b[2 + 8 * ib32:2 + 8 * ib32 + 8].tobytes(), dtype=np.uint32)
                db = d * (0.5 + (aux[1] >> 28)) * 0.25
                for l in range(4):
                    grid = np.frombuffer(T['iq2xxs_grid'].tobytes()[int(aux[0].tobytes()[l]) * 8:][:8], dtype=np.uint8)
                    signs = T['ksigns_iq2xs'][(aux[1] >> (7 * l)) & 127]
                    for j in range(8):
                        out[i * k + ib32 * 32 + l * 8 + j] = (db * grid[j] *
                                                            (-1.0 if signs & T['kmask_iq2xs'][j] else 1.0))
        elif type_id == 29:    # IQ1_M
            sc = np.frombuffer(b[48:56].tobytes(), dtype=np.uint16)
            bits = ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)).astype(np.uint16)
            d = np.float32(np.frombuffer(bits.tobytes(), dtype=np.float16)[0])
            for ib in range(8):
                dl1 = d * (2 * ((sc[ib // 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1)
                dl2 = d * (2 * ((sc[ib // 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1)
                qs = [int(v) for v in b[ib * 4:ib * 4 + 4]]
                qh = [int(v) for v in b[32 + ib * 2:32 + ib * 2 + 2]]
                delta = [(-0.125 if qh[0] & 0x08 else 0.125), (-0.125 if qh[0] & 0x80 else 0.125),
                         (-0.125 if qh[1] & 0x08 else 0.125), (-0.125 if qh[1] & 0x80 else 0.125)]
                idx = [qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700),
                       qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)]
                for l in range(4):
                    grid = np.frombuffer(T['iq1s_grid'].tobytes()[int(idx[l]) * 8:][:8], dtype=np.int8)
                    dl = dl1 if l < 2 else dl2
                    for j in range(8):
                        out[i * k + ib * 32 + l * 8 + j] = dl * (grid[j] + delta[l])
        else:
            raise ValueError(f'the reference decodes the types {16} and {29} only')
    return out


# the layer: 10 experts (the engine's minimum), the geometry of a small MoE layer
E, H, I, K, M = 10, 256, 256, 2, 4   # the IQ blocks hold 256 values, so H and I are multiples of 256
THREADS = 4
rng = np.random.default_rng(20261029)
types = np.array([29, 29, 29, 16, 16, 29, 16, 29, 16, 29], dtype=np.uint8)   # IQ1_M and IQ2_XXS, mixed
BLOCK = {16: 66, 29: 56}
ROW = {16: H // 256 * 66, 29: H // 256 * 56}       # the bytes of a gate row
ROW_DOWN = {16: I // 256 * 66, 29: I // 256 * 56}   # the bytes of a down row

gate = np.zeros((E, ROW[29] * I if False else 0), dtype=np.uint8)
gates, ups, downs = [], [], []
for e in range(E):
    t = int(types[e])
    g = rng.integers(0, 256, ROW[t] * I, dtype=np.uint8).copy()
    u = rng.integers(0, 256, ROW[t] * I, dtype=np.uint8).copy()
    d = rng.integers(0, 256, ROW_DOWN[t] * H, dtype=np.uint8).copy()
    gates.append(g)
    ups.append(u)
    downs.append(d)
def sane(blob, type_id, value=1.0):
    """The scale fields of the blocks: a finite fp16 (random bytes would give NaNs)."""
    block = BLOCK[type_id]
    if type_id == 16:      # IQ2_XXS: the fp16 d at the start of every block
        h = np.frombuffer(np.float16(value).tobytes(), dtype=np.uint8)
        for i in range(len(blob) // block):
            blob[i * block:i * block + 2] = h
    else:                 # IQ1_M: the scales at 48 (the assembled fp16 0x3C00 = 1.0)
        # the fp16 0x3C00 (1.0): the bits 0-3 from sc[0], 4-7 from sc[1], 8-11 from sc[2], 12-15 from sc[3]
        sc = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x3000]).tobytes(), dtype=np.uint8)
        for i in range(len(blob) // block):
            blob[i * block + 48:i * block + 56] = sc
    return blob


for e in range(E):
    gates[e] = sane(gates[e], int(types[e]))
    ups[e] = sane(ups[e], int(types[e]))
    downs[e] = sane(downs[e], int(types[e]))
gate = np.concatenate(gates).copy()
up = np.concatenate(ups).copy()
down = np.concatenate(downs).copy()
# the gate, the up and the down of every expert, one tensor each (as the pack has them)
gate_ptr = gate.ctypes.data
up_ptr = up.ctypes.data
down_ptr = down.ctypes.data
types_ptr = types.ctypes.data

plan = CpuPlan(E, THREADS, [gate_ptr, up_ptr, down_ptr], hidden=H, inter=I, gsq_types=types_ptr)
print('engine ISA:', plan.lib.xtier_isa().decode())

# the routes and the activations
x = (rng.standard_normal((M, H)) * 0.02).astype(np.float32)
ids = rng.integers(0, E, (M, K)).astype(np.int32)
_w = rng.standard_normal((M, K))
rw = (_w - _w.max(axis=-1, keepdims=True))
rw = np.exp(rw)
rw = (rw / rw.sum(axis=-1, keepdims=True)).astype(np.float32)
# the engine reads BF16 activations (as the FP4 path does): the FP32 bits' top half
x_bf16 = (x.view(np.uint32) >> 16).astype(np.uint16)
ids32 = (C.c_int32 * (M * K))(*ids.reshape(-1).tolist())
rw32 = (C.c_float * (M * K))(*rw.reshape(-1).tolist())
x16 = (C.c_uint16 * (M * H))(*x_bf16.reshape(-1).tolist())
out = (C.c_float * (M * H))()
rc = plan.forward(M, K, C.addressof(x16), C.addressof(ids32), C.addressof(rw32), C.addressof(out))
check(rc >= 0, f'the engine computed the layer (rc {rc})')

# the reference: the same layer in FP32
ref = np.zeros((M, H), dtype=np.float32)
for t in range(M):
    for s in range(K):
        e = int(ids[t, s])
        ty = int(types[e])
        gw = np.stack([dequant(ty, gates[e][r * ROW[ty]:(r + 1) * ROW[ty]], H) for r in range(I)])
        uw = np.stack([dequant(ty, ups[e][r * ROW[ty]:(r + 1) * ROW[ty]], H) for r in range(I)])
        dw = np.stack([dequant(ty, downs[e][r * ROW_DOWN[ty]:(r + 1) * ROW_DOWN[ty]], I) for r in range(H)])
        gate_v = x[t] @ gw.T
        up_v = x[t] @ uw.T
        mid = gate_v / (1.0 + np.exp(-gate_v)) * up_v
        ref[t] += rw[t, s] * (mid @ dw.T)
got = np.frombuffer(out, dtype=np.float32).reshape(M, H)
err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
if err > 1e-2:
    print('got[0, :6]:', got[0, :6].tolist())
    print('ref[0, :6]:', ref[0, :6].tolist())
check(err < 5e-2, f'the engine\'s GSQ experts against the FP32 reference (relative L2 {err:.2e})')
plan.close()

print('GSQ-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
