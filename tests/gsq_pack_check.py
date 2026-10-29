"""The GSQ pack reader (xtier/backend/gsq.py): the geometry, the tensor table and the mapped GGUF files, and the
engine's plan on what it hands over.

  python tests/gsq_pack_check.py
"""
import ctypes
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from xtier.backend import gsq  # noqa: E402
from xtier.engine import CpuPlan  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


E, H, I = 10, 256, 256
BLOCK = {16: 66, 29: 56}
ROW = {16: H // 256 * 66, 29: H // 256 * 56}
ROW_DOWN = {16: I // 256 * 66, 29: I // 256 * 56}
types = [29, 29, 29, 16, 16, 29, 16, 29, 16, 29]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / 'model'
    pack = model / 'maya-ds4-gsq'
    pack.mkdir(parents=True)
    # the GGUF file: the bytes of the experts (the reader only maps it)
    size = sum(ROW[t] * I for t in types) + sum(ROW_DOWN[t] * H for t in types)
    raw = bytearray(np.random.default_rng(7).integers(0, 256, size, dtype=np.uint8).tobytes())
    # the scale fields of the blocks: a finite fp16 (random bytes would give NaNs)
    d16 = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    iq1m = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x3000]).tobytes(), dtype=np.uint8)
    at = 0
    for e in range(E):
        t = types[e]
        for role in ('w1', 'w3', 'w2'):
            rows = I if role != 'w2' else H
            block = BLOCK[t]
            for i in range(rows * (H if role != 'w2' else I) // 256):
                off = at + i * block
                if t == 16:
                    raw[off:off + 2] = d16.tobytes()
                else:
                    raw[off + 48:off + 56] = iq1m.tobytes()
            at += (ROW[t] if role != 'w2' else ROW_DOWN[t]) * rows
    blob = bytes(raw)
    (model / 'model.gguf').write_bytes(blob)
    # the pack's geometry and tensor table
    (pack / 'ds4.txt').write_text(
        f'vocab_size 129280\nhidden_size {H}\nmoe_intermediate_size {I}\nnum_hidden_layers 2\n'
        f'n_routed_experts {E}\nnum_experts_per_tok 6\nformat gguf\nmodel_dir {model}\n')
    lines = ['file 0 model.gguf']
    offset = 0
    for e in range(E):
        t = types[e]
        for role, rows, cols in (('w1', I, H), ('w3', I, H), ('w2', H, I)):
            size_e = (ROW[t] if role != 'w2' else ROW_DOWN[t]) * (I if role != 'w2' else H)
            lines.append(f't layers.0.ffn.experts.{e}.{role}.weight '
                         f'{"IQ2_XXS" if t == 16 else "IQ1_M"} 0 {offset} {size_e} {rows} {cols}')
            offset += size_e
    (pack / 'tensors.txt').write_text('\n'.join(lines) + '\n')

    reader = gsq.GsqPack(pack)
    check(reader.experts == E and reader.hidden == H and reader.inter == I and reader.layers == 2,
          f'the geometry: {reader.experts} experts, H {reader.hidden}, I {reader.inter}, {reader.layers} layers')
    gate, up, down, type_ids = reader.layer_sources(0)
    check(len(type_ids) == E and list(type_ids) == types, f'the type of every expert: {list(type_ids)}')
    check(gate and up and down and gate != up != down, 'the gate, the up and the down are separate tensors')
    check(0 < gate < 2 ** 47, 'the addresses are the mapped file')

    # the engine's plan on what the reader hands over: the layer it computes
    plan = CpuPlan(E, 4, [gate, up, down], hidden=H, inter=I, gsq_types=type_ids)
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((2, H)) * 0.02).astype(np.float32)
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
    x16 = (ctypes.c_uint16 * (2 * H))(*xb.reshape(-1).tolist())
    ids = (ctypes.c_int32 * 2)(0, 1)
    rw = (ctypes.c_float * 2)(0.5, 0.5)
    out = (ctypes.c_float * (2 * H))()
    rc = plan.forward(2, 1, ctypes.addressof(x16), ctypes.addressof(ids), ctypes.addressof(rw), ctypes.addressof(out))
    values = np.frombuffer(out, dtype=np.float32)
    check(rc >= 0 and np.isfinite(values).all() and np.abs(values).sum() > 0,
          f'the engine computed the layer from the mapped file (rc {rc}, {np.abs(values).sum():.3f})')
    plan.close()
    reader.close()

print('GSQ-PACK-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
