"""The GSQ cold pool (the engine's RAM slots and the drive): the experts that are not in RAM are read from the
checkpoint's own file (O_DIRECT) when a forward routes to them, and the layer computes the same.

  python tests/gsq_cold_pool_check.py

The pack is a real file on disk (the store reads it with O_DIRECT), so the pool's path is the one the Optane drive
serves in production: the expert's three tensors are read from the file into the RAM slot of the coldest resident
expert, and the kernel reads them there.
"""
import ctypes
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from xtier.backend import gsq  # noqa: E402
from xtier.engine import ColdStore, CpuPlan  # noqa: E402

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
types_down = [16, 29, 16, 29, 16, 16, 29, 16, 29, 16]
SLOTS = 4          # the RAM slots of the layer (the other 6 experts are on the drive)


def forward(plan, ids):
    """One forward of the routes (the ids per token); the FP32 output."""
    m, k = len(ids), len(ids[0])
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((m, H)) * 0.02).astype(np.float32)
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
    x16 = (ctypes.c_uint16 * (m * H))(*xb.reshape(-1).tolist())
    ids_a = (ctypes.c_int32 * (m * k))(*[i for row in ids for i in row])
    rw = (ctypes.c_float * (m * k))(*[0.5] * (m * k))
    out = (ctypes.c_float * (m * H))()
    rc = plan.forward(m, k, ctypes.addressof(x16), ctypes.addressof(ids_a), ctypes.addressof(rw), ctypes.addressof(out))
    if rc < 0:
        raise RuntimeError('the forward failed')
    return np.frombuffer(out, dtype=np.float32).copy()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / 'model'
    pack = model / 'maya-ds4-gsq'
    pack.mkdir(parents=True)
    # the GGUF file: the experts' bytes (the store reads them with O_DIRECT, the pack maps them)
    size = sum(ROW[t] * I for t in types) * 2 + sum(ROW_DOWN[t] * H for t in types_down)
    raw = bytearray(np.random.default_rng(7).integers(0, 256, size, dtype=np.uint8).tobytes())
    # the scale fields of the blocks: a finite fp16 (random bytes would give NaNs)
    d16 = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    iq1m = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x3000]).tobytes(), dtype=np.uint8)
    offsets = {}
    at = 0
    for role in ('w1', 'w3', 'w2'):
        for e in range(E):
            t = types_down[e] if role == 'w2' else types[e]
            rows = H if role == 'w2' else I
            row_bytes = ROW_DOWN[t] if role == 'w2' else ROW[t]
            block = BLOCK[t]
            for i in range(rows * (row_bytes // block)):
                off = at + i * block
                if t == 16:
                    raw[off:off + 2] = d16.tobytes()
                else:
                    raw[off + 48:off + 56] = iq1m.tobytes()
            offsets[(role, e)] = at
            at += row_bytes * rows
    (model / 'model.gguf').write_bytes(bytes(raw))
    (pack / 'ds4.txt').write_text(
        f'vocab_size 129280\nhidden_size {H}\nmoe_intermediate_size {I}\nnum_hidden_layers 2\n'
        f'n_routed_experts {E}\nnum_experts_per_tok 6\nformat gguf\nmodel_dir {model}\n')
    lines = ['file 0 model.gguf']
    for e in range(E):
        for role, rows, cols in (('w1', I, H), ('w3', I, H), ('w2', H, I)):
            t = types_down[e] if role == 'w2' else types[e]
            row_bytes = ROW_DOWN[t] if role == 'w2' else ROW[t]
            lines.append(f't layers.0.ffn.experts.{e}.{role}.weight {"IQ2_XXS" if t == 16 else "IQ1_M"} 0 '
                         f'{offsets[(role, e)]} {row_bytes * rows} {rows} {cols}')
    (pack / 'tensors.txt').write_text('\n'.join(lines) + '\n')

    reader = gsq.GsqPack(pack)
    gate, up, down, type_gate, type_up, type_down = reader.layer_sources(0)
    locations = reader.expert_locations(0)

    # the reference: the plan without the pool (the experts are read from the mapped file)
    ref_plan = CpuPlan(E, 4, [gate, up, down], hidden=H, inter=I, gsq_types=(type_gate, type_up, type_down))
    routes = [[0, 1], [5, 7]]        # two tokens, two experts each: 0 and 1 are in RAM, 5 and 7 are not
    reference = forward(ref_plan, routes)
    ref_plan.close()

    # the pool: the store is the checkpoint's own file (read-only, O_DIRECT), the hot experts in the RAM slots
    store = ColdStore([model / 'model.gguf'], threads=2, writable=False)
    plan = CpuPlan(E, 4, [gate, up, down], hidden=H, inter=I, gsq_types=(type_gate, type_up, type_down))
    files, offsets_a = [], []
    for e in range(E):
        for role in ('w1', 'w3', 'w2'):
            files.append(0)
            offsets_a.append(locations[e][role][1])
    resident = [0, 1, 2, 3]
    scores = [float(10 - e) for e in range(E)]
    plan.attach_cold_gsq(store, files, offsets_a, resident, scores, 50.0)
    plan.dynamic = SLOTS
    table = plan.slot_table()
    check(table[0] >= 0 and table[5] < 0, f'the RAM slots hold the resident experts ({table})')

    values = forward(plan, routes)
    forwards, loads, seconds, small_forwards, small_loads, small_seconds = plan.cold_stats()
    check(loads == 2, f'the experts that were not in RAM were read from the drive ({loads} reads)')
    check(forwards >= 1 and small_forwards >= 1, f'the forwards were counted ({forwards} forwards)')
    # the store moved the resident experts' bytes (read into their slots at load) and the cold ones' (read on demand)
    want = sum(locations[e][role][2] for e in list(resident) + [5, 7] for role in ('w1', 'w3', 'w2'))
    read = store.stats()[0]
    check(read == want, f'the drive moved the experts\' bytes ({read} of {want})')
    rel = np.linalg.norm(values - reference) / max(np.linalg.norm(reference), 1e-30)
    check(rel < 1e-5, f'the pooled layer computed the same as the one without the pool (relative L2 {rel:.2e})')
    check(seconds > 0, f'the forward waited for the drive ({seconds * 1e3:.1f} ms)')

    # the second forward: the experts are in RAM now (the pool keeps them), so nothing is read
    values2 = forward(plan, routes)
    forwards2, loads2 = plan.cold_stats()[0], plan.cold_stats()[1]
    check(loads2 == loads, f'the experts stayed in RAM ({loads2} reads after the second forward)')
    rel2 = np.linalg.norm(values2 - reference) / max(np.linalg.norm(reference), 1e-30)
    check(rel2 < 1e-5, f'the second forward computed the same (relative L2 {rel2:.2e})')
    plan.close()
    reader.close()

print('GSQ-COLD-POOL-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
