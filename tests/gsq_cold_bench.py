"""The GSQ cold pool on the real checkpoint: the RAM slots and the drive (the Optane), measured.

  python tests/gsq_cold_bench.py [--layers 0,10,20,39] [--steps 20] [--fraction 0.3]

What it measures, per layer: the time to read the hot experts into the RAM slots (the load), then a decode step's
routes (drawn from the pack's usage profile: the experts that are really used) through the layer: the experts that
are not in RAM are read from the drive (O_DIRECT), the misses per step, the bytes the drive moved and the bandwidth.
The same routes without the pool (the experts read from the mapped file) are the reference.

The store's files are the checkpoint's own shards: the drive is the one the model sits on.
"""
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from xtier.backend import gsq  # noqa: E402
from xtier.cache import gsq_cold  # noqa: E402
from xtier.engine import CpuPlan  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--pack', default='/mnt/optane/DeepSeek-V4.1-Flash/maya-ds4-gsq')
P.add_argument('--layers', default='0,10,20,39', help='the layers to measure (they sit on different shards)')
P.add_argument('--steps', type=int, default=20, help='the decode steps per layer')
P.add_argument('--fraction', type=float, default=0.3,
               help='the share of the experts on the drive (XTIER_COLD_FRACTION)')
P.add_argument('--topk', type=int, default=6, help='the experts per token')
P.add_argument('--threads', type=int, default=int(os.environ.get('XTIER_THREADS', 24)))
A = P.parse_args()

os.environ['XTIER_COLD_FRACTION'] = str(A.fraction)

pack = gsq.GsqPack(A.pack)
print(f'pack: {pack.path} ({pack.experts} experts, H {pack.hidden}, I {pack.inter}, {pack.layers} layers)')
counts = pack.profile()
print(f'the usage profile: {len(counts)} layers' if counts else 'the usage profile: none')

layers = [int(x) for x in A.layers.split(',')]
rng = np.random.default_rng(11)


def routes_for(layer, steps):
    """The decode steps' routes: the experts drawn from the layer's usage profile (the ones that are really used)."""
    row = np.asarray(counts.get(layer, [1] * pack.experts), dtype=np.float64)
    p = row / row.sum()
    out = []
    for _ in range(steps):
        out.append(rng.choice(pack.experts, size=A.topk, replace=False, p=p).tolist())
    return out


def run(plan, routes):
    """The steps through the layer; the time they took."""
    H = pack.hidden
    x = (rng.standard_normal((1, H)) * 0.02).astype(np.float32)
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
    x16 = (ctypes.c_uint16 * H)(*xb.reshape(-1).tolist())
    rw = (ctypes.c_float * A.topk)(*([1.0 / A.topk] * A.topk))
    out = (ctypes.c_float * H)()
    ids = (ctypes.c_int32 * A.topk)()
    t0 = time.perf_counter()
    for r in routes:
        ids[:] = (ctypes.c_int32 * A.topk)(*r)
        plan.forward(1, A.topk, ctypes.addressof(x16), ctypes.addressof(ids), ctypes.addressof(rw),
                     ctypes.addressof(out))
    return time.perf_counter() - t0


print(f'\n{"layer":>5} {"experts":>8} {"on the drive":>12} {"load":>8} {"step":>9} {"misses":>7} {"MB/step":>8} '
      f'{"GB/s":>6} {"ref step":>9}')
for layer in layers:
    gate, up, down, tg, tu, td = pack.layer_sources(layer)
    plan = CpuPlan(pack.experts, A.threads, [gate, up, down], hidden=pack.hidden, inter=pack.inter,
                   gsq_types=(tg, tu, td))
    routes = routes_for(layer, A.steps)
    # the reference: the same routes with the experts read from the mapped file (the first run warms the page cache,
    # the second is what the file's cache serves)
    run(plan, routes)
    t_ref = run(plan, routes)
    # the pool: the hot experts in the RAM slots, the rest in the checkpoint's files on the drive
    t0 = time.perf_counter()
    gsq_cold.attach(plan, pack, layer)
    load = time.perf_counter() - t0
    before = plan.cold_stats()
    stats_before = gsq_cold._store[0].stats()
    t_pool = run(plan, routes)
    after = plan.cold_stats()
    misses = after[1] - before[1]
    stats = gsq_cold._store[0].stats()
    read = stats[0] - stats_before[0]
    waited = stats[3] - stats_before[3]   # the seconds the store spent reading
    mb = read / 1e6
    bandwidth = read / 1e9 / waited if waited > 0 else 0.0
    print(f'{layer:>5} {pack.experts:>8} {plan.slot_table().count(-1):>10} {load:>7.2f}s '
          f'{t_pool / A.steps * 1e3:>8.1f}ms {misses / A.steps:>7.1f} {mb / A.steps:>8.1f} {bandwidth:>6.2f} '
          f'{t_ref / A.steps * 1e3:>8.1f}ms')
    plan.close()

pack.close()
