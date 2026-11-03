"""The batched prefill of the GSQ path (xtier/cache/gsq_pool.py): the hot experts' routes on the GPU (their IQ
blocks decoded in the tile kernels) and the rest on the CPU engine, against the engine's own result.

  python tests/gsq_prefill_check.py

The point of the path: the hot experts' bytes are in the VRAM (the upload happened once, at the load), so a batched
prefill computes them on the GPU with no PCIe transfer at all, and the CPU engine only sees the routes of the experts
that are not cached.

The tolerance: the engine decodes the blocks in FP32 and multiplies in FP32; the GPU decodes in FP32 too, but the dot
takes the weights in FP16 and keeps the intermediate in FP16 (the card's tensor cores). The error is therefore the
FP16 rounding of the weights and of the intermediate, not the decode.
"""

import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier import arch  # noqa: E402
from xtier.cache.gsq_pool import GsqPool, prefill  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


def rel(a, b):
    """The relative L2 error of a against b."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


BLOCK = 256
BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 29: 56}     # the bytes of one block, per type (csrc/engine/gsq.h)
E, H, I, THREADS = 10, 512, 256, 4                 # the engine's minimum is 10 experts; H and I: multiples of 256
K, M = 2, 64                                      # the routes per token, the tokens of the batch
CACHED = [0, 1, 2, 3]                              # the experts the GPU holds (one of every IQ format)
TOL = 5e-3                                        # the FP16 arithmetic of the GPU against the engine's FP32
rng = np.random.default_rng(20261029)
DTYPE = arch.dtype()
print(f'xtier: {arch.describe()}, activations {DTYPE}')

# The type of every expert per role: the pack quantizes every role on its own, so the down's type differs.
types = np.array([29, 17, 16, 19, 29, 17, 16, 19, 29, 16], dtype=np.uint8)       # the gate and the up
types_down = np.array([16, 19, 29, 17, 16, 19, 29, 17, 16, 19], dtype=np.uint8)   # the down


def sane(blob, type_id):
    """The scale fields of the blocks: a finite fp16 (random bytes would give NaNs)."""
    nb = BLOCK_BYTES[type_id]
    if type_id == 29:   # IQ1_M: the scales at 48 (the block's d is assembled from their high nibbles)
        sc = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x1000]).tobytes(), dtype=np.uint8)
        for i in range(len(blob) // nb):
            blob[i * nb + 48:i * nb + 56] = sc
    else:               # the fp16 d at the start of every block
        h = np.frombuffer(np.float16(2 ** -8).tobytes(), dtype=np.uint8)
        for i in range(len(blob) // nb):
            blob[i * nb:i * nb + 2] = h
    return blob


def tensor(type_id, rows, values):
    """One expert's tensor: `rows` rows of `values` values (a run of 256-value blocks a row)."""
    nb = BLOCK_BYTES[type_id]
    blob = rng.integers(0, 256, rows * (values // BLOCK) * nb, dtype=np.uint8).copy()
    return sane(blob, type_id)


gates, ups, downs = [], [], []
for e in range(E):
    gates.append(tensor(int(types[e]), I, H))
    ups.append(tensor(int(types[e]), I, H))
    downs.append(tensor(int(types_down[e]), H, I))
gate, up, down = np.concatenate(gates), np.concatenate(ups), np.concatenate(downs)


class Pack:
    """The pack of the synthetic layer: the experts' bytes and their types (the API the pool uses)."""

    def __init__(self, experts, hidden, inter):
        self.experts, self.hidden, self.inter = experts, hidden, inter

    def expert_locations(self, layer):
        return [{'w1': (0, 0, len(gates[e]), int(types[e])),
                 'w3': (0, 0, len(ups[e]), int(types[e])),
                 'w2': (0, 0, len(downs[e]), int(types_down[e]))} for e in range(self.experts)]

    def expert_bytes(self, role, layer, expert):
        blob = {'w1': gates[expert], 'w3': ups[expert], 'w2': downs[expert]}[role]
        kind = int(types_down[expert]) if role == 'w2' else int(types[expert])
        return blob.ctypes.data, len(blob), kind


plan = CpuPlan(E, THREADS, [gate.ctypes.data, up.ctypes.data, down.ctypes.data], hidden=H, inter=I,
               gsq_types=(types.ctypes.data, types.ctypes.data, types_down.ctypes.data))
print('engine ISA:', plan.lib.xtier_isa().decode())

# The batch: every token routes to a cached expert and to one the CPU computes (the routes are mixed on purpose).
x = rng.integers(-256, 256, (M, H)).astype(np.float32) / 256.0    # exact in BF16 and in FP16
ids = np.empty((M, K), dtype=np.int32)
for t in range(M):
    ids[t, 0] = CACHED[t % len(CACHED)]
    ids[t, 1] = 4 + (t % (E - len(CACHED)))
rw = (0.5 + 0.5 * rng.random((M, K))).astype(np.float32)
x16 = (C.c_uint16 * (M * H))(*(x.view(np.uint32) >> 16).astype(np.uint16).reshape(-1).tolist())
ids32 = (C.c_int32 * (M * K))(*ids.reshape(-1).tolist())
rw32 = (C.c_float * (M * K))(*rw.reshape(-1).tolist())

out = (C.c_float * (M * H))()
rc = plan.forward(M, K, C.addressof(x16), C.addressof(ids32), C.addressof(rw32), C.addressof(out))
check(rc >= 0, f'the engine computed the layer (rc {rc})')
ref = np.frombuffer(out, dtype=np.float32).reshape(M, H).copy()

# The GPU: the pool holds the hot experts' bytes in the VRAM and computes their routes; the CPU engine the rest.
os.environ.setdefault('XTIER_PROBE_DISPATCH_CPU', str(max(os.sched_getaffinity(0))))
box = Mailbox(plan)
pool = GsqPool(Pack(E, H, I), 0, len(CACHED), box, dtype=DTYPE)
stream = torch.cuda.Stream()
n = pool.fill(CACHED, stream)
check(n == len(CACHED), f'the pool uploaded the hot experts ({n} experts, {pool.bytes()} bytes in the VRAM)')

xd = torch.from_numpy(x).to(DTYPE).cuda()
ids_d = torch.from_numpy(ids).int().cuda()
rw_d = torch.from_numpy(rw).float().cuda()


class CountingPlan:
    """The engine's plan, counting the calls and keeping the output pointer (the engine's part must be empty)."""

    def __init__(self, plan):
        self.plan, self.calls, self.out = plan, 0, None

    def forward(self, n, k, x, ids, w, out):
        self.calls += 1
        self.out = out
        return self.plan.forward(n, k, x, ids, w, out)


counting = CountingPlan(plan)
got = prefill(pool, counting, xd, ids_d, rw_d)
torch.cuda.synchronize()
got = got.cpu().numpy()
# the engine's part: every route was computed on the GPU (the hot experts from the VRAM, the others uploaded), so
# the engine's output buffer is untouched
engine_out = np.frombuffer((C.c_float * (M * H)).from_address(counting.out), dtype=np.float32)
check(counting.calls == 1 and not engine_out.any(),
      f'the engine computed nothing (its {counting.calls} call wrote no value): the GPU did the whole prefill')

errs = [rel(got[t], ref[t]) for t in range(M)]
print('xtier: the batched prefill against the engine: the relative L2 ' +
      ', '.join(f'{e:.2e}' for e in errs[:6]) + (f', ... (the max {max(errs):.2e})' if len(errs) > 6 else ''))
check(max(errs) < TOL, f'the batched prefill against the engine (relative L2 {max(errs):.2e}, tolerance {TOL:.0e})')

box.close()
plan.close()
print('GSQ-PREFILL-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
