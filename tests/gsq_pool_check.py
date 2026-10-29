"""The GPU's hot expert cache for the GSQ path (xtier/cache/gsq_pool.py) against the CPU engine.

  python tests/gsq_pool_check.py

The GSQ pool keeps the hot experts' IQ bytes in the VRAM and decodes them on the GPU while the CPU engine computes
the rest: the routes of the cached experts get the id -1 for the CPU (the engine skips them) and the GPU's part is
added to the output. This builds a small synthetic GSQ layer (the IQ blocks hold 256 values, so H and I are multiples
of 256), runs the engine's forward for the routes and the pool's decode for the same routes, and compares.

The cached experts cover the four IQ formats (the gate and the up of one, the down of another), and the call mixes
them with the routes the CPU computes.

The activations are the same numbers on both sides: the engine reads BF16 (it widens them) and the GPU's dot takes
FP16, so the values are multiples of 1/256 (exact in BF16 and in FP16).

The tolerance: the engine decodes the blocks in FP32 and multiplies in FP32; the GPU decodes in FP32 too, but the
dot takes the weights in FP16 and keeps the intermediate in FP16 (the card's tensor cores). The error is therefore
the FP16 rounding of the weights and of the intermediate, not the decode.
"""

import ctypes as C
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier import arch  # noqa: E402
from xtier.cache.gsq_pool import GsqPool  # noqa: E402
from xtier.engine import CpuPlan, Mailbox, engine  # noqa: E402

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
E, H, I, K, M = 10, 512, 256, 1, 8                 # the engine's minimum is 10 experts; H and I: multiples of 256
THREADS = 4
CACHED = [0, 1, 2, 3]                              # the experts the GPU holds (one of every IQ format)
OTHERS = [4, 5, 6, 7]                              # the experts the CPU computes
TOL = 5e-3                                        # the FP16 arithmetic of the GPU against the engine's FP32
rng = np.random.default_rng(20261029)
DTYPE = arch.dtype()
print(f'xtier: {arch.describe()}, activations {DTYPE}')

# The type of every expert per role: the pack quantizes every role on its own, so the down's type differs. The
# cached experts (0..3) cover the four formats on both roles.
types = np.array([29, 17, 16, 19, 29, 17, 16, 19, 29, 16], dtype=np.uint8)       # the gate and the up
types_down = np.array([16, 19, 29, 17, 16, 19, 29, 17, 16, 19], dtype=np.uint8)   # the down


def sane(blob, type_id):
    """The scale fields of the blocks: a finite fp16 (random bytes would give NaNs).

    The scale is 2^-8, so the weights are O(1) and the dots stay in the FP16 range.
    """
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


# The engine: the layer's tensors (one per role) and the type of every expert per role.
plan = CpuPlan(E, THREADS, [gate.ctypes.data, up.ctypes.data, down.ctypes.data], hidden=H, inter=I,
               gsq_types=(types.ctypes.data, types.ctypes.data, types_down.ctypes.data))
print('engine ISA:', plan.lib.xtier_isa().decode())

# The routes: the first tokens go to the cached experts (the GPU), the others to the experts the CPU computes.
x = rng.integers(-256, 256, (M, H)).astype(np.float32) / 256.0    # exact in BF16 and in FP16
ids = np.array([[e] for e in CACHED + OTHERS], dtype=np.int32)
rw = (0.5 + 0.5 * rng.random((M, K))).astype(np.float32)
x16 = (C.c_uint16 * (M * H))(*(x.view(np.uint32) >> 16).astype(np.uint16).reshape(-1).tolist())
ids32 = (C.c_int32 * (M * K))(*ids.reshape(-1).tolist())
rw32 = (C.c_float * (M * K))(*rw.reshape(-1).tolist())

out = (C.c_float * (M * H))()
rc = plan.forward(M, K, C.addressof(x16), C.addressof(ids32), C.addressof(rw32), C.addressof(out))
check(rc >= 0, f'the engine computed the layer (rc {rc})')
ref = np.frombuffer(out, dtype=np.float32).reshape(M, H).copy()

# The CPU's part of the cached routes: the engine skips the id -1 (the routes are computed on the GPU).
skip = (C.c_int32 * (M * K))(*([-1] * len(CACHED) + OTHERS).copy())
out = (C.c_float * (M * H))()      # zeroed: the engine writes only the routes it computes
rc = plan.forward(M, K, C.addressof(x16), C.addressof(skip), C.addressof(rw32), C.addressof(out))
got = np.frombuffer(out, dtype=np.float32).reshape(M, H)
check(rc >= 0 and not got[:len(CACHED)].any(), 'the engine skips the id -1 routes (their rows stay zero)')
check(np.array_equal(got[len(CACHED):], ref[len(CACHED):]), 'the engine computes the other routes unchanged')

# The FP32 reference: the engine's own decode of every row, then the dots in FP32.
lib = engine()


def decode_row(type_id, blob, n):
    """One row of n values as FP32 (the engine's decode)."""
    dst = np.zeros(n, dtype=np.float32)
    if lib.xtier_gsq_decode(type_id, blob.ctypes.data, dst.ctypes.data, n) < 0:
        raise RuntimeError('xtier: the engine refused to decode the block')
    return dst


def expert_out(e, xt):
    """The expert's output for one token's activations (FP32, the exact values)."""
    rg, rd = len(gates[e]) // I, len(downs[e]) // H
    gate_f = np.stack([decode_row(int(types[e]), gates[e][r * rg:(r + 1) * rg], H) for r in range(I)])
    up_f = np.stack([decode_row(int(types[e]), ups[e][r * rg:(r + 1) * rg], H) for r in range(I)])
    down_f = np.stack([decode_row(int(types_down[e]), downs[e][r * rd:(r + 1) * rd], I) for r in range(H)])
    g, u = gate_f @ xt, up_f @ xt
    return (g / (1.0 + np.exp(-g)) * u) @ down_f.T


want = np.stack([rw[t, 0] * expert_out(e, x[t]) for t, e in enumerate(CACHED)])   # the route's weight

# The GPU: the pool holds the hot experts' bytes in the VRAM and computes their routes.
os.environ.setdefault('XTIER_PROBE_DISPATCH_CPU', str(max(os.sched_getaffinity(0))))
box = Mailbox(plan)
pool = GsqPool(Pack(E, H, I), 0, len(CACHED), box, dtype=DTYPE)
stream = torch.cuda.Stream()
n = pool.fill(CACHED, stream)
check(n == len(CACHED), f'the pool uploaded the hot experts ({n} experts, {pool.bytes()} bytes in the VRAM)')

xd = torch.from_numpy(x).to(DTYPE).cuda()
ids_d = torch.from_numpy(ids).int().cuda()
rw_d = torch.from_numpy(rw).float().cuda()
gpu = torch.empty((M, H), dtype=torch.float32, device='cuda')
pool.decode(torch.cuda.current_stream().cuda_stream, M, K, xd.data_ptr(), ids_d.data_ptr(), rw_d.data_ptr(),
            gpu.data_ptr())
torch.cuda.synchronize()
got = gpu.cpu().numpy()

errs = [rel(got[t], ref[t]) for t in range(len(CACHED))]
err_cpu = max(rel(got[t], ref[t]) for t in range(len(CACHED), M))
err_ref = max(rel(got[t], want[t]) for t in range(len(CACHED)))
print('xtier: the GPU\'s experts against the engine\'s: relative L2 ' +
      ', '.join(f'{int(types[e])}/{int(types_down[e])}: {err:.2e}' for e, err in zip(CACHED, errs)) +
      f' (the engine against the FP32 reference: {rel(ref[0], want[0]):.2e})')
check(max(errs) < TOL, f'the GPU\'s experts against the engine\'s (relative L2 {max(errs):.2e}, tolerance {TOL:.0e})')
check(err_ref < TOL, f'the GPU\'s experts against the FP32 reference (relative L2 {err_ref:.2e})')
check(err_cpu == 0.0, f'the CPU\'s routes through the pool are the engine\'s own (relative L2 {err_cpu:.2e})')

box.close()
plan.close()
print('GSQ-POOL-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
