"""The GPU kernels with the activation dtype of the device and the software FP4 decode, against an FP32 reference.

  python tests/sm75_gpu_check.py

On a Turing card (sm_75) this is what the branch is about: no BF16 tensor cores (the activations and the dot products
are FP16) and no FP4 conversion instruction (the E2M1 values are decoded in software). The same checks run on any
GPU: the dtype and the decode follow the device.

- the software E2M1 decode of the cache's kernels (all 16 codes, both nibbles) against the reference values;
- the grouped GPU prefill (backend/prefill.py) of a synthetic layer against the same layer computed in FP32.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from xtier import arch  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        bad.append(what)


DTYPE = arch.dtype()          # FP16 below sm_80 (Turing), BF16 from sm_80 on
HW = arch.has_fp4_convert()   # the FP4 conversion instruction (sm_100 and newer)
print(f'xtier: {arch.describe()}, activations {DTYPE}, hardware FP4 conversion {HW}')

# the E2M1 values (the sign in bit 3): 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e2m1_reference(codes):
    """The E2M1 codes as FP32 values (the software decode's reference)."""
    out = []
    for c in codes:
        m, v = c & 7, E2M1[c & 7]
        out.append(-v if c & 8 else v)
    return out


from xtier.cache.pool import _halves, _tl_dtype  # noqa: E402

# the cache's kernel decodes two E2M1 values of a byte at a time: the low nibble is the even column
BN, BK = 16, 32
codes = torch.arange(16, dtype=torch.uint8, device='cuda')
q = torch.zeros((1, BN, BK // 2), dtype=torch.uint8, device='cuda')
q[0, :, :] = codes                       # byte j holds the codes (j % 16): both nibbles
scales = torch.full((1, BN, BK // 16), 0x38, dtype=torch.uint8, device='cuda')   # E4M3: 1.0
rows = torch.arange(BN, dtype=torch.int32, device='cuda')
lo, hi = _halves[(1,)](q, scales, 0, rows, 0, q.stride(0), q.stride(1), scales.stride(0), scales.stride(1),
                       BN, BK, False, HW, _tl_dtype(DTYPE))
want_lo = torch.tensor(e2m1_reference([c & 15 for c in codes.tolist()]), dtype=torch.float32)
want_hi = torch.tensor(e2m1_reference([c >> 4 for c in codes.tolist()]), dtype=torch.float32)
err_lo = float((lo.float() - want_lo.to('cuda')).abs().max())
err_hi = float((hi.float() - want_hi.to('cuda')).abs().max())
check(err_lo == 0.0, f'the E2M1 decode: the low nibbles are exact (max error {err_lo})')
check(err_hi == 0.0, f'the E2M1 decode: the high nibbles are exact (max error {err_hi})')

# the grouped GPU prefill of a synthetic layer against the same layer computed in FP32
H, I, E, K, M = 128, 64, 4, 2, 8
GROUP = 16                                   # the NVFP4 block size: one E4M3 scale per 16 values
g = torch.Generator().manual_seed(20261029)


def packed(shape, scale_shape):
    """Random E2M1 weights with E4M3 block scales (the NVFP4 layout: two values per byte, the low nibble first)."""
    w = torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)
    s = torch.randint(0x30, 0x40, scale_shape, generator=g, dtype=torch.uint8)   # E4M3 around 1.0
    return w, s


def decode(w, s, group=GROUP):
    """The packed weights as FP32: [E, rows, cols] values with their block scales applied."""
    lo = torch.stack([(w & 15), (w >> 4)], -1).reshape(*w.shape[:-1], -1).float()
    a = (lo.to(torch.int64) & 7)
    v = torch.where(a < 4, a.float() * 0.5, torch.where(a == 4, torch.full_like(a, 2.0),
           torch.where(a == 5, torch.full_like(a, 3.0), torch.where(a == 6, torch.full_like(a, 4.0),
           torch.full_like(a, 6.0)))))
    v = torch.where(lo >= 8, -v, v)
    scale = s.view(torch.float8_e4m3fn).float()                 # the E4M3 bytes, bit for bit
    scale = scale.repeat_interleave(group, dim=-1)
    return v * scale


w13, s13 = packed((E, 2 * I, H // 2), (E, 2 * I, H // GROUP))
w2, s2 = packed((E, H, I // 2), (E, H, I // GROUP))
g13 = torch.rand((E, 2), generator=g) + 0.5
g2 = torch.rand((E,), generator=g) + 0.5
w13_f = decode(w13, s13).cuda()
w2_f = decode(w2, s2).cuda()
g13, g2 = g13.cuda(), g2.cuda()

x = (torch.randn(M, H, generator=g) * 0.5).to(DTYPE).cuda()
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).int().cuda()
rw = torch.softmax(torch.randn(M, K, generator=g), -1).cuda()

# the reference: the same MoE in FP32 (the route sums in FP32, as the kernels do)
ref = torch.zeros((M, H), dtype=torch.float32, device='cuda')
xf = x.float()
for t in range(M):
    for s in range(K):
        e = int(ids[t, s])
        gate = xf[t] @ w13_f[e, :I].T * g13[e, 0]
        up = xf[t] @ w13_f[e, I:].T * g13[e, 1]
        mid = torch.nn.functional.silu(gate) * up
        ref[t] += float(rw[t, s]) * (mid @ w2_f[e].T * g2[e])

from xtier.backend.prefill import Prefill  # noqa: E402

pointers = [t.data_ptr() for t in (w13, w2, s13, s2, g13, g2)]
prefill = Prefill(E, pointers, native=None, hidden=H, inter=I, dtype=DTYPE)
out = torch.empty((M, H), dtype=DTYPE, device='cuda')
stream = torch.cuda.current_stream()
prefill.run(x.data_ptr(), out.data_ptr(), ids.data_ptr(), rw.data_ptr(), M, K, stream.cuda_stream, dtype=DTYPE)
torch.cuda.synchronize()
err = float((out.float() - ref).norm() / ref.norm())
check(err < 3e-3, f'the GPU prefill in {DTYPE} against the FP32 reference (relative L2 {err:.2e})')
prefill.close()

print('SM75-GPU-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
