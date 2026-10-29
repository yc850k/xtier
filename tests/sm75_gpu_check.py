"""The GPU kernels with the activation dtype of the device and the software FP4 decode, against an FP32 reference.

  python tests/sm75_gpu_check.py

On a Turing card (sm_75) this is what the branch is about: no BF16 tensor cores (the activations and the dot products
are FP16) and no FP4 conversion instruction (the E2M1 values are decoded in software). The same checks run on any GPU:
the dtype and the decode follow the device.

- the cache's gate/up kernel (cache/pool.py) on packed NVFP4 and MXFP4 experts: the decoded weights and the 16-bit
  dot products against the same layer computed in FP32;
- the grouped GPU prefill (backend/prefill.py) of a synthetic layer against the same layer computed in FP32.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from xtier import arch  # noqa: E402
from xtier.backend.prefill import Prefill  # noqa: E402
from xtier.cache.pool import _gate_up, _tl_dtype  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


DTYPE = arch.dtype()          # FP16 below sm_80 (Turing), BF16 from sm_80 on
HW4 = arch.has_fp4_convert()  # the FP4 conversion instruction (sm_100 and newer)
HW8 = arch.has_fp8()          # the FP8 conversion of the E4M3 scales (sm_89 and newer)
print(f'xtier: {arch.describe()}, activations {DTYPE}, hardware FP4 conversion {HW4}, FP8 {HW8}')

g = torch.Generator().manual_seed(20261029)


def e2m1(values):
    """The E2M1 codes as FP32 values (the sign in bit 3)."""
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    a = values & 7
    v = torch.where(a < 4, a.float() * 0.5, torch.where(a == 4, torch.full_like(a, 2.0),
           torch.where(a == 5, torch.full_like(a, 3.0), torch.where(a == 6, torch.full_like(a, 4.0),
           torch.full_like(a, 6.0)))))
    return torch.where(values >= 8, -v, v)


def unpack(w, group):
    """The packed weights as FP32: two values per byte (the low nibble first), the block scale applied."""
    values = torch.stack([(w & 15), (w >> 4)], -1).reshape(*w.shape[:-1], -1)
    return e2m1(values)


def scales(s, group, mx):
    """The block scales as FP32: E4M3 (NVFP4, per 16 values) or UE8M0 (MXFP4, per 32 values)."""
    if mx:
        e = s.to(torch.int64)
        value = torch.where(e == 0, torch.full_like(e, 0, dtype=torch.float64), torch.exp2((e - 127).float()))
    else:
        value = s.view(torch.float8_e4m3fn).float()
    return value.repeat_interleave(group, dim=-1)


# the cache's gate/up kernel: one token through one expert of a slot (the decode path's arithmetic)
def cache_gate_up(mx):
    H, I, BN, BK, BM = 128, 64, 32, 64, 16
    group = 32 if mx else 16
    q13 = torch.randint(0, 256, (1, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    s13 = (torch.randint(120, 128, (1, 2 * I, H // group), dtype=torch.uint8, generator=g) if mx
           else torch.randint(0x30, 0x40, (1, 2 * I, H // group), dtype=torch.uint8, generator=g))
    g13 = (torch.rand((1, 2), generator=g) + 0.5).float()
    x = (torch.randn(BM, H, generator=g) * 0.5).to(DTYPE)
    # the device copies stay alive until the kernel has run (the launch is asynchronous)
    x_d, sl, se = x.cuda(), torch.zeros(1, dtype=torch.int32, device='cuda'), torch.zeros(1, dtype=torch.int32, device='cuda')
    q_d, s_d, g_d = q13.cuda(), s13.cuda(), g13.cuda()
    mid = torch.zeros((BM, I), dtype=DTYPE, device='cuda')
    _gate_up[(1, I // BN)](x_d, sl, se, q_d, s_d, g_d, mid,
                           BM, H, I, 0.0, BM, BN, BK, mx, 1.0, 0.0, False, HW4, HW8, _tl_dtype(DTYPE))
    torch.cuda.synchronize()
    # the reference: the same layer in FP32 (the decoded weights, the dot products in FP32)
    w = unpack(q13, group).float() * scales(s13, group, mx).float()
    xf = x.float()
    gate = xf @ w[0, :I].T * g13[0, 0]
    up = xf @ w[0, I:].T * g13[0, 1]
    want = torch.nn.functional.silu(gate) * up
    err = float((mid.float() - want.cuda()).norm() / want.norm())
    name = 'mxfp4' if mx else 'nvfp4'
    check(err < 3e-3, f'the cache\'s gate/up on {name} experts in {DTYPE} (relative L2 {err:.2e})')


cache_gate_up(False)
cache_gate_up(True)

# the grouped GPU prefill of a synthetic layer against the same layer computed in FP32
H, I, E, K, M = 128, 64, 4, 2, 8
GROUP = 16                                   # the NVFP4 block size: one E4M3 scale per 16 values
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
s13 = torch.randint(0x30, 0x40, (E, 2 * I, H // GROUP), dtype=torch.uint8, generator=g)
s2 = torch.randint(0x30, 0x40, (E, H, I // GROUP), dtype=torch.uint8, generator=g)
g13 = (torch.rand((E, 2), generator=g) + 0.5).float()
g2 = (torch.rand((E,), generator=g) + 0.5).float()
w13_f = (unpack(w13, GROUP).float() * scales(s13, GROUP, False).float()).cuda()
w2_f = (unpack(w2, GROUP).float() * scales(s2, GROUP, False).float()).cuda()
g13_d, g2_d = g13.cuda(), g2.cuda()   # the device copies (the host ones stay for the engine's pointers)

x = (torch.randn(M, H, generator=g) * 0.5).to(DTYPE).cuda()
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).int().cuda()
rw = torch.softmax(torch.randn(M, K, generator=g), -1).cuda()

ref = torch.zeros((M, H), dtype=torch.float32, device='cuda')
xf = x.float()
for t in range(M):
    for s in range(K):
        e = int(ids[t, s])
        gate = xf[t] @ w13_f[e, :I].T * g13_d[e, 0]
        up = xf[t] @ w13_f[e, I:].T * g13_d[e, 1]
        mid = torch.nn.functional.silu(gate) * up
        ref[t] += float(rw[t, s]) * (mid @ w2_f[e].T * g2_d[e])

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
