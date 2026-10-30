"""The mailbox staging (the routing weights and the ids) against the engine called directly: the same numbers,
bitwise.

The decode path stages the call's inputs into the mailbox's slot (a Triton kernel writes into the mapped host
memory) and the engine reads them; this checks that the engine sees what the caller staged.

The experts are synthetic NVFP4 rows (the engine's layout: the E2M1 values packed two per byte, one E4M3 scale per
16 values); the engine's arithmetic is not what this checks, only that the staged inputs are the ones it reads.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('XTIER_MAIN_CPUS', 'auto')
os.environ.setdefault('XTIER_THREADS', '4')
os.environ.setdefault('XTIER_PROBE_DISPATCH_CPU', '27')   # the dispatcher's own CPU (the default 80 may be offline)
import torch  # noqa: E402

from xtier.backend import kernels  # noqa: E402
from xtier.engine import CpuPlan, Mailbox  # noqa: E402

E, H, I, K = 16, 128, 64, 3
GROUP = 16                 # the NVFP4 block size: one E4M3 scale per 16 values
torch.manual_seed(20261030)
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
s13 = torch.randint(0x30, 0x40, (E, 2 * I, H // GROUP), dtype=torch.uint8)
s2 = torch.randint(0x30, 0x40, (E, H, I // GROUP), dtype=torch.uint8)
g13 = torch.ones((E, 2), dtype=torch.float32)
g2 = torch.ones((E,), dtype=torch.float32)
plan = CpuPlan(E, 4, [w13.data_ptr(), w2.data_ptr(), s13.data_ptr(), s2.data_ptr(), g13.data_ptr(), g2.data_ptr()],
               hidden=H, inter=I, scale_format='nvfp4')
box = Mailbox(plan)

bad = 0
for m in range(1, 5):
    x = torch.randn(m, H, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(m)]).int()
    rw = torch.softmax(torch.randn(m, K), -1)
    ref = torch.empty(m, H)
    plan.forward(m, K, x.data_ptr(), ids.data_ptr(), rw.data_ptr(), ref.data_ptr())

    xd, idsd, rwd = x.cuda(), ids.cuda(), rw.cuda()
    out = torch.empty(m, H, device='cuda')
    stream = torch.cuda.current_stream().cuda_stream
    slot = box.slot(stream, m, K)
    kernels.pack(slot, m, K, xd.data_ptr(), idsd.data_ptr(), rwd.data_ptr(), H)
    box.notify(stream, m, K)
    kernels.unpack(slot, m, out.data_ptr(), H)
    torch.cuda.synchronize()
    err = (out.cpu() - ref).abs().max().item()
    same = torch.equal(out.cpu(), ref)
    print(f'm {m}: the max error {err} the bitwise match {same}')
    bad += 0 if same else 1

print('OK' if bad == 0 else 'FAIL')
sys.exit(0 if bad == 0 else 1)
