"""The E2M1 (FP4) decode of XTier's kernels: the software path against the hardware conversion, bit for bit.

The packed experts hold E2M1 values with block scales; the GPUs that have the conversion instruction
(cvt.rn.f16x2.e2m1x2, sm_100 and newer) do it in hardware and the others read the nibbles with bit math
(xtier.backend.fp4.e2m1_f16). The two give the same FP16 bits, so a kernel computes the same results with either.

  python tests/fp4_decode_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from xtier.backend.fp4 import e2m1_f16  # noqa: E402

DEV = 'cuda'
torch.manual_seed(20261102)


@triton.jit
def decode_nibbles(SRC, DST, HW: tl.constexpr):
    i = tl.arange(0, 256)
    code = tl.load(SRC + i)
    if HW:
        # the instruction: two E2M1 values at a time (the low nibble first)
        r = tl.inline_asm_elementwise(
            asm="{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
            constraints="=r,r", args=[code.to(tl.uint32)], dtype=tl.uint32, is_pure=True, pack=1)
        lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        tl.store(DST + i, lo.to(tl.float32) + hi.to(tl.float32) * 256.0)
    else:
        lo = e2m1_f16(code & 15)
        hi = e2m1_f16((code >> 4) & 15)
        tl.store(DST + i, lo.to(tl.float32) + hi.to(tl.float32) * 256.0)


def main():
    src = torch.arange(256, dtype=torch.uint8, device=DEV)
    soft = torch.empty(256, dtype=torch.float32, device=DEV)
    decode_nibbles[(1,)](src, soft, False)

    # the reference: the E2M1 grid, built from its definition
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
    low = (src & 15).long()
    high = ((src >> 4) & 15).long()
    want = (magnitudes[low & 7] * torch.where(low & 8 != 0, -1.0, 1.0)
            + 256.0 * magnitudes[high & 7] * torch.where(high & 8 != 0, -1.0, 1.0))
    bad = int((soft != want).sum())
    print(f'the 256 e2m1 byte pairs: the mismatches {bad}')
    if bad:
        for i in range(256):
            if soft[i] != want[i]:
                print(f'  the byte {src[i].item():#04x}: the software {soft[i].item()} the reference {want[i].item()}')
                break

    hw_ok = None
    try:
        hw = torch.empty(256, dtype=torch.float32, device=DEV)
        decode_nibbles[(1,)](src, hw, True)
        hw_ok = int((hw != want).sum())
        print(f'the hardware decode: the mismatches {hw_ok}')
    except Exception as exc:  # noqa: BLE001 - the instruction needs sm_100
        print(f'the hardware decode is not available here: {type(exc).__name__}: {exc}')

    ok = bad == 0 and (hw_ok is None or hw_ok == 0)
    print('OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
