"""The E4M3 (FP8) encode of XTier's kernels: the software path against torch's own float8_e4m3fn cast.

The software path is what a card without FP8 tensor cores takes (Turing, the RTX 30 series): Triton has no
float8e4nv below sm_89, so a kernel that stores e4m3 bytes rounds them with bit math
(xtier.backend.fp4.e4m3_bits). The values are the ones cvt.rn.satfinite.e4m3x2.f32 gives: round to nearest even,
saturating to +-448, NaN to 0x7F with its sign.

  python tests/fp8_encode_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from xtier.backend.fp4 import e4m3_bits  # noqa: E402

DEV = 'cuda'
torch.manual_seed(20261102)


@triton.jit
def encode_bytes(SRC, DST, HW: tl.constexpr):
    i = tl.arange(0, 1024)
    tl.store(DST + i, e4m3_bits(tl.load(SRC + i), HW))


def main():
    # the values a kernel would store: the normals, the subnormals, the saturations and the specials
    g = torch.Generator(device='cpu').manual_seed(20261102)
    values = [
        torch.randn(256, generator=g) * 10.0,          # the normals
        torch.linspace(-0.02, 0.02, 128),               # the subnormals (2^-6 down)
        torch.tensor([0.0, -0.0, 448.0, -448.0, 449.0, -449.0, 1e30, -1e30, float('inf'), float('-inf'),
                      float('nan'), -float('nan')]),
        torch.linspace(-8.0, 8.0, 500),
    ]
    src = torch.cat(values).float().to(DEV)
    out = torch.empty(src.numel(), dtype=torch.uint8, device=DEV)
    encode_bytes[(1,)](src, out, False)

    # torch's own cast: the reference (it saturates and rounds the same way)
    want = src.to(torch.float8_e4m3fn).view(torch.uint8)
    same = out == want
    # the NaN's sign: torch keeps it, the software path writes 0x7F with the sign
    nan = torch.isnan(src)
    same |= nan & (out & 0x7F == 0x7F)
    bad = int((~same).sum())
    print(f'the {src.numel()} values: the mismatches {bad} (the NaNs {int(nan.sum())})')
    if bad:
        for i in range(src.numel()):
            if not bool(same[i]):
                print(f'  the value {src[i].item()}: the software {out[i].item():#04x} the torch {want[i].item():#04x}')
                break

    hw_ok = None
    try:
        hw = torch.empty(src.numel(), dtype=torch.uint8, device=DEV)
        encode_bytes[(1,)](src, hw, True)
        hw_ok = int((hw != want).sum())
        print(f'the hardware encode: the mismatches {hw_ok}')
    except Exception as exc:  # noqa: BLE001 - no float8e4nv below sm_89
        print(f'the hardware encode is not available here: {type(exc).__name__}: {exc}')

    ok = bad == 0 and (hw_ok is None or hw_ok == 0)
    print('OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
