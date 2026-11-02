"""The E4M3 (FP8) decode of XTier's kernels: the software path against torch's float8_e4m3fn values and, on a GPU
with the conversion (sm_89 and newer), against the hardware path bit for bit.

The software path is what a card without FP8 tensor cores takes (Turing, the RTX 30 series): Triton has no
float8e4nv below sm_89, so the e4m3 bytes are read with bit math (xtier.backend.fp4.e4m3_f32). The checkpoint's
format is untouched: the bytes stay e4m3, only the reading is done in software.

  python tests/fp8_decode_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from xtier.backend.fp4 import e4m3_f32  # noqa: E402

DEV = 'cuda'
torch.manual_seed(20261102)


@triton.jit
def decode_bytes(SRC, DST, HW: tl.constexpr):
    i = tl.arange(0, 256)
    tl.store(DST + i, e4m3_f32(tl.load(SRC + i), HW))


def main():
    src = torch.arange(256, dtype=torch.uint8, device=DEV)
    soft = torch.empty(256, dtype=torch.float32, device=DEV)
    decode_bytes[(1,)](src, soft, False)

    # torch's own e4m3 values: the reference the software path must match
    want = src.view(torch.float8_e4m3fn).float()
    # 0x7F and 0xFF are NaN in e4m3 (the only NaN codes)
    nan_codes = (src & 0x7F) == 0x7F
    same = torch.isclose(soft, want, rtol=0, atol=0) | (torch.isnan(soft) & torch.isnan(want))
    bad = int((~same).sum())
    print(f'the 256 e4m3 bytes: the mismatches {bad} (the NaN codes {int(nan_codes.sum())})')
    if bad:
        for i in range(256):
            if not bool(same[i]):
                print(f'  the code {src[i].item():#04x}: the software {soft[i].item()} the torch {want[i].item()}')
                break

    # the hardware path, where the card has it: the same bytes, bit for bit
    hw_ok = None
    try:
        hw = torch.empty(256, dtype=torch.float32, device=DEV)
        decode_bytes[(1,)](src, hw, True)
        same_hw = torch.isclose(hw, want, rtol=0, atol=0) | (torch.isnan(hw) & torch.isnan(want))
        hw_ok = int((~same_hw).sum())
        print(f'the hardware decode: the mismatches {hw_ok}')
    except Exception as exc:  # noqa: BLE001 - no float8e4nv below sm_89
        print(f'the hardware decode is not available here: {type(exc).__name__}: {exc}')

    ok = bad == 0 and (hw_ok is None or hw_ok == 0)
    print('OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
