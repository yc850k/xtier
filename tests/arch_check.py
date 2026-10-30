"""The capability gates of the sm_75 adaptation: what the GPU may run, and the dtype the kernels use.

  python tests/arch_check.py

No GPU needed: XTIER_ARCH stands in for the device's compute capability (the same override the tests of the
hardware-specific paths use).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from xtier import arch  # noqa: E402
from xtier.backend.prefill import _down_precision  # noqa: E402
from xtier.packs import supported  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        bad.append(what)


def with_arch(spec):
    """The gates of a device with this compute capability."""
    os.environ['XTIER_ARCH'] = spec
    arch.capability.cache_clear()
    return arch


for spec, bf16, tf32, fp8, tma, fp4, mxfp4 in (('7.5', False, False, False, False, False, False),
                                              ('8.0', True, True, False, False, False, False),
                                              ('8.9', True, True, True, False, False, False),
                                              ('9.0', True, True, True, True, False, False),
                                              ('10.0', True, True, True, True, True, False),
                                              ('12.0', True, True, True, True, True, True)):
    a = with_arch(spec)
    check(a.capability() == tuple(int(x) for x in spec.split('.')), f'{spec}: the capability is read as {spec}')
    check(a.has_bf16() == bf16, f'{spec}: BF16 tensor cores {bf16}')
    check(a.has_tf32() == tf32, f'{spec}: TF32 tensor cores {tf32}')
    check(a.has_fp8() == fp8, f'{spec}: FP8 tensor cores {fp8}')
    check(a.has_tma() == tma, f'{spec}: TMA {tma}')
    check(a.has_fp4() == fp4, f'{spec}: FP4 tensor cores {fp4}')
    check(a.has_fp4_convert() == fp4, f'{spec}: the FP4 conversion instruction {fp4}')
    check(a.has_blackwell_mxfp4_moe() == mxfp4, f'{spec}: the SM120 MXFP4 fused MoE {mxfp4}')

# the dot products follow the device: FP16 below sm_80 (no BF16 tensor cores), BF16 from sm_80 on; the activations
# are the model's dtype (BF16 everywhere: a card without BF16 tensor cores emulates it, and the models whose
# hyper-connection mixes the residual stream need BF16's range)
with_arch('7.5')
check(arch.dot_dtype() == torch.float16, 'sm_75: the dot products are FP16')
check(arch.dtype() == torch.bfloat16, 'sm_75: the activations are BF16')
with_arch('12.0')
check(arch.dot_dtype() == torch.bfloat16, 'sm_120: the dot products are BF16')
check(arch.dtype() == torch.bfloat16, 'sm_120: the activations are BF16')
os.environ['XTIER_DTYPE'] = 'fp16'
check(arch.dtype() == torch.float16, 'XTIER_DTYPE=fp16: the activations are FP16')
del os.environ['XTIER_DTYPE']

# the down projection's precision: TF32 where the GPU has TF32 tensor cores, the 16-bit dot below them
os.environ.pop('XTIER_DOWN_PRECISION', None)
with_arch('12.0')
check(_down_precision() == 'tf32x3', 'sm_120: the down projection is TF32')
with_arch('7.5')
check(_down_precision() == '16bit', 'sm_75: the down projection is the 16-bit dot (no TF32)')
os.environ['XTIER_DOWN_PRECISION'] = 'tf32'
check(_down_precision() == 'tf32', 'XTIER_DOWN_PRECISION wins')
del os.environ['XTIER_DOWN_PRECISION']

# the packs' features stay off on a card that cannot run them
with_arch('7.5')
check(not supported((8, 9), 'the FP8 weights'), 'sm_75: the FP8 features are off')
check(not supported((9, 0), 'the TMA overlap'), 'sm_75: the TMA features are off')
check(not supported((10, 0), 'the FP4 kernels'), 'sm_75: the FP4 features are off')
check(supported((7, 5), 'the FP4 experts on the CPU'), 'sm_75: the CPU-side features are on')
with_arch('12.0')
check(supported((8, 9), 'the FP8 weights'), 'sm_120: the FP8 features are on')

# the activations: BF16 or FP16; BF16 is accepted everywhere (a card without BF16 tensor cores emulates it, and the
# models whose hyper-connection mixes the residual stream need BF16's range)
with_arch('7.5')
check(arch.check_activation_dtype(torch.bfloat16) == torch.bfloat16, 'sm_75: BF16 activations are accepted')
check(arch.check_activation_dtype(torch.float16) == torch.float16, 'sm_75: FP16 activations are accepted')
try:
    arch.check_activation_dtype(torch.float32)
    check(False, 'FP32 activations are refused')
except TypeError:
    check(True, 'FP32 activations are refused')
with_arch('12.0')
check(arch.check_activation_dtype(torch.bfloat16) == torch.bfloat16, 'sm_120: BF16 activations are accepted')

print('ARCH-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
