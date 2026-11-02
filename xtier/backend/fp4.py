"""The FP4 formats' decode, in the GPU kernels.

The packed experts hold E2M1 values (4 bits) with block scales: E4M3 (NVFP4, one scale per 16 values) or UE8M0
(MXFP4, one per 32). The GPUs that have the conversion instructions do it in hardware:

- ``cvt.rn.f16x2.e2m1x2`` converts two E2M1 values at a time (sm_100 and newer);
- the E4M3 scales are converted by the FP8 instructions (sm_89 and newer).

Below those levels the values are decoded in software, which gives the same numbers (the E2M1 values are exactly
representable in FP16, the E4M3 scales are a power of two times 1 + m/8): the GPU prefill already decoded the packed
weights in software, and the cache's decode kernels now do the same on a Turing card (sm_75).
"""

import triton
import triton.language as tl


@triton.jit
def e2m1_f16(code):
    """E2M1 nibble (the sign in bit 3) -> FP16: the software form of ``cvt.rn.f16x2.e2m1x2``."""
    m = code & 7
    v = tl.where(m < 4, m.to(tl.float32) * 0.5, tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))))
    return (v * tl.where((code & 8) != 0, -1.0, 1.0)).to(tl.float16)


@triton.jit
def e4m3_f32(code, HW: tl.constexpr = True):
    """The E4M3 block scales as FP32: the hardware conversion where the GPU has FP8 (sm_89 and newer), the formula
    below it. E4M3FN: the sign in bit 7, the exponent in bits 6..3 (bias 7), the mantissa in bits 2..0; the exponent 0
    is subnormal (m * 2^-9), the exponent 15 with the mantissa 7 is NaN."""
    if HW:
        return code.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    e = ((code >> 3) & 15).to(tl.int32)     # the exponent, signed: e - 7 goes below zero
    m = (code & 7).to(tl.float32)
    value = tl.where(e == 0, m * (2.0 ** -9), (1.0 + m / 8.0) * tl.exp2((e - 7).to(tl.float32)))
    value = tl.where((e == 15) & (m == 7.0), float('nan'), value)
    return value * tl.where((code >> 7) != 0, -1.0, 1.0)


@triton.jit
def ue8m0_f32(code):
    """The UE8M0 block scales as FP32: the exponent in the byte (2^(e-127); the byte 0 is 2^-127)."""
    return tl.where(code == 0, 2.0 ** -127, (code.to(tl.uint32) << 23).to(tl.float32, bitcast=True))


@triton.jit
def e4m3_bits(x, HW: tl.constexpr = True):
    """FP32 -> the E4M3 byte, as ``cvt.rn.satfinite.e4m3x2.f32`` gives it: round to nearest even, saturating to
    +-448, NaN to 0x7F with its sign. From 2^-6 up the FP32 bits are rounded at mantissa bit 20 (a carry moves into
    the exponent); below, the value is |x| * 2^9 subnormal steps rounded to the nearest even (8 steps are the
    smallest normal, the code 0x08).

    The hardware cast needs sm_89 (Triton has no float8e4nv below it), so the software path is the one a Turing card
    takes when a kernel stores FP8 (the KV cache's e4m3 bytes).
    """
    if HW:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = bits & 0x7FFFFFFF
    r = a + 0x7FFFF + ((a >> 20) & 1)
    normal = (((r >> 23) - 120) << 3) | ((r >> 20) & 7)
    t = a.to(tl.float32, bitcast=True) * 512.0
    lo = tl.floor(t)
    steps = lo.to(tl.int32)
    half = t - lo
    steps += ((half > 0.5) | ((half == 0.5) & ((steps & 1) == 1))).to(tl.int32)
    code = tl.where(a < 0x3C800000, steps, tl.minimum(normal, 0x7E))
    code = tl.where(a > 0x7F800000, 0x7F, code)
    return (code | sign).to(tl.uint8)
