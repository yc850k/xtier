"""What this host's GPU can compute: the compute capability, the dtypes its tensor cores take, and the features
XTier must leave off.

Turing (sm_75) is the oldest GPU XTier runs on. It has FP16 tensor cores with FP32 accumulation, but:

- no BF16 (sm_80): a BF16 dot product is not compiled below sm_80 and cuBLAS has no BF16 GEMM there, so on
  Turing the model's activations are FP16 (``--dtype float16``) and XTier's GPU kernels take FP16 operands. The
  CPU engine keeps its BF16 contract: the activations are converted to BF16 when they are staged for it.
- no FP8 (sm_89): the FP8 paths (the DeepSeek and GLM packs' FP8 weights, the FP8 GEMM tuning configs) are off.
- no TMA (sm_90): the Qwen hyper-connection mix's overlapped kernel stays off (it checks the capability itself).
- no FP4 tensor cores (sm_100) and no FP4 conversion instruction: the Blackwell instruction
  ``cvt.rn.f16x2.e2m1x2`` that the cache's decode kernels use to convert two E2M1 values at a time does not exist
  below sm_100, so the software E2M1 decode (as in the GPU prefill) is used there.

The capability is read once per process. ``XTIER_ARCH`` overrides it (e.g. ``XTIER_ARCH=7.5`` on a host without a
GPU, for the tests); ``XTIER_ARCH=none`` means "no GPU": only the CPU paths run.
"""

import functools
import os

SM75 = (7, 5)     # Turing: FP16 tensor cores, no BF16, no FP8, no TMA, no FP4
SM80 = (8, 0)     # Ampere: BF16 and TF32
SM89 = (8, 9)     # Ada: FP8
SM90 = (9, 0)     # Hopper: TMA
SM100 = (10, 0)   # Blackwell: FP4 tensor cores and the FP4 conversion instruction
SM120 = (12, 0)   # Blackwell (consumer, RTX PRO 5000): the SM120 MXFP8 / MXFP4 CUTLASS runners


def _override():
    """XTIER_ARCH=7.5 (or 'none' for a host without a GPU) -> the capability to assume."""
    spec = os.environ.get('XTIER_ARCH', '').strip().lower()
    if not spec:
        return None
    if spec in ('none', 'cpu'):
        return None
    major, _, minor = spec.partition('.')
    try:
        return int(major), int(minor or 0)
    except ValueError:
        raise ValueError(f'xtier: XTIER_ARCH={spec!r} is not a compute capability (e.g. 7.5)')


def _driver_capability():
    """The compute capability of the device, read through the driver API (no CUDA context: the plugin's hooks are
    installed before the process creates one)."""
    import ctypes
    try:
        lib = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return None
    if lib.cuInit(0) != 0:
        return None
    device = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(device), 0) != 0:
        return None
    major, minor = ctypes.c_int(), ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    if lib.cuDeviceGetAttribute(ctypes.byref(major), 75, device) != 0:
        return None
    if lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, device) != 0:
        return None
    return major.value, minor.value


@functools.lru_cache(maxsize=1)
def capability():
    """The compute capability of the current device, (major, minor). None without a GPU."""
    override = _override()
    if override is not None:
        return override
    if os.environ.get('XTIER_ARCH', '').strip().lower() in ('none', 'cpu'):
        return None
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_initialized():   # the process has a context: ask torch
        return torch.cuda.get_device_capability()
    cap = _driver_capability()
    if cap is not None:
        return cap
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return None


def at_least(major, minor=0) -> bool:
    """Whether the device is at least this compute capability."""
    cap = capability()
    return cap is not None and cap >= (major, minor)


def has_bf16() -> bool:
    """BF16 tensor cores (sm_80)."""
    return at_least(*SM80)


def has_tf32() -> bool:
    """TF32 tensor cores (sm_80): the down projection's default precision."""
    return at_least(*SM80)


def has_fp8() -> bool:
    """FP8 tensor cores (sm_89)."""
    return at_least(*SM89)


def has_tma() -> bool:
    """TMA (cp.async.bulk, sm_90)."""
    return at_least(*SM90)


def has_fp4() -> bool:
    """FP4 tensor cores (sm_100)."""
    return at_least(*SM100)


def has_fp4_convert() -> bool:
    """The FP4 conversion instruction (cvt.rn.f16x2.e2m1x2, sm_100): the cache's decode kernels use it to convert
    two E2M1 values at a time; below it the software decode is used."""
    return at_least(*SM100)


def has_blackwell_mxfp4_moe() -> bool:
    """FlashInfer's CUTLASS MXFP4 fused MoE (the SM120 runner the GPU prefill's FP4 path uses)."""
    return capability() is not None and capability()[0] == SM120[0]


def dot_dtype():
    """The dtype the GPU's tensor cores take for a 16-bit dot product: BF16 from sm_80 on, FP16 below."""
    import torch
    return torch.bfloat16 if has_bf16() else torch.float16


def dtype():
    """The dtype of the activations XTier's GPU kernels compute in. ``XTIER_DTYPE=bf16|fp16|auto`` (default auto:
    the device's dot dtype)."""
    import torch
    spec = os.environ.get('XTIER_DTYPE', 'auto').strip().lower()
    if spec in ('', 'auto'):
        return dot_dtype()
    if spec == 'bf16':
        return torch.bfloat16
    if spec == 'fp16':
        return torch.float16
    raise ValueError(f'xtier: XTIER_DTYPE={spec!r} is not bf16, fp16 or auto')


def check_activation_dtype(dtype):
    """The dtype of the activations: BF16 or FP16. BF16 needs a GPU with BF16 tensor cores (sm_80 and newer); below
    that (Turing, sm_75) the model has to run in FP16. The CPU engine keeps its BF16 contract either way: the
    activations are converted when they are staged for it."""
    import torch
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f'xtier: BF16 or FP16 activations required, got {dtype}')
    if dtype == torch.bfloat16 and not has_bf16():
        raise TypeError(f'xtier: this GPU ({describe()}) has no BF16 tensor cores: run the model with '
                        '--dtype float16')
    return dtype


def describe() -> str:
    """One line about the device, for the logs."""
    cap = capability()
    if cap is None:
        return 'no GPU'
    return f'sm_{cap[0]}{cap[1]}'


def shared_memory_kb():
    """The shared memory a kernel may use per block, in KiB: Turing has 64, the newer cards more. The kernels' tiles
    are sized so that their pipeline fits."""
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        for name in ('shared_memory_per_block_optin', 'shared_memory_per_multiprocessor'):
            value = getattr(props, name, 0)
            if value:
                return value // 1024
    cap = capability()
    return 64 if cap is None or cap[0] <= 7 else 228
