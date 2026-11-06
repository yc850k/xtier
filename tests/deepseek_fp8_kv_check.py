"""The DeepSeek V4.1 FP8 KV cache dequantizer of the sm_75 pack (packs/deepseek_v4/kv_fp8.py): the e4m3 bytes of a
V4.1 paged pool must dequantize bitwise like PyTorch's own e4m3 -> fp32 conversion times the ue8m0 scales, on every
byte value (the hardware conversion is missing below sm_89, so the pack converts the bytes with bit math).

  CUDA_HOME=<nvcc toolkit> python tests/deepseek_fp8_kv_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout  # noqa: E402

from xtier.packs.deepseek_v4 import kv_fp8  # noqa: E402

DEV, PAGE_SIZE, PAGES = 'cuda', 64, 8
LAYOUT = KVLayout.V41
torch.manual_seed(20261030)


def ue8m0_to_fp32(b):
    """The ue8m0 scale bytes -> fp32 (0 is the denormal 2 ** -127, 255 is NaN)."""
    b = b.to(torch.int32)
    normal = (b << 23).view(torch.float32)
    denormal = torch.full(b.shape, 0x00400000, dtype=torch.int32, device=b.device).view(torch.float32)
    scale = torch.where(b == 0, denormal, normal)
    return torch.where(b == 255, torch.tensor(float('nan'), device=b.device), scale)


def build_pool():
    """A paged pool of random e4m3 bytes and ue8m0 scales: every byte value shows up in the data."""
    page_bytes = LAYOUT.page_bytes(PAGE_SIZE)
    pool = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=DEV)
    for page in range(PAGES):
        for token in range(PAGE_SIZE):
            data = token * LAYOUT.data_bytes
            pool[page, data : data + LAYOUT.data_bytes] = torch.randint(
                0, 256, (LAYOUT.data_bytes,), dtype=torch.uint8, device=DEV
            )
            scale = LAYOUT.scale_offset(PAGE_SIZE) + token * LAYOUT.scale_bytes
            pool[page, scale : scale + LAYOUT.scale_bytes] = torch.randint(
                0, 255, (LAYOUT.scale_bytes,), dtype=torch.uint8, device=DEV
            )
    return pool


pool = build_pool()
tokens = torch.arange(PAGE_SIZE * PAGES, dtype=torch.int32, device=DEV)

# the reference: PyTorch's e4m3 -> fp32 times the ue8m0 scales, in the pool's own byte layout
ref = torch.empty(PAGE_SIZE * PAGES, 1, LAYOUT.data_bytes, dtype=torch.bfloat16, device=DEV)
for i in range(PAGE_SIZE * PAGES):
    page, token = i // PAGE_SIZE, i % PAGE_SIZE
    data = pool[page, token * LAYOUT.data_bytes : (token + 1) * LAYOUT.data_bytes]
    scale = pool[
        page,
        LAYOUT.scale_offset(PAGE_SIZE) + token * LAYOUT.scale_bytes : LAYOUT.scale_offset(PAGE_SIZE)
        + (token + 1) * LAYOUT.scale_bytes,
    ]
    vals = data.view(torch.float8_e4m3fn).to(torch.float32)
    vals = vals * ue8m0_to_fp32(scale).repeat_interleave(LAYOUT.tile_size)
    ref[i, 0] = vals.to(torch.bfloat16)

got = kv_fp8.dequantize_k_cache_paged_v41(pool, tokens, PAGE_SIZE)

same = (got.view(torch.int16) == ref.view(torch.int16)) | (torch.isnan(got) & torch.isnan(ref))
bad = int((~same).sum())
print('V41: the tokens:', got.shape[0], 'the values:', got.shape[-1], 'the mismatches:', bad)
if bad:
    rows = (~same).any(dim=-1).nonzero().flatten()
    i = int(rows[0])
    j = int((~same[i, 0]).nonzero()[0])
    print('  the first bad value: the token', i, 'the index', j, 'the kernel', got[i, 0, j].item(),
          'the reference', ref[i, 0, j].item())

# the V4 layout: per token 448 e4m3 nope + 64 bf16 rope, then 7 ue8m0 scales (padded to 8)
V4 = KVLayout.V4
v4_pool = torch.zeros(PAGES, V4.page_bytes(PAGE_SIZE), dtype=torch.uint8, device=DEV)
for page in range(PAGES):
    for token in range(PAGE_SIZE):
        row = token * V4.data_bytes
        v4_pool[page, row : row + V4.data_bytes] = torch.randint(
            0, 256, (V4.data_bytes,), dtype=torch.uint8, device=DEV
        )
        scale = V4.scale_offset(PAGE_SIZE) + token * V4.scale_bytes
        v4_pool[page, scale : scale + V4.scale_bytes] = torch.randint(
            0, 255, (V4.scale_bytes,), dtype=torch.uint8, device=DEV
        )

v4_ref = torch.empty(PAGE_SIZE * PAGES, 1, 512, dtype=torch.bfloat16, device=DEV)
for i in range(PAGE_SIZE * PAGES):
    page, token = i // PAGE_SIZE, i % PAGE_SIZE
    row = token * V4.data_bytes
    data = v4_pool[page, row : row + V4.data_bytes]
    scale = v4_pool[
        page,
        V4.scale_offset(PAGE_SIZE) + token * V4.scale_bytes : V4.scale_offset(PAGE_SIZE)
        + (token + 1) * V4.scale_bytes,
    ]
    nope = data[:448].view(torch.float8_e4m3fn).to(torch.float32)
    nope = nope * ue8m0_to_fp32(scale[:7]).repeat_interleave(64)
    v4_ref[i, 0, :448] = nope.to(torch.bfloat16)
    v4_ref[i, 0, 448:] = data[448:].view(torch.bfloat16)

v4_got = kv_fp8.dequantize_k_cache_paged(v4_pool, tokens, PAGE_SIZE)
v4_same = (v4_got.view(torch.int16) == v4_ref.view(torch.int16)) | (
    torch.isnan(v4_got) & torch.isnan(v4_ref)
)
v4_bad = int((~v4_same).sum())
print('V4: the tokens:', v4_got.shape[0], 'the values:', v4_got.shape[-1], 'the mismatches:', v4_bad)
if v4_bad:
    rows = (~v4_same).any(dim=-1).nonzero().flatten()
    i = int(rows[0])
    j = int((~v4_same[i, 0]).nonzero()[0])
    print('  the first bad value: the token', i, 'the index', j, 'the kernel', v4_got[i, 0, j].item(),
          'the reference', v4_ref[i, 0, j].item())

print('OK' if bad == 0 and v4_bad == 0 else 'FAIL')
sys.exit(0 if bad == 0 and v4_bad == 0 else 1)
