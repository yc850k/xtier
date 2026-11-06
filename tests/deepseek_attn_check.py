"""The portable DeepSeek V4.1 attention (the sm_75 path): the gather-and-dequantize over a paged V4 KV cache
against a plain reference.

The KV cache holds the packed V4 layout: per token 448 e4m3 nope + 64 bf16 rope (576 bytes) and, after the
page's token rows, 7 ue8m0 scales (8 bytes per token).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout  # noqa: E402
from sglang.kernels.ops.attention.flash_mla_sm120 import _sm120_sparse_decode_fwd  # noqa: E402

DEV, PAGE_SIZE, PAGES, HEADS = 'cuda', 64, 4, 64
V4 = KVLayout.V4
torch.manual_seed(20261030)


def ue8m0_to_fp32(b):
    normal = (b.to(torch.int32) << 23).view(torch.float32)
    denormal = torch.full(b.shape, 0x00400000, dtype=torch.int32, device=b.device).view(torch.float32)
    return torch.where(b == 0, denormal, normal)


def build_cache():
    """A paged cache of random bytes, the layout the kernel reads."""
    pool = torch.zeros(PAGES, V4.page_bytes(PAGE_SIZE), dtype=torch.uint8, device=DEV)
    for page in range(PAGES):
        for token in range(PAGE_SIZE):
            row = token * V4.data_bytes
            pool[page, row : row + V4.data_bytes] = torch.randint(
                0, 256, (V4.data_bytes,), dtype=torch.uint8, device=DEV
            )
            scale = V4.scale_offset(PAGE_SIZE) + token * V4.scale_bytes
            pool[page, scale : scale + V4.scale_bytes] = torch.randint(
                0, 255, (V4.scale_bytes,), dtype=torch.uint8, device=DEV
            )
    return pool


pool = build_cache()
page_bytes = pool.shape[1]
# the backend's view: the padded pages sliced to the tokens' bytes (the page stride stays the padded one)
cache = pool[:, : PAGE_SIZE * V4.bytes_per_token].view(PAGES, PAGE_SIZE, 1, V4.bytes_per_token)

# the queries: (B, 1, H, D) bf16
B, TOPK = 1, 64
q = torch.randn(B, 1, HEADS, 512, device=DEV, dtype=torch.bfloat16)
indices = torch.randint(0, PAGE_SIZE * PAGES, (B, 1, TOPK), dtype=torch.int32, device=DEV)
indices[0, 0, -3:] = -1                      # the invalid slots
topk_length = torch.tensor([TOPK - 3], dtype=torch.int32, device=DEV)
attn_sink = torch.randn(HEADS, device=DEV, dtype=torch.float32)
scale = 512 ** -0.5

out, lse = _sm120_sparse_decode_fwd(q, cache, indices, topk_length, attn_sink, 512, scale)
torch.cuda.synchronize()

# the reference: the same rows, gathered and dequantized by hand
valid = indices >= 0
flat = indices.reshape(-1).long()
rows = []
for i in range(B):
    for t in range(TOPK):
        slot = int(indices[i, 0, t])
        if slot < 0:
            rows.append(torch.zeros(512, device=DEV))
            continue
        page, token = slot // PAGE_SIZE, slot % PAGE_SIZE
        data = pool[page, token * V4.data_bytes : (token + 1) * V4.data_bytes]
        scales = pool[
            page,
            V4.scale_offset(PAGE_SIZE) + token * V4.scale_bytes : V4.scale_offset(PAGE_SIZE)
            + (token + 1) * V4.scale_bytes,
        ]
        nope = data[:448].view(torch.float8_e4m3fn).to(torch.float32)
        nope = nope * ue8m0_to_fp32(scales[:7]).repeat_interleave(64)
        row = torch.empty(512, device=DEV)
        row[:448] = nope
        row[448:] = data[448:].view(torch.bfloat16).float()
        rows.append(row)
kv = torch.stack(rows).view(B, 1, TOPK, 512)

scores = torch.einsum('bshd,bstd->bsht', q.float(), kv) * scale
mask = (~valid).unsqueeze(2).expand_as(scores)
scores = scores.masked_fill(mask, float('-inf'))
lse_ref = torch.logsumexp(scores, dim=-1)
lse_ref = torch.logsumexp(torch.stack([lse_ref, attn_sink.view(1, 1, HEADS).expand_as(lse_ref)]), dim=0)
weights = torch.exp(scores - lse_ref.unsqueeze(-1))
out_ref = torch.einsum('bsht,bstv->bshv', weights, kv)
out_ref = out_ref.to(torch.bfloat16)

err = (out.float() - out_ref.float()).abs()
# the random bytes hold the fp8 NaN patterns (0x7f, 0xff): those positions are NaN in both
finite = torch.isfinite(err)
err = err[finite].max().item() if finite.any() else 0.0
nan_mismatch = int((torch.isnan(out) ^ torch.isnan(out_ref)).sum())
lse_err = (lse - lse_ref).abs()
lse_finite = torch.isfinite(lse_err)
scale_err = lse_err[lse_finite].max().item() if lse_finite.any() else 0.0
lse_nan_mismatch = int((torch.isnan(lse) ^ torch.isnan(lse_ref)).sum())
print('the output shape:', tuple(out.shape), 'the max error:', err, 'the lse error:', scale_err,
      'the NaN mismatches:', nan_mismatch, lse_nan_mismatch)
ok = err < 0.05 and scale_err < 0.05 and nan_mismatch == 0 and lse_nan_mismatch == 0
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
