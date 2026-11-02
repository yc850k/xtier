"""The Triton sparse MLA (the sm_75 attention path) against a plain reference.

The KV cache holds the packed V4 layout: per token 448 e4m3 nope + 64 bf16 rope (576 bytes) and, after the
page's token rows, 7 ue8m0 scales (8 bytes per token). The kernel reads the pages in place, decodes the e4m3
values in software (the card has no FP8 conversion instruction) and computes the attention on the FP16 tensor
cores; the reference gathers the same rows in PyTorch and does the softmax in FP32.

The scales are the ones a quantizer picks (a power of two that puts the block's values in the e4m3 range), and the
random bytes avoid the e4m3 NaN patterns: a NaN in a key makes the whole row's online softmax NaN, which says
nothing about the kernel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout  # noqa: E402

from xtier.packs.deepseek_v4 import sparse_mla  # noqa: E402

DEV, PAGE_SIZE, PAGES, HEADS = 'cuda', 64, 4, 64
V4 = KVLayout.V4
torch.manual_seed(20261102)


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
            nope = torch.randint(0, 256, (448,), dtype=torch.uint8, device=DEV)
            nope[nope == 0x7F] = 0x01            # the e4m3 NaN patterns
            nope[nope == 0xFF] = 0xFE
            pool[page, row : row + 448] = nope
            rope = torch.randn((V4.data_bytes - 448) // 2, device=DEV).to(torch.bfloat16)
            pool[page, row + 448 : row + V4.data_bytes] = rope.view(torch.uint8)
            scale = V4.scale_offset(PAGE_SIZE) + token * V4.scale_bytes
            pool[page, scale : scale + V4.scale_bytes] = torch.full(
                (V4.scale_bytes,), 119, dtype=torch.uint8, device=DEV
            )   # 2^(119-127) = 2^-8: the e4m3 values land around +-1
    return pool


pool = build_cache()
page_bytes = pool.shape[1]
cache = pool[:, : PAGE_SIZE * V4.bytes_per_token].view(PAGES, PAGE_SIZE, 1, V4.bytes_per_token)

B, TOPK = 2, 64
q = torch.randn(B, 1, HEADS, 512, device=DEV, dtype=torch.float16)
indices = torch.randint(0, PAGE_SIZE * PAGES, (B, 1, TOPK), dtype=torch.int32, device=DEV)
indices[0, 0, -3:] = -1                      # the invalid slots
topk_length = torch.tensor([TOPK - 3, TOPK], dtype=torch.int32, device=DEV)
attn_sink = torch.randn(HEADS, device=DEV, dtype=torch.float32)
scale = 512 ** -0.5

out, lse = sparse_mla.sparse_mla(q, cache, indices, topk_length, attn_sink, scale)
torch.cuda.synchronize()

# the reference: the same rows, gathered and dequantized by hand
valid = indices >= 0
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
lse_ref = torch.logsumexp(scores, dim=-1)      # the keys only: SGLang's convention for the lse it returns
lse_norm = torch.logsumexp(torch.stack([lse_ref, attn_sink.view(1, 1, HEADS).expand_as(lse_ref)]), dim=0)
weights = torch.exp(scores - lse_norm.unsqueeze(-1))
out_ref = torch.einsum('bsht,bstv->bshv', weights, kv)

err = (out.float() - out_ref).abs()
finite = torch.isfinite(err)
err = err[finite].max().item() if finite.any() else 0.0
nan_mismatch = int((torch.isnan(out) ^ torch.isnan(out_ref)).sum())
lse_err = (lse.reshape(B, HEADS) - lse_ref.reshape(B, HEADS)).abs()
lse_finite = torch.isfinite(lse_err)
scale_err = lse_err[lse_finite].max().item() if lse_finite.any() else 0.0
lse_nan_mismatch = int((torch.isnan(lse) ^ torch.isnan(lse_ref)).sum())
print('the output shape:', tuple(out.shape), 'the max error:', err, 'the lse error:', scale_err,
      'the NaN mismatches:', nan_mismatch, lse_nan_mismatch)
ok = err < 0.05 and scale_err < 0.05 and nan_mismatch == 0 and lse_nan_mismatch == 0
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
