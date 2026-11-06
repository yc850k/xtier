"""The DeepSeek V4.1 attention on Turing: a Triton sparse MLA over SGLang's packed FP8 KV pages, in place of FlashMLA.

Upstream's approach for the cards below SM90 (RTX 30 / 40 series, PR #10) is taken over from there: SGLang serves
every DeepSeek V4 attention call through FlashMLA (the sgl-flash-mla wheel), which is built for SM90 / SM100 only, so
the call is hooked and a Triton sparse MLA computes the same attention over the same pages. The upstream's kernel
(xtier/packs/deepseek_v4/sparse_mla.py in origin/main) is the one here, with the three changes Turing needs:

- the dot products run in FP16, not BF16: Turing has FP16 tensor cores and no BF16 ones, so a BF16 dot is emulated
  (the pack's own dtype is FP16 on this card, so the values are the ones the model computes anyway);
- the tiles are smaller: Turing's SM has 64 KiB of shared memory, the upstream's tiling is sized for Hopper's 99 KiB;
- the FP8 values are decoded with the branch's own software codec (xtier/backend/fp4.py), exactly as the upstream does
  on the cards without the FP8 conversion instruction.

The kernel reads both caches of a row (the window, then the compressed keys) into one online softmax: the FP8 values
decoded exactly and scaled, Q.K on the tensor cores with FP32 accumulation, the softmax scale on the FP32 scores, P in
FP16 for the PV product, FP32 accumulation, the sink in the FP32 normalization. Decode-sized calls split a row's keys
over programs and merge the partial results in FP32.

The hook is registered on the symbol SGLang calls: ``sgl_kernel.flash_mla.flash_mla_with_kvcache`` (the local import in
the DSV4 backend). The upstream's symbol (``flash_mla_interface``) is hooked too when that module exists, so a newer
SGLang takes the same path.
"""

import functools
import logging

import torch
import triton
import triton.language as tl

from xtier import arch
from xtier.backend.fp4 import e4m3_f32

logger = logging.getLogger(__name__)

NOPE, DIM = 448, 512             # FP8 values per token; with the 64 rope values, the head dim of Q, K and V
DATA_BYTES, SCALE_BYTES = 576, 8  # a token's data row (448 + 2 x 64 bytes) and scale row (7 UE8M0 scales + 1 pad)
BLOCK_N = 16                     # keys per iteration: Turing's SM has 64 KiB of shared memory, Hopper's 99 KiB
HEADS_PER_PROGRAM = 16           # heads per program (the upstream's 32 needs Hopper's shared memory)
NUM_WARPS = 4
KEYS_PER_SPLIT = 128             # decode-sized calls: keys of a row per program
LOG2E = 1.4426950408889634


@triton.jit
def _keys(C, CB, IDX, LEN, row, start, topk, page_size, page_bytes, stride, n, d, HAS_LEN: tl.constexpr):
    """Keys at positions start + n of a row's index list as an FP16 tile [BLOCK_N, 512], and which are valid."""
    pos = start + n
    idx = tl.load(IDX + row * stride + pos, mask=pos < topk, other=-1)
    valid = idx >= 0
    if HAS_LEN:
        valid = valid & (pos < tl.load(LEN + row))
    idx = tl.where(valid, idx, 0).to(tl.int64)
    page = idx // page_size
    slot = idx % page_size
    data = page * page_bytes + slot * 576
    nope = d < 448
    fp8 = valid[:, None] & nope[None, :]
    b = tl.load(C + data[:, None] + d[None, :], mask=fp8, other=0)
    e = tl.load(C + (page * page_bytes + page_size * 576 + slot * 8)[:, None] + (d // 64)[None, :], mask=fp8,
                other=127).to(tl.int32)
    scale = tl.where(e == 0, 0x00400000, e << 23).to(tl.float32, bitcast=True)   # UE8M0: 2^(e - 127)
    rope = tl.load(CB + (data[:, None] + 448) // 2 + (d[None, :] - 448), mask=valid[:, None] & (d >= 448)[None, :],
                   other=0.0)
    k = tl.where(nope[None, :], e4m3_f32(b, HW=False) * scale, rope.to(tl.float32))
    return k.to(tl.float16), valid


@triton.jit
def _step(q, k, valid, m_i, l_i, acc, scale):
    s = tl.where(valid[None, :], tl.dot(q, tl.trans(k)) * scale, float('-inf'))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_use = tl.where(m_new == float('-inf'), 0.0, m_new)   # no valid key yet: every weight stays 0
    alpha = tl.math.exp2(m_i - m_use)
    p = tl.math.exp2(s - m_use[:, None])
    acc = tl.dot(p.to(tl.float16), k, acc * alpha[:, None])
    return m_new, l_i * alpha + tl.sum(p, 1), acc


@triton.jit
def _finish(m_i, l_i, acc, SINK, O, LSE, row, h, hmask, d, SO_B, SO_H, H: tl.constexpr, HAS_SINK: tl.constexpr):
    """out = acc / (l + 2^(sink * log2 e - m)); a row without valid keys writes 0. LSE (natural log) of the keys."""
    m_use = tl.where(m_i == float('-inf'), 0.0, m_i)
    total = l_i
    if HAS_SINK:
        sink = tl.load(SINK + h, mask=hmask, other=0.0) * 1.4426950408889634
        total = total + tl.math.exp2(sink - m_use)
    out = tl.where(l_i[:, None] > 0, acc / total[:, None], 0.0)
    tl.store(O + row * SO_B + h[:, None] * SO_H + d[None, :], out.to(O.dtype.element_ty), mask=hmask[:, None])
    lse = tl.where(l_i > 0, (m_use + tl.math.log2(tl.where(l_i > 0, l_i, 1.0))) * 0.6931471805599453,
                   float('-inf'))
    tl.store(LSE + row * H + h, lse, mask=hmask)


@triton.jit
def _attend(Q, C1, CB1, I1, N1, C2, CB2, I2, N2, SINK, O, LSE, M, L, ACC, scale, SQ_B, SQ_H, SO_B, SO_H, SI1, SI2,
            page_size1, page_bytes1, page_size2, page_bytes2, topk1, topk2, keys,
            H: tl.constexpr, HB: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr, HAS_N1: tl.constexpr,
            HAS_N2: tl.constexpr, EXTRA: tl.constexpr, HAS_SINK: tl.constexpr):
    """One query row, HB heads and key range [split * keys, + keys) of the row's window keys followed by its
    compressed keys: the output (SPLITS == 1) or the running max M, sum L and unnormalized output ACC."""
    row, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = hb * HB + tl.arange(0, HB)
    hmask = h < H
    d = tl.arange(0, 512)
    n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + row * SQ_B + h[:, None] * SQ_H + d[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
    m_i = tl.full([HB], float('-inf'), tl.float32)
    l_i = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, 512], tl.float32)
    lo = split * keys
    hi = lo + keys
    end1 = topk1
    if HAS_N1:
        end1 = tl.minimum(end1, tl.load(N1 + row))
    for start in range(lo, tl.minimum(hi, end1), BLOCK_N):
        k, valid = _keys(C1, CB1, I1, N1, row, start, topk1, page_size1, page_bytes1, SI1, n, d, HAS_N1)
        m_i, l_i, acc = _step(q, k, valid, m_i, l_i, acc, scale)
    if EXTRA:
        end2 = topk2
        if HAS_N2:
            end2 = tl.minimum(end2, tl.load(N2 + row))
        for start in range(tl.maximum(lo, topk1), tl.minimum(hi, topk1 + end2), BLOCK_N):
            k, valid = _keys(C2, CB2, I2, N2, row, start - topk1, topk2, page_size2, page_bytes2, SI2, n, d, HAS_N2)
            m_i, l_i, acc = _step(q, k, valid, m_i, l_i, acc, scale)
    if SPLITS == 1:
        _finish(m_i, l_i, acc, SINK, O, LSE, row, h, hmask, d, SO_B, SO_H, H, HAS_SINK)
    else:
        at = (row * SPLITS + split) * H + h
        tl.store(M + at, m_i, mask=hmask)
        tl.store(L + at, l_i, mask=hmask)
        tl.store(ACC + at[:, None] * 512 + d[None, :], acc, mask=hmask[:, None])


@triton.jit
def _merge(M, L, ACC, SINK, O, LSE, SO_B, SO_H, H: tl.constexpr, HB: tl.constexpr, SPLITS: tl.constexpr,
           HAS_SINK: tl.constexpr):
    """A row's key ranges merged in FP32: weights 2^(m - max m), then as one pass."""
    row, hb = tl.program_id(0), tl.program_id(1)
    h = hb * HB + tl.arange(0, HB)
    hmask = h < H
    d = tl.arange(0, 512)
    top = tl.full([HB], float('-inf'), tl.float32)
    for s in tl.static_range(SPLITS):
        top = tl.maximum(top, tl.load(M + (row * SPLITS + s) * H + h, mask=hmask, other=float('-inf')))
    top_use = tl.where(top == float('-inf'), 0.0, top)
    total = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, 512], tl.float32)
    for s in tl.static_range(SPLITS):
        at = (row * SPLITS + s) * H + h
        w = tl.math.exp2(tl.load(M + at, mask=hmask, other=float('-inf')) - top_use)
        total += w * tl.load(L + at, mask=hmask, other=0.0)
        acc += w[:, None] * tl.load(ACC + at[:, None] * 512 + d[None, :], mask=hmask[:, None], other=0.0)
    _finish(top, total, acc, SINK, O, LSE, row, h, hmask, d, SO_B, SO_H, H, HAS_SINK)


@functools.lru_cache
def _sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _views(cache):
    """A paged cache [pages, page_size, 1, 584] as flat uint8 and BF16 tensors, its page size and page stride."""
    raw = cache.view(torch.uint8)
    flat = raw.as_strided((raw.shape[0] * raw.stride(0),), (1,))
    return flat, flat.view(torch.bfloat16), raw.shape[1], raw.stride(0)


def _rows(indices, rows):
    indices = indices.reshape(rows, -1)
    return indices if indices.stride(-1) == 1 else indices.contiguous()


def sparse_mla(q, k_cache, indices, topk_length, attn_sink, softmax_scale, extra_k_cache=None, extra_indices=None,
               extra_topk_length=None):
    """FlashMLA's sparse FP8 decode on the V4 KV layout: q [rows, 1, heads, 512], indices [rows, 1, topk] slots
    (-1: none) into k_cache [pages, page_size, 1, 584], optional extra_* into a second cache, attn_sink [heads] FP32.
    Returns (out [rows, 1, heads, 512], lse [rows, heads, 1] of the keys, natural log)."""
    rows, s_q, heads, dim = q.shape
    if s_q != 1 or dim != DIM:
        raise NotImplementedError(f'xtier: sparse MLA takes one query per row and head dim {DIM}, not {tuple(q.shape)}')
    q = q.view(rows, heads, dim) if q.is_contiguous() else q.reshape(rows, heads, dim)
    if q.stride(-1) != 1:
        q = q.contiguous()
    out = torch.empty((rows, heads, dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    if rows == 0:
        return out.unsqueeze(1), lse.unsqueeze(-1)
    c1, cb1, ps1, pb1 = _views(k_cache)
    i1 = _rows(indices, rows)
    extra = extra_k_cache is not None and extra_indices is not None
    if extra:
        c2, cb2, ps2, pb2 = _views(extra_k_cache)
        i2 = _rows(extra_indices, rows)
    else:
        c2, cb2, ps2, pb2, i2 = c1, cb1, ps1, pb1, i1
    topk1, topk2 = i1.shape[1], (i2.shape[1] if extra else 0)
    hb = HEADS_PER_PROGRAM
    programs = rows * triton.cdiv(heads, hb)
    keys = triton.cdiv(topk1 + topk2, 64) * 64
    target = 2 * _sms(q.device.index)
    if programs < target:   # decode-sized: split the keys over more programs
        keys = max(KEYS_PER_SPLIT, triton.cdiv(triton.cdiv(topk1 + topk2, triton.cdiv(target, programs)), 64) * 64)
    splits = triton.cdiv(topk1 + topk2, keys)
    if splits > 1:
        m = torch.empty((rows, splits, heads), dtype=torch.float32, device=q.device)
        l = torch.empty_like(m)
        acc = torch.empty((rows, splits, heads, DIM), dtype=torch.float32, device=q.device)
    else:
        m = l = acc = lse
    n1 = topk_length if topk_length is not None else i1
    n2 = extra_topk_length if (extra and extra_topk_length is not None) else i1
    sink = attn_sink if attn_sink is not None else lse
    _attend[(rows, triton.cdiv(heads, hb), splits)](
        q, c1, cb1, i1, n1, c2, cb2, i2, n2, sink, out, lse, m, l, acc, softmax_scale * LOG2E,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1), i1.stride(0), i2.stride(0),
        ps1, pb1, ps2, pb2, topk1, topk2, keys,
        H=heads, HB=hb, BLOCK_N=BLOCK_N, SPLITS=splits, HAS_N1=topk_length is not None,
        HAS_N2=extra and extra_topk_length is not None, EXTRA=extra, HAS_SINK=attn_sink is not None,
        num_warps=NUM_WARPS, num_stages=1)
    if splits > 1:
        _merge[(rows, triton.cdiv(heads, hb))](m, l, acc, sink, out, lse, out.stride(0), out.stride(1), H=heads, HB=hb,
                                               SPLITS=splits, HAS_SINK=attn_sink is not None, num_warps=4)
    return out.unsqueeze(1), lse.unsqueeze(-1)


def flash_mla_with_kvcache(original, *args, **kwargs):
    """AROUND hook of flash_mla_with_kvcache (registered below SM90): the DeepSeek V4 backend's sparse FP8 calls on
    the V4 layout, in the Triton sparse MLA."""
    cache = kwargs.get('k_cache')
    if (args or kwargs.get('indices') is None or not kwargs.get('is_fp8_kvcache') or cache is None
            or cache.shape[-1] != DATA_BYTES + SCALE_BYTES or kwargs.get('head_dim_v', DIM) != DIM):
        return original(*args, **kwargs)
    scale = kwargs.get('softmax_scale')
    q = kwargs['q']
    return sparse_mla(q, cache, kwargs['indices'], kwargs.get('topk_length'), kwargs.get('attn_sink'),
                      scale if scale is not None else q.shape[-1] ** -0.5, kwargs.get('extra_k_cache'),
                      kwargs.get('extra_indices_in_kvcache'), kwargs.get('extra_topk_length'))


def no_sparse_prefill(original, self, *args, **kwargs):
    """AROUND hook of DeepseekV4AttnBackend._use_sparse_prefill (registered below SM90): prefill stays on the paged
    caches, as on SM120, instead of FlashMLA's sparse prefill over a BF16 workspace."""
    return False


def shared_memory_ok() -> bool:
    """Whether the tiles fit the card's shared memory: Turing's SM has 64 KiB, the newer cards 228 KiB."""
    need = (BLOCK_N * DIM + HEADS_PER_PROGRAM * DIM) * 2
    return need <= arch.shared_memory_kb() * 1024
