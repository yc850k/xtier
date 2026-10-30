"""The DeepSeek V4.1 FP8 KV cache on sm_75 (Turing).

The K cache of the V4.1 layout is 512 e4m3 bytes per token plus 16 ue8m0
scales. SGLang dequantizes it with a Triton kernel that loads the bytes through
an ``fp8`` pointer: the hardware conversion needs sm_89, so Triton refuses the
type on Turing ("type fp8e4nv not supported in this architecture"). SGLang also
refuses the whole dequantizer below sm_100 ("DeepSeek V4.1 KV cache
dequantization requires CUDA SM100 or newer").

This module replaces the kernel with one that loads the bytes as uint8 and
converts them with bit math (the same values, no hardware fp8), and it drops
the SM100 gate. The FP4 layout keeps SGLang's kernel: it needs sm_100 anyway.
"""

import logging

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    NUM_SCALE_TILES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
    _ue8m0_to_fp32,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

logger = logging.getLogger(__name__)


@triton.jit
def _e4m3_to_fp32(bits):
    """The e4m3 bytes -> fp32 (the bit math of the hardware conversion).

    e4m3 is 1 sign, 4 exponent (bias 7), 3 mantissa; fp32 is 1 sign, 8
    exponent (bias 127), 23 mantissa. The normal values map to the exponent
    ``e + 120`` and the mantissa ``m << 20``; the subnormals (``e == 0``) are
    ``m * 2 ** -9`` and need the exponent they land in.
    """
    bits = bits.to(tl.int32)
    sign = (bits & 0x80) << 24
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    normal = sign | ((exp + 120) << 23) | (mant << 20)
    subnormal = tl.where(
        mant == 1,
        sign | (118 << 23),
        tl.where(
            mant == 2,
            sign | (119 << 23),
            tl.where(
                mant == 3,
                sign | (119 << 23) | (1 << 22),
                sign | (120 << 23) | ((mant - 4) << 21),
            ),
        ),
    )
    return tl.where(
        (exp == 15) & (mant == 7),  # the only NaN of e4m3 (0x7f and 0xff)
        float("nan"),
        tl.where(exp == 0, tl.where(mant == 0, sign, subnormal), normal).to(
            tl.float32, bitcast=True
        ),
    )


@triton.jit
def _dequant_k_cache_v41_fp8_kernel(
    output_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # V41: 512 e4m3 values per token, then 16 ue8m0 scales (one per 32 values).
    tl.static_assert(DATA_BYTES == 512 and SCALE_BYTES == 16 and TILE_SIZE == 32)
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * DATA_BYTES
    token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * SCALE_BYTES

    offs = tl.arange(0, DATA_BYTES)
    vals = _e4m3_to_fp32(tl.load(buf_uint8_ptr + token_data_base + offs))
    scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + offs // TILE_SIZE)
    out = vals * _ue8m0_to_fp32(scale_u8)
    tl.store(output_ptr + token_id * output_stride_0 + offs, out.to(tl.bfloat16))


def dequantize_k_cache_paged_v41(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: torch.Tensor = None,
    layout=KVLayout.V41,
):
    """The V4.1 dequantizer for sm_75: the bytes in, the bf16 K values out."""
    layout = KVLayout.parse(layout)
    assert layout is KVLayout.V41, layout
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    assert bytes_per_page >= page_size * layout.bytes_per_token, (
        f"{bytes_per_page=} cannot hold {page_size} tokens of {layout}"
    )
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, layout.data_bytes),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, layout.data_bytes)
        assert out.dtype == torch.bfloat16
    if num_tokens == 0:
        return out

    _dequant_k_cache_v41_fp8_kernel[(num_tokens,)](
        out,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DATA_BYTES=layout.data_bytes,
        SCALE_BYTES=layout.scale_bytes,
        TILE_SIZE=layout.tile_size,
        S_OFFSET_BYTES=layout.scale_offset(page_size),
    )
    return out


@triton.jit
def _dequant_k_cache_v4_kernel(
    output_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # V4: per token 448 e4m3 nope + 64 bf16 rope (576 contiguous bytes) + 7 ue8m0
    # scales padded to 8 bytes; the scales sit after the page's token rows.
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = _e4m3_to_fp32(tl.load(buf_uint8_ptr + fp8_off))
        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id)
        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * _ue8m0_to_fp32(scale_u8)).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = token_data_base + DIM_NOPE + rope_offs * 2
    lo = tl.load(buf_uint8_ptr + bf16_off).to(tl.int32)
    hi = tl.load(buf_uint8_ptr + bf16_off + 1).to(tl.int32)
    rope_bits = (lo | (hi << 8)).to(tl.uint16)
    tl.store(
        output_ptr + out_row_base + DIM_NOPE + rope_offs,
        rope_bits.to(tl.bfloat16, bitcast=True),
    )


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: torch.Tensor = None,
    layout=KVLayout.V4,
):
    """The V4 dequantizer for sm_75: the bytes in, the bf16 K values out."""
    layout = KVLayout.parse(layout)
    if layout is KVLayout.V41:
        return dequantize_k_cache_paged_v41(
            quant_k_cache, page_table_1_flattened, page_size, out=out, layout=layout
        )
    assert layout is KVLayout.V4, layout
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16
    if num_tokens == 0:
        return out

    _dequant_k_cache_v4_kernel[(num_tokens,)](
        out,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=page_size * NOPE_ROPE_BYTES,
    )
    return out


def install():
    """Replace SGLang's dequantizers with the sm_75 ones."""
    from sglang.kernels.ops.attention.dsv4 import dequant_k_cache

    if getattr(dequant_k_cache, '_xtier_sm75', False):
        return False
    dequant_k_cache._xtier_sm75 = True
    dequant_k_cache.dequantize_k_cache_paged_v41 = dequantize_k_cache_paged_v41
    dequant_k_cache.dequantize_k_cache_paged = dequantize_k_cache_paged
    logger.info(
        'xtier: the DeepSeek V4.1 FP8 KV cache runs on sm_75 '
        '(the e4m3 bytes are converted with bit math)'
    )
    return True
