"""The DeepSeek V4.1 FP4 fake quant of the compressed KV on sm_75 (Turing).

SGLang fuses the tail RoPE and the NVFP4 fake quant of the compressed KV into one Triton kernel. The kernel casts
the block scale to NVFP4's e4m3, and Triton emits that cast only on sm_89 and newer ("type fp8e4nv not supported in
this architecture"); on a Turing card the kernel fails to compile when the first request compresses a KV group.

The pack points the call at SGLang's own torch path (the tail RoPE and the FP4 fake quant in PyTorch, the numbers the
fused kernel computes).
"""

import logging

logger = logging.getLogger(__name__)

ROPE_FQ4 = 'sglang.srt.layers.attention.dsv4.dsv41_sparse._rope_fq4'


def _rope_fq4_torch(x, freqs, rope_dim, *, compressed_kv=False, positions=None):
    """The tail RoPE and the FP4 fake quant, in PyTorch."""
    from sglang.kernels.ops.attention.dsv4.torch_quant import (
        fake_quant_compressed_kv,
        fake_quant_fp4,
    )
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

    if positions is not None:
        freqs = freqs[positions]
    quant = fake_quant_compressed_kv if compressed_kv else fake_quant_fp4
    return quant(rope_tail(x, freqs, rope_dim))


def install(registry, hook_type):
    """Replace the fused RoPE and FP4 kernel with the torch path."""
    registry.register(ROPE_FQ4, _rope_fq4_torch, hook_type.REPLACE)
    logger.info('xtier: the DeepSeek V4.1 compressed KV is fake-quantized in PyTorch (the fused kernel needs FP8)')
    return True
