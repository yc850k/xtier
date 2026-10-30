"""The DeepSeek V4.1 RoPE kernel on a card that runs FP16.

The model's fused Q/K RoPE is a JIT kernel compiled for BF16 (the dtype is fixed in its source), so a Turing card,
whose activations are FP16 (it has no BF16 tensor cores), cannot run it.

SGLang's own Triton RoPE (the one the ROCm and the XPU paths use) takes the dtype as it is, so this replaces the JIT
kernel with it when the activations are not BF16.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def install():
    """Use the Triton RoPE when the activations are not BF16 (the JIT kernel is BF16 only)."""
    from sglang.kernels.ops.attention.dsv4 import elementwise

    if getattr(elementwise.fused_rope_inplace, '_xtier_fp16', False):
        return False
    original = elementwise.fused_rope_inplace

    def fused_rope_inplace(q, k, freqs_cis, positions, inverse=False):
        if q.dtype != torch.bfloat16:
            # the JIT kernel is BF16 only: the Triton one takes the dtype as it is
            from sglang.kernels.ops.attention.deepseek_v4_rope import apply_rotary_emb_triton

            apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
            if k is not None:
                apply_rotary_emb_triton(k, freqs_cis, positions=positions, inverse=inverse)
            return
        return original(q, k, freqs_cis, positions, inverse)

    fused_rope_inplace._xtier_fp16 = True
    fused_rope_inplace._xtier_original = original
    elementwise.fused_rope_inplace = fused_rope_inplace
    # the model imported the function by name: its own reference has to be replaced too
    try:
        from sglang.srt.models import deepseek_v4

        deepseek_v4.fused_rope_inplace = fused_rope_inplace
    except ImportError:
        pass
    logger.info('xtier: the DeepSeek V4.1 RoPE runs on Triton (the JIT kernel is BF16 only)')
    return True
