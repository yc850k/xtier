"""The DeepSeek V4.1 attention on sm_75 (Turing).

SGLang runs the DSV4 attention through the FlashMLA CUDA kernel (``sgl_kernel``), which needs
sm_90+ (the wgmma and TMA instructions). Turing has neither, so the pack points the call at
SGLang's own portable path instead: the gather-and-dequantize plus einsum of
``flash_mla_sm120`` (the same numbers, in PyTorch). The KV cache stays in the packed V4 layout
(448 e4m3 nope + 64 bf16 rope + the ue8m0 scales per token), so no kernel reads fp8 through an
fp8 pointer - the bytes are converted in software.

The kernel is reached through the DSV4 backend's local import of ``flash_mla_with_kvcache``, so
patching the module attribute is enough to move the whole backend (prefill and decode) onto the
portable path.
"""

import logging
import os

from xtier import config

logger = logging.getLogger(__name__)

FLASH_MLA = 'sgl_kernel.flash_mla.flash_mla_with_kvcache'
SPARSE_PREFILL = 'sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._use_sparse_prefill'


def _portable_flash_mla(**kwargs):
    """The FlashMLA call, in PyTorch: gather the KV rows, dequantize, attend."""
    from sglang.kernels.ops.attention.flash_mla_sm120 import _sm120_sparse_decode_fwd

    q = kwargs['q']
    softmax_scale = kwargs.get('softmax_scale')
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, lse = _sm120_sparse_decode_fwd(
        q,
        kwargs['k_cache'],
        kwargs['indices'],
        kwargs.get('topk_length'),
        kwargs.get('attn_sink'),
        kwargs['head_dim_v'],
        softmax_scale,
        kwargs.get('extra_k_cache'),
        kwargs.get('extra_indices_in_kvcache'),
        kwargs.get('extra_topk_length'),
    )
    # the portable path gives BF16 (the Blackwell hosts' activation dtype); on Turing the activations are FP16
    if config.debug():
        import torch
        kv = kwargs['k_cache'][0, 0, 0, :256].float()   # a slice: the whole cache would be gigabytes
        logger.error('xtier debug: the attention: the q not finite %s the out not finite %s the kv bytes not finite %s',
                     int((~torch.isfinite(q.float())).sum()), int((~torch.isfinite(out.float())).sum()),
                     int((~torch.isfinite(kv)).sum()))
    return out.to(q.dtype), lse


def _no_sparse_prefill(self, *args, **kwargs):
    """The sparse prefill kernel is Hopper's; the portable path computes the same attention densely."""
    return False


def install(registry, hook_type):
    """Replace the FlashMLA kernel with the portable one."""
    import sgl_kernel.flash_mla as flash_mla

    flash_mla.flash_mla_with_kvcache = _portable_flash_mla
    registry.register(SPARSE_PREFILL, _no_sparse_prefill, hook_type.REPLACE)
    logger.info('xtier: the DeepSeek V4.1 attention runs in PyTorch on sm_75 (no FlashMLA)')
    return True
