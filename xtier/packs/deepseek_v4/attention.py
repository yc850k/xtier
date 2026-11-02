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


def _sparse_flash_mla(original, module):
    """FlashMLA's sparse FP8 call in the Triton sparse MLA; the PyTorch path takes over if the kernel cannot run.

    The upstream's kernel (below SM90, the RTX 30 / 40 series) is the one in ``sparse_mla``; it reads the same packed
    FP8 pages and gives the same attention. The calls that are not the sparse FP8 decode keep the PyTorch path: the
    FlashMLA wheel has no build for this card.

    ``XTIER_SPARSE_MLA=0`` keeps the PyTorch path (the measurement of the two).
    """
    state = {'failed': False, 'used': 0}

    def call(**kwargs):
        cache = kwargs.get('k_cache')
        sparse = (kwargs.get('indices') is not None and kwargs.get('is_fp8_kvcache') and cache is not None
                  and cache.shape[-1] == module.DATA_BYTES + module.SCALE_BYTES
                  and kwargs.get('head_dim_v', module.DIM) == module.DIM)
        if sparse and not state['failed'] and os.environ.get('XTIER_SPARSE_MLA', '1') != '0':
            try:
                scale = kwargs.get('softmax_scale')
                q = kwargs['q']
                out = module.sparse_mla(
                    q, cache, kwargs['indices'], kwargs.get('topk_length'), kwargs.get('attn_sink'),
                    scale if scale is not None else q.shape[-1] ** -0.5, kwargs.get('extra_k_cache'),
                    kwargs.get('extra_indices_in_kvcache'), kwargs.get('extra_topk_length'))
                state['used'] += 1
                if state['used'] == 1:
                    print('[xtier] the Triton sparse MLA took the first attention call', flush=True)
                return out
            except Exception as exc:      # the kernel does not fit this card: the PyTorch path
                state['failed'] = True
                logger.warning('xtier: the Triton sparse MLA failed (%s); the attention runs in PyTorch', exc)
        return _portable_flash_mla(**kwargs)

    return call


def install(registry, hook_type):
    """Replace the FlashMLA kernel with the Triton sparse MLA (the PyTorch path is the fallback)."""
    import sgl_kernel.flash_mla as flash_mla

    from xtier.packs.deepseek_v4 import sparse_mla

    flash_mla.flash_mla_with_kvcache = _sparse_flash_mla(flash_mla.flash_mla_with_kvcache, sparse_mla)
    # a newer SGLang reaches the kernel through the interface module (the upstream's symbol)
    try:
        from sglang.kernels.ops.attention import flash_mla_interface
    except ImportError:
        flash_mla_interface = None
    if flash_mla_interface is not None and hasattr(flash_mla_interface, 'flash_mla_with_kvcache'):
        flash_mla_interface.flash_mla_with_kvcache = _sparse_flash_mla(
            flash_mla_interface.flash_mla_with_kvcache, sparse_mla)
    registry.register(SPARSE_PREFILL, _no_sparse_prefill, hook_type.REPLACE)
    logger.info('xtier: the DeepSeek V4.1 attention runs in the Triton sparse MLA on sm_75 (no FlashMLA)')
    return True
