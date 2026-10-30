"""The TileLang kernels of DeepSeek V4.1 that a Turing card cannot run.

The model's hyper-connection mix and its indexer have TileLang kernels (the MHC's pre-mix and the DSA indexer). They
are compiled for the card's compute capability, but the kernels themselves need the tensor cores of sm_80 and newer
(the MMA shapes and the TMA they use), so on a Turing card they compile and then fail to load.

This turns them off there: SGLang's own paths take over (the MHC's pre-mix has a pure-PyTorch implementation, the
indexer a Triton one), the same way the other packs turn off what the card cannot run.

The MHC's Sinkhorn has no switch: it is always the TileLang kernel, and on Turing it runs but returns NaN (the
hidden state turns into NaN and the engine rejects the routing weights). SGLang ships a Triton port of it (the
gfx1250 path), so the patch points the Sinkhorn at that port here.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The switches SGLang reads (sglang.srt.environ): the MHC's TileLang pre-mix and post-mix, the indexer's TileLang
# kernel and the fused post+pre boundary.
SWITCHES = (
    ('SGLANG_OPT_USE_TILELANG_MHC_PRE', 'the MHC pre-mix kernel'),
    ('SGLANG_OPT_USE_TILELANG_MHC_POST', 'the MHC post-mix kernel'),
    ('SGLANG_OPT_USE_TILELANG_INDEXER', 'the DSA indexer kernel'),
)


def _sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """The MHC's Sinkhorn on SGLang's Triton port (the TileLang kernel needs the CK-backed addressing of sm_90+)."""
    from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_triton

    return _hc_split_sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


def install():
    """Turn the TileLang kernels off on a card that cannot load them."""
    from sglang.srt.environ import envs

    changed = []
    for name, what in SWITCHES:
        if os.environ.get(name) is not None:
            continue   # the user set it: their call
        flag = getattr(envs, name, None)
        if flag is None:
            continue
        try:
            flag.set(False)
        except Exception:   # the flag is already set
            continue
        changed.append(what)

    # The Sinkhorn has no switch: the TileLang kernel runs and returns NaN here.
    from sglang.kernels.ops.layernorm import mhc as mhc_kernels

    if getattr(mhc_kernels.hc_split_sinkhorn, '_xtier_triton', False):
        return changed
    mhc_kernels.hc_split_sinkhorn = _sinkhorn_triton
    _sinkhorn_triton._xtier_triton = True
    changed.append('the MHC sinkhorn (the Triton port)')

    if changed:
        logger.info('xtier: the TileLang kernels are off: %s (the card cannot load them)', ', '.join(changed))
    return changed
