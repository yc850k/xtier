"""The TileLang kernels of DeepSeek V4.1 that a Turing card cannot run.

The model's hyper-connection mix and its indexer have TileLang kernels (the MHC's pre-mix and the DSA indexer). They
are compiled for the card's compute capability, but the kernels themselves need the tensor cores of sm_80 and newer
(the MMA shapes and the TMA they use), so on a Turing card they compile and then fail to load.

This turns them off there: SGLang's own paths take over (the MHC's pre-mix has a pure-PyTorch implementation, the
indexer a Triton one), the same way the other packs turn off what the card cannot run.
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
    if changed:
        logger.info('xtier: the TileLang kernels are off: %s (the card cannot load them)', ', '.join(changed))
    return changed
