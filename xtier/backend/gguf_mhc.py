"""The MHC prefill kernel's block size for a checkpoint whose hc_hidden_size its table does not have.

SGLang's MHC prefill (``sglang.kernels.ops.layernorm.mhc.mhc_pre``) picks the split-K kernel's block over the
hidden size from a table of two shapes (16384 and 28672: the Blackwell checkpoints, which take the DeepGEMM path
instead). The split-K kernel itself is generic: it wants ``hc_hidden_size % hidden_block == 0`` and
``split_size % hidden_block == 0``.

This patch extends the table with the block the other shapes need (the DeepSeek V4.1 Maya pack has hc_hidden_size
20480). The function is rewritten from its own source (the table is a literal in it), and the rewrite is checked
against the source it replaces: if SGLang's code changes, the patch does nothing.
"""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

MHC_MODULE = 'sglang.kernels.ops.layernorm.mhc'

# The table SGLang has: the hidden sizes and the block over them.
TABLE = """            if hc_hidden_size == 16384:
                hidden_block = 256
            elif hc_hidden_size == 28672:
                hidden_block = 128
            else:
                raise NotImplementedError(
                    f"mhc_pre splitk kernel only supports hc_hidden_size in {{16384, 28672}}, "
                    f"got {hc_hidden_size}"
                )"""

# What replaces it: the largest block that divides the hidden size and the split.
PATCHED = """            hidden_block = _xtier_hidden_block(hc_hidden_size, hc_hidden_size // n_splits_pre)"""


def _hidden_block(hc_hidden_size, split_size):
    """The kernel's block over the hidden size: the largest of 256, 128, 64 that divides both."""
    for block in (256, 128, 64):
        if hc_hidden_size % block == 0 and split_size % block == 0:
            return block
    raise NotImplementedError(f'the MHC prefill kernel: no block for the hidden size {hc_hidden_size}')


def install():
    """Let the MHC prefill kernel take the other checkpoints' hidden sizes."""
    import importlib

    mhc = importlib.import_module(MHC_MODULE)
    if getattr(mhc.mhc_pre, '_xtier_block', False):
        return False
    source = textwrap.dedent(inspect.getsource(mhc.mhc_pre))
    if TABLE not in source:
        logger.info('xtier: the MHC prefill kernel changed (its block table is not the one this patch knows)')
        return False
    patched = source.replace(TABLE, PATCHED)
    namespace = dict(vars(mhc))
    namespace['_xtier_hidden_block'] = _hidden_block
    try:
        exec(compile(patched, f'<{MHC_MODULE}.mhc_pre>', 'exec'), namespace)
    except SyntaxError:
        logger.info('xtier: the MHC prefill kernel changed (its source does not patch)')
        return False
    mhc.mhc_pre = namespace['mhc_pre']
    mhc.mhc_pre._xtier_block = True
    logger.info('xtier: the MHC prefill kernel takes the other hidden sizes')
    return True
