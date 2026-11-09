"""Acceleration packs: per-model-family fixes and kernels, registered as hooks on SGLang (XTIER_PACKS=all|none|list).

A pack's hooks only act when its model's code runs, so every selected pack is installed in every scheduler process.
Packs in SERVING also hook the process that serves the API (how prompts are tokenized, how the server arguments are
resolved): ``install_serving``.
"""

import importlib
import logging
import os

from xtier import arch

logger = logging.getLogger(__name__)
PACKS = ('qwen4_exp', 'glm5_next', 'deepseek_v4', 'minimax_m3', 'speculative')
_unsupported = set()


def supported(need, what, instead=None):
    """Whether this GPU can run a feature that needs the compute capability `need` (e.g. (8, 9) for the FP8 tensor
    cores, (9, 0) for TMA, (10, 0) for the FP4 tensor cores). A feature the card cannot run is replaced (the model's
    own code runs instead, or XTier's own kernel for the same thing) and is reported once.

    `instead` names what takes its place (e.g. 'the Triton sparse MLA'); without it the message says the feature is
    off, which is right when the model's own code takes over."""
    if need is None or arch.at_least(*need):
        return True
    if what not in _unsupported:
        _unsupported.add(what)
        if instead:
            logger.warning('xtier: %s on %s needs sm_%d%d: %s instead', what, arch.describe(), need[0], need[1],
                           instead)
        else:
            logger.warning('xtier: %s is off on %s: it needs sm_%d%d', what, arch.describe(), need[0], need[1])
    return False


def selected() -> list[str]:
    spec = os.environ.get('XTIER_PACKS', 'all').strip()
    if spec in ('', 'all'):
        return list(PACKS)
    if spec == 'none':
        return []
    names = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [n for n in names if n not in PACKS]
    if unknown:
        raise ValueError(f'xtier: unknown packs {unknown} (available: {PACKS})')
    return names


def install(registry, hook_type) -> list[str]:
    names = selected()
    for name in names:
        importlib.import_module(f'xtier.packs.{name}').install(registry, hook_type)
    return names


# packs with hooks in the process that serves the API (prompt tokenization, argument resolution), installed there
# and nowhere else
SERVING = ('minimax_m3',)


def install_serving(registry, hook_type) -> list[str]:
    names = []
    for name in selected():
        if name in SERVING and importlib.import_module(f'xtier.packs.{name}').install_serving(registry, hook_type):
            names.append(name)
    return names
