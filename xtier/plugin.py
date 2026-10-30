"""SGLang general-plugin entry point (entry point group ``sglang.srt.plugins``).

SGLang calls ``load()`` early in every process and then applies the hooks registered through
``sglang.srt.plugins.hook_registry.HookRegistry``. XTier registers its model hooks only in scheduler processes (the
ones that build models: importing the hook targets elsewhere would only cost startup time); ``install()`` does it
unconditionally, for tools that build a ModelRunner in their own process. The other processes, the one that serves
the API among them, get the few hooks on how prompts are tokenized and the host fixes of serving
(``install_serving()``). SGLang logs and swallows exceptions raised
here, so the backend checks after every model load that it took over (``xtier.backend.hooks.check``).
"""

import logging
import os
import sys

from xtier import __version__, config

logger = logging.getLogger(__name__)
_installed = []
_serving = []


def _in_scheduler() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == 'run_scheduler_process':
            return True
        frame = frame.f_back
    return False


def _install_gguf() -> None:
    """The GGUF checkpoints: a split checkpoint is read as one file, and the DeepSeek V4.1 names are mapped.

    SGLang's GGUF loader reads the checkpoint with the ``gguf`` package's reader (one file) and derives the tensor
    names from the architectures the package and Transformers know. A Maya pack of DeepSeek V4.1 is a split checkpoint
    of an architecture neither knows, so both are supplied here.
    """
    from xtier.backend import gguf_mhc, gguf_names, gguf_split, gguf_weights

    gguf_split.install()
    gguf_split.install_loader()
    gguf_weights.install_method()
    gguf_weights.install_iterator()
    gguf_mhc.install()
    try:
        from sglang.srt.model_loader.gguf_name_maps import GGUF_HF_NAME_MAP_BUILDERS
    except ImportError:
        return
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as _FM
    _orig_init = _FM.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        print(f'[xtier] DEBUG the MoE layer: the quant config {type(self.quant_config).__name__} '
              f'-> the method {type(self.quant_method).__name__}', flush=True)

    _FM.__init__ = _init
    from sglang.srt.layers.quantization import gguf as _gguf
    _orig = _gguf.GGUFConfig.get_quant_method

    def _get_quant_method(self, layer, prefix):
        m = _orig(self, layer, prefix)
        print(f'[xtier] DEBUG the GGUF config: {type(layer).__name__} {prefix} -> {type(m).__name__}', flush=True)
        return m

    _gguf.GGUFConfig.get_quant_method = _get_quant_method
    if 'deepseek_v41' not in GGUF_HF_NAME_MAP_BUILDERS:
        GGUF_HF_NAME_MAP_BUILDERS['deepseek_v41'] = gguf_names.build_deepseek_v41_name_map
        logger.info('xtier: the GGUF name map for DeepSeek V4.1 registered')


def install() -> None:
    if _installed:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from xtier import packs, platform
    from xtier.backend import hooks

    _install_gguf()
    hooks.install(HookRegistry, HookType)
    from xtier.cache import elastic
    if elastic.install(HookRegistry, HookType):
        logger.info('xtier: elastic expert cache on (XTIER_CACHE_ELASTIC=1)')
    names = packs.install(HookRegistry, HookType)
    fixes = platform.install(HookRegistry, HookType)
    _installed.append(True)
    from xtier import arch
    logger.info('xtier %s: MoE backend hooks, packs %s and host fixes %s registered on %s (pid %d)', __version__,
                names, fixes, arch.describe(), os.getpid())


def install_serving() -> None:
    if _serving:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from xtier import packs, platform

    names = packs.install_serving(HookRegistry, HookType)
    fixes = platform.install_serving(HookRegistry, HookType)
    _serving.append(True)
    logger.info('xtier %s: serving hooks of packs %s and host fixes %s registered (pid %d)', __version__, names,
                fixes, os.getpid())


def load() -> None:
    if not config.enabled():
        logger.debug('xtier %s: XTIER is not 1, nothing registered', __version__)
        return
    if os.environ.get('XTIER_MAIN_CPUS'):
        # XTier places the threads itself. SGLang binds every scheduler to its GPU's NUMA node when the main process
        # starts it (numactl, SGLANG_NUMA_BIND_V2): XTier's main CPUs of that node only (2 on the RTX PRO 5000 host)
        # and a preferred memory policy that puts the scheduler's own allocations there, where the engine's
        # node-bound shards then find no pages (10-23: OOM kill, CONSTRAINT_MEMORY_POLICY nodemask=0). Off here, in
        # the main process, before any scheduler starts (an explicit --numa-node still applies).
        os.environ.setdefault('SGLANG_AUTO_NUMA_BIND', '0')
        from xtier.engine import placement
        if not placement.deferred():   # with TP the rank's placement waits for its first MoE layer
            placement.apply(config.threads())
    if _in_scheduler():
        install()
        # SGLang configures logging only after loading the plugins: one line on stderr
        print(f'[xtier] {__version__}: hooks installed in scheduler pid {os.getpid()}, CPUs '
              f'{len(os.sched_getaffinity(0))}, SGLANG_AUTO_NUMA_BIND={os.environ.get("SGLANG_AUTO_NUMA_BIND", "")}',
              file=sys.stderr, flush=True)
    else:
        install_serving()
