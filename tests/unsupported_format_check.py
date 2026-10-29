"""A checkpoint format the CPU engine cannot read (a GGUF checkpoint, unquantized experts): the MoE layer stays with
SGLang and the model runs, instead of the load failing on the first layer.

  python tests/unsupported_format_check.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtier.backend import formats  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        bad.append(what)


class GGUFMoEMethod:
    """SGLang's GGUF MoE method (the experts are dequantized from the GGUF file)."""


class UnquantizedFusedMoEMethod:
    """SGLang's method for unquantized (BF16/FP16) experts."""


for cls in (GGUFMoEMethod, UnquantizedFusedMoEMethod):
    try:
        formats.detect(cls())
        check(False, f'{cls.__name__}: no CPU engine format (the layer must stay with SGLang)')
    except NotImplementedError as e:
        check(True, f'{cls.__name__}: no CPU engine format ({e})')

# the hook: the layer stays with SGLang, and the load's self-check does not fail
from xtier.backend import hooks  # noqa: E402
from xtier.backend.method import LayerConfig  # noqa: E402

cfg = LayerConfig(layer_id=7, threads=4)
method = hooks.xtier_method(GGUFMoEMethod(), cfg)
check(isinstance(method, GGUFMoEMethod), 'the GGUF layer stays with SGLang')
check(hooks.DELEGATED and hooks.DELEGATED[-1][0] == 7, f'the delegation is reported ({hooks.DELEGATED})')

# a format XTier can read is taken
class ModelOptNvFp4FusedMoEMethod:
    pass


from xtier.backend.formats import ModelOptNvFp4  # noqa: E402


class FakeNvFp4(ModelOptNvFp4FusedMoEMethod):
    pass


FakeNvFp4.__name__ = 'ModelOptNvFp4FusedMoEMethod'
check(isinstance(formats.detect(FakeNvFp4()), ModelOptNvFp4), 'the NVFP4 method has a CPU engine format')

print('UNSUPPORTED-FORMAT-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
