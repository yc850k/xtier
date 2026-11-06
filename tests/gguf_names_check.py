"""The DeepSeek V4.1 GGUF name map (xtier/backend/gguf_names.py): every checkpoint tensor the model needs must reach
a model parameter.

The shared expert is the case that matters: its three tensors (the gate, the up and the down) were missing from the
map, so the model's shared expert kept uninitialized weights - the hidden state turned into NaN and the engine
rejected the routing weights.

  python tests/gguf_names_check.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtier.backend.gguf_names import build_deepseek_v41_name_map  # noqa: E402

LAYERS = 40


def main():
    config = SimpleNamespace(num_hidden_layers=LAYERS)
    names = build_deepseek_v41_name_map(config)

    # the shared expert: the gate and the up are stacked into the model's fused gate_up projection
    for layer in range(LAYERS):
        prefix = f'blk.{layer}.'
        model_prefix = f'model.layers.{layer}.mlp.shared_experts.'
        for gguf, model in (
            ('ffn_gate_shexp.weight', 'gate_proj.weight'),
            ('ffn_up_shexp.weight', 'up_proj.weight'),
            ('ffn_down_shexp.weight', 'down_proj.weight'),
        ):
            want = model_prefix + model
            got = names.get(prefix + gguf)
            if got != want:
                raise AssertionError(f'{prefix}{gguf} maps to {got!r}, not {want!r}')

    # the routed experts are not in the model: the CPU engine reads them from the checkpoint file
    for layer in range(LAYERS):
        for role in ('gate', 'up', 'down'):
            name = f'blk.{layer}.ffn_{role}_exps.weight'
            if name in names:
                raise AssertionError(f'{name} is in the map: the model must not hold the routed experts')

    # the attention and the hyper-connection are mapped
    for layer in range(LAYERS):
        for gguf in ('attn_q_b.weight', 'attn_output_b.weight', 'hc_attn_fn.weight', 'ffn_gate_inp.weight'):
            if f'blk.{layer}.{gguf}' not in names:
                raise AssertionError(f'blk.{layer}.{gguf} is not in the map')

    print(f'ok: the name map covers {len(names)} tensors (the shared expert included)')


if __name__ == '__main__':
    main()
