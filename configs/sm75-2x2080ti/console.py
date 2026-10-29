"""XTier console entry (xtier/console): a MoE model on a Turing host (2x RTX 2080 Ti, sm_75).

The checkpoint comes from the environment: ``MODEL=/path/to/checkpoint python -m xtier.console`` (the launcher
takes it the same way). The card's other fields follow the host: FP16 (no BF16 tensor cores), the CPU engine's AVX2
build, few hot expert slots (11 GB per card), the smaller tiles of Turing's 64 KiB of shared memory.

The models of the Blackwell hosts are not offered here: their checkpoints, their kernels and their settings belong
to those machines.
"""

import os
from pathlib import Path

from xtier.console.core import common_options, miss_env, miss_option

HERE = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get('MODEL', '').strip()
NAME = os.environ.get('XTIER_SM75_NAME', 'MoE 模型（FP16）')

MODEL = {
    'id': 'sm75', 'order': 90,
    'name': NAME, 'code': 'SM75', 'port': 18200,
    'served': os.environ.get('SERVED_NAME', os.environ.get('XTIER_SM75_NAME', 'XTier-SM75')),
    'spec': '—', 'arch': 'Turing sm_75 · FP16 · 专家在 CPU 引擎，热专家在 GPU',
    'hardware': '2× RTX 2080 Ti · 2× Xeon E5-2680 v4（AVX2）', 'ready_timeout': 1800,
    'options': common_options('—', threads=('24', '16', '12'), contexts=('32768', '16384', '8192')) + [
        miss_option(),
    ],
    'notes': [
        {'when': {}, 'text': f'检查点：{CHECKPOINT or "未设置（MODEL=/path/to/checkpoint）"}'},
        {'when': {'packs': 'on'}, 'text': 'Turing 上没有 BF16/FP8/FP4 张量核：加速包里跑不了的功能自动关闭（启动时日志里报一次）'},
    ],
}


def build(o, port):
    """(environment additions, argv) for validated options o."""
    env = {'MODEL': CHECKPOINT, 'PORT': str(port), 'XTIER_THREADS': o['threads']}
    env.update(miss_env(o))
    if o['packs'] == 'off':   # no pack kernels: SGLang's own code
        env.update(XTIER_HC_MIX_OVERLAP='0', XTIER_QSA_KEYS_PER_CTA='0', XTIER_QSA_FP4='0')
    ctx = o['context']
    args = ['--context-length', ctx, '--max-total-tokens', ctx, '--mem-fraction-static', '0.85',
            '--cuda-graph-max-bs-decode', '2']
    return env, [str(HERE / 'run-xtier-sm75.sh'), *args]


def tags(o):
    """Extra parts of the server's label."""
    return ['FP16'] + (['未命中专家 GPU 代算'] if miss_env(o) else [])
