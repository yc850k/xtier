"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 (GGUF) on a Turing host (2x RTX 2080 Ti, sm_75).

The checkpoint is the GGUF one on the Optane drive: the four shards of the model (IQ2_XXS, mixed per expert) and
the Engram table (IQ4_NL) in the same directory. SGLang reads it with ``--quantization gguf --load-format gguf``.

The card's other fields follow the host: FP16 (no BF16 tensor cores), the CPU engine's AVX2 build, few hot expert
slots (11 GB per card), the smaller tiles of Turing's 64 KiB of shared memory.

The experts of a GGUF checkpoint are not in a format the CPU engine can read (it reads NVFP4 and MXFP4), so the MoE
layers stay with SGLang's own GGUF kernels: XTier's hybrid path is off for this checkpoint (the load log says so).
The Blackwell hosts' cards are not offered here: their checkpoints and their kernels belong to those machines.
"""

import os
from pathlib import Path

from xtier.console.core import common_options, miss_env, miss_option

HERE = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get('MODEL', '/home/userroot/work/model/ds41-gguf')
NAME = os.environ.get('XTIER_SM75_NAME', 'DeepSeek V4.1 Flash 1017 · GGUF')

MODEL = {
    'id': 'sm75', 'order': 90,
    'name': NAME, 'code': 'DS41-GGUF', 'port': 18200,
    'served': os.environ.get('SERVED_NAME', 'DeepSeek-V4.1-Flash'),
    'spec': '—', 'arch': '40 层 · 384 专家 top-6 · GGUF IQ2_XXS / IQ1_M 混合 · 2 张 2080 Ti',
    'hardware': '2× RTX 2080 Ti · 2× Xeon E5-2680 v4（AVX2）', 'ready_timeout': 3600,
    'options': common_options('DSpark', threads=('24', '16', '12'), contexts=('32768', '16384', '8192')) + [
        miss_option(),
    ],
    'notes': [
        {'when': {}, 'text': f'检查点：{CHECKPOINT}（--quantization gguf --load-format gguf）'},
        {'when': {'packs': 'on'}, 'text': 'Turing 上没有 BF16/FP8/FP4 张量核：加速包里跑不了的功能自动关闭（启动时日志里报一次）'},
        {'when': {'packs': 'on'}, 'text': 'GGUF 的专家不是 CPU 引擎能读的格式（NVFP4/MXFP4），MoE 层留给 SGLang 自己的 '
                                          'GGUF 内核：混合推理对这个检查点不生效', 'warn': True},
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
            '--cuda-graph-max-bs-decode', '2']   # the GGUF flags are in the launcher
    return env, [str(HERE / 'run-xtier-sm75.sh'), *args]


def tags(o):
    """Extra parts of the server's label."""
    return ['GGUF', 'FP16'] + (['未命中专家 GPU 代算'] if miss_env(o) else [])
