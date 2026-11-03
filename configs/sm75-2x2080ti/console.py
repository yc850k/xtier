"""XTier console entry (xtier/console): DeepSeek V4.1 Flash 1017 (GGUF) on a Turing host (2x RTX 2080 Ti, sm_75).

The checkpoint is the GGUF one on the Optane drive: the four shards of the model (IQ2_XXS, mixed per expert) and
the Engram table (IQ4_NL) in the same directory. SGLang reads it with ``--quantization gguf --load-format gguf``.

The card's other fields follow the host: FP16 (the card has no BF16 tensor cores, so the model runs in the
dtype its tensor cores take; the hyper-connection's residual stream needs more range than FP16 has, so the pack keeps
that stream in FP32), the CPU engine's AVX2 build, few hot expert slots (11 GB per card), the smaller tiles of
Turing's 64 KiB of shared memory.

The routed experts are not in the GGUF shards: they come from the GSQ pack (the IQ experts in the format the CPU
engine reads) on the Optane drive, and the hybrid path is on for this checkpoint (the load log names the layers it
took). The Blackwell hosts' cards are not offered here: their checkpoints and their kernels belong to those machines.
"""

import os
from pathlib import Path

from xtier.console.core import common_options, miss_env, miss_option

HERE = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get('MODEL', '/home/userroot/work/model/ds41-gguf')
NAME = os.environ.get('CONSOLE_NAME', 'DeepSeek V4.1 Flash 1017 · GGUF')

MODEL = {
    'id': 'sm75', 'order': 90,
    'name': NAME, 'code': 'DS41-GGUF', 'port': 18200,
    'served': os.environ.get('SERVED_NAME', 'DeepSeek-V4.1-Flash'),
    'spec': '—', 'arch': '40 层 · 384 专家 top-6 · GGUF IQ2_XXS / IQ1_M 混合 · 2 张 2080 Ti',
    'hardware': '2× RTX 2080 Ti · 2× Xeon E5-2680 v4（AVX2）', 'ready_timeout': 3600,
    'bench': {'kind': 'text', 'dec': '0.44', 'pre': '1.35',
              'title': '文本请求（81 token 提示、64 token 输出，TP=2，2026-11-03）：DEC = decode 的 tok/s（0.24→0.44，'
                       '随上下文增长），PRE = 冷 prefill 的 tok/s（热的时候 0.4）；瓶颈是 CPU 引擎的 MoE 与读盘，'
                       '不是显存（两张卡都用满：各 20.9/22.5 GB）'},
    'options': common_options('DSpark', threads=('24', '16', '12'), contexts=('32768', '16384', '8192')) + [
        miss_option(),
    ],
    'notes': [
        {'when': {}, 'text': f'检查点：{CHECKPOINT}（--quantization gguf --load-format gguf）'},
        {'when': {'packs': 'on'}, 'text': 'Turing 上没有 BF16/FP8/FP4 张量核：模型跑 FP16（超连接的残差流留在 FP32），'
                                          '加速包里跑不了的功能自动关闭（启动时日志里报一次）'},
        {'when': {'packs': 'on'}, 'text': '路由专家来自傲腾盘上的 GSQ 包（CPU 引擎读的 IQ 格式）：混合推理对这个检查点生效；'
                                          '内存放不下的专家按路由读盘（XTIER_COLD=1）'},
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
