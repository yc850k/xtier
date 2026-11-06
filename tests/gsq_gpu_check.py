"""The GPU's IQ decode (xtier/backend/gsq_gpu.py) against the engine's, on the GPU.

  python tests/gsq_gpu_check.py

The hot expert cache computes the cached experts on the GPU, so the GPU has to decode the IQ blocks exactly as the
CPU engine does (llama.cpp's dequantize_row_iq*). This checks the Triton port against the engine's decode, bit for
bit, for the four formats of the GSQ checkpoints (IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M).

Runs on any GPU with a Triton build (the decode is the same everywhere).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xtier.backend import gsq_gpu  # noqa: E402
from xtier.engine import engine  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


BLOCK = 256
BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 29: 56}
TYPES = (16, 17, 19, 29)
BN = 8           # the rows a program decodes
ROUNDS = 64      # the blocks of every format


def engine_decode(type_id, blob, n):
    """The engine's decode of a row of n values (FP32)."""
    lib = engine()
    dst = np.zeros(n, dtype=np.float32)
    rc = lib.xtier_gsq_decode(type_id, blob.ctypes.data, dst.ctypes.data, n)
    if rc < 0:
        raise RuntimeError(f'xtier: the engine refused to decode the type {type_id}')
    return dst


def main():
    dev = torch.device('cuda', torch.cuda.current_device())
    t = gsq_gpu.tables(dev)
    lib = engine()
    rng = np.random.default_rng(20261029)
    for type_id in TYPES:
        nb = BLOCK_BYTES[type_id]
        # the random blocks: the scale fields are set to finite values (the engine's decode would give NaNs)
        blob = rng.integers(0, 256, ROUNDS * nb, dtype=np.uint8).copy()
        if type_id == 16:      # IQ2_XXS: the fp16 d at the start of every block
            h = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            for i in range(ROUNDS):
                blob[i * nb:i * nb + 2] = h
        elif type_id == 17:    # IQ2_XS: the fp16 d at the start of every block
            h = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            for i in range(ROUNDS):
                blob[i * nb:i * nb + 2] = h
        elif type_id == 19:    # IQ1_S: the fp16 d at the start of every block
            h = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
            for i in range(ROUNDS):
                blob[i * nb:i * nb + 2] = h
        else:                 # IQ1_M: the scales at 48 (the fp16 0x3C00 = 1.0)
            sc = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x3000]).tobytes(), dtype=np.uint8)
            for i in range(ROUNDS):
                blob[i * nb + 48:i * nb + 56] = sc
        ref = engine_decode(type_id, blob, ROUNDS * BLOCK)
        # the GPU: the rows of one block each (the base of every row is its block's offset)
        b8 = torch.from_numpy(blob).to(dev)
        b16 = b8.view(torch.uint16)
        for start in range(0, ROUNDS, BN):
            rows = min(BN, ROUNDS - start)
            got = torch.empty((rows, BLOCK), dtype=torch.float32, device=dev)
            gsq_gpu.decode_rows[(1,)](
                type_id, b8, b16, start * nb, nb, 0, nb, got,
                t['iq1s'], t['iq2xs'], t['iq2xxs'], t['ksigns'], rows)
            want = ref[start * BLOCK:(start + rows) * BLOCK].reshape(rows, BLOCK)
            if not torch.equal(got.cpu(), torch.from_numpy(want)):
                diff = (got.cpu() - torch.from_numpy(want)).abs()
                check(False, f'type {type_id}: the GPU decode matches the engine\'s (rows {start}..{start + rows}, '
                           f'{int((diff > 0).sum())} of {diff.numel()} values differ, max {diff.max():.3g})')
                break
        else:
            check(True, f'type {type_id}: the GPU decode matches the engine\'s bit for bit ({ROUNDS} blocks)')


main()
print('GSQ-GPU-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
