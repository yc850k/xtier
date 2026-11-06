"""The split GGUF reader (xtier/backend/gguf_split.py): the parts read as one file.

  python tests/gguf_split_check.py

Two files written as a llama.cpp split (the first part has the metadata, the second the rest of the tensors) are read
as one checkpoint: the tensors of both and the metadata of the first.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


def write(path, tensors, metadata=True):
    """One GGUF file with the given tensors (name -> numpy array)."""
    import gguf

    writer = gguf.GGUFWriter(path, 'llama')
    if metadata:
        writer.add_name('test')
        writer.add_context_length(4096)
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    stem = root / 'model'
    first = root / 'model-00001-of-00002.gguf'
    second = root / 'model-00002-of-00002.gguf'
    rng = np.random.default_rng(7)
    # the first part: the metadata and the layer's first tensors
    write(first, {
        'blk.0.attn_q_a.weight': rng.standard_normal((1280, 5120)).astype(np.float32),
        'blk.0.attn_kv.weight': rng.standard_normal((512, 5120)).astype(np.float32),
        'blk.0.attn_norm.weight': rng.standard_normal((5120,)).astype(np.float32),
    })
    # the second part: the rest of the tensors (the split's own metadata)
    write(second, {
        'blk.1.attn_q_a.weight': rng.standard_normal((1280, 5120)).astype(np.float32),
        'blk.1.attn_kv.weight': rng.standard_normal((512, 5120)).astype(np.float32),
        'blk.1.attn_norm.weight': rng.standard_normal((5120,)).astype(np.float32),
        'token_embd.weight': rng.standard_normal((5120, 32)).astype(np.float32),
    }, metadata=False)

    from xtier.backend import gguf_split

    check(gguf_split.parts(first) == [str(first), str(second)], 'the parts of the checkpoint are found')
    reader = gguf_split.SplitReader(first)
    names = [t.name for t in reader.tensors]
    check(len(names) == 7, f'the tensors of both parts ({len(names)})')
    check('blk.0.attn_norm.weight' in names and 'token_embd.weight' in names, 'the tensors of the first and the '
          'second part are there')
    check('blk.0.attn_kv.weight' in names, 'the q and the kv are apart (the loader fuses them)')
    q = [t for t in reader.tensors if t.name == 'blk.0.attn_q_a.weight'][0]
    check(q.data.shape == (1280, 5120), f'the q of the first layer ({q.data.shape})')
    name = reader.get_field('general.name')
    check(name is not None and bytes(name.parts[-1]).decode() == 'test', 'the metadata of the first part is there')

print('GGUF-SPLIT-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
