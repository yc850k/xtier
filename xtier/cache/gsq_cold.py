"""The GSQ cold pool (XTIER_COLD=1): the experts of a layer in the RAM slots, the rest on the drive.

The GSQ experts live in the checkpoint's own GGUF files (on the Optane drive), so the pool needs no copy of its own:
the store is the checkpoint's shards, opened read-only and read with O_DIRECT (nothing goes through the page cache).
A layer's hot experts (the pack's usage profile's ranking) are read into the RAM slots at load; the rest are read
into a slot when a forward routes to them (the engine's policy: the coldest resident expert leaves RAM, the count
decaying with the half-life).

The pool is per process: the store (the checkpoint's files) is opened once and shared by the layers.
"""

import logging
import os
import threading
import time

from xtier.engine import ColdStore

logger = logging.getLogger(__name__)

ROLES = ('w1', 'w3', 'w2')     # the gate, the up, the down (the pack's names)
BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 29: 56}   # the bytes of a 256-value block (gsq_gpu._block_bytes)

_stores = {}                  # the pack's path -> (the store, the pack's file index -> the store's index)
_store_lock = threading.Lock()
_layers = []                  # (layer id, plan, store) of every layer with a cold pool (the stats reporter)


def enabled():
    return os.environ.get('XTIER_COLD', '0') == '1'


def _float(name, default):
    return float(os.environ.get(name, default))


def ranking(pack, layer_id, experts):
    """The layer's experts, the most used first (the pack's usage profile; the expert order without one)."""
    order = pack.ranking(layer_id)
    return order if len(order) == experts else list(range(experts))


def _store_for(pack):
    """The store: the checkpoint's own files, opened read-only (the experts' shards only)."""
    with _store_lock:
        if pack.path not in _stores:
            used = {int(entry[1]) for name, entry in pack.tensors.items() if '.ffn.experts.' in name}
            paths, index = [], {}
            for i in sorted(used):
                path = pack.file_path(i)
                if not path.exists():
                    raise FileNotFoundError(f'xtier cold: the checkpoint file {path} is missing')
                index[i] = len(paths)
                paths.append(path)
            if not paths:
                raise RuntimeError('xtier cold: the pack has no expert files')
            threads = int(os.environ.get('XTIER_COLD_THREADS', 8))
            _stores[pack.path] = (ColdStore(paths, threads=threads, writable=False), index)
            logger.info('xtier cold: the store: %d checkpoint files on %s (read-only, O_DIRECT)',
                        len(paths), paths[0].parent)
        return _stores[pack.path]


def attach(plan, pack, layer_id, gpu_slots=0, rank=0, part=0):
    """The cold pool of one layer: the store (the checkpoint's own files), the RAM slots (the hot experts) and the
    engine's attach (the store's locations of every expert's three tensors).

    plan: the layer's engine plan; pack: the pack's experts; layer_id: the model's layer; gpu_slots: the number of
    experts the GPU hot cache holds (they are computed by the GPU: their RAM slots go first); rank, part: the
    tensor-parallel rank and the size of its slice of the intermediate. The RAM slots hold the rank's slice of the
    expert (the rows of the gate and the up, the whole down), so the same RAM holds one and a half times the experts
    and the pool's working set fits: the drive is read once, not on every call.
    """
    t0 = time.perf_counter()
    experts = int(pack.experts)
    store, index = _store_for(pack)
    locations = pack.expert_locations(int(layer_id))
    if len(locations) != experts:
        raise RuntimeError(f'xtier cold: the pack has {len(locations)} experts of the layer {layer_id}, not {experts}')
    hidden = int(pack.hidden)
    # the store's locations of every expert's three tensors ([e * 3 + role]): the gate and the up are read from the
    # rank's rows (a contiguous range inside the tensor), the down whole (its columns are strided inside a row)
    files, offsets = [], []
    for e in range(experts):
        for role in ROLES:
            file, offset, size, kind = locations[e][role]
            files.append(index[file])
            if role != 'w2' and part:
                row = (hidden // 256) * BLOCK_BYTES[kind]
                offset += rank * part * row
            offsets.append(int(offset))
    # the RAM slots: the hot experts (the profile's ranking), the rest are read from the drive when they are routed to
    rank = ranking(pack, layer_id, experts)
    cold = int(round(_float('XTIER_COLD_FRACTION', 0.3) * experts))
    slots = max(1, experts - cold)
    # the GPU pool's experts are in the VRAM (the GPU reads their bytes from the checkpoint): the RAM slots go to the
    # ones after them
    resident = rank[int(gpu_slots):int(gpu_slots) + slots]
    half_life = _float('XTIER_COLD_HALF_LIFE', 50)
    # the experts that are not in RAM start with the count their ranking gives (the hot ones ahead); the GPU pool's
    # experts start at 0: the GPU computes them, their RAM slots go first.
    position = {e: r for r, e in enumerate(rank)}
    scores = [0.0 if position[e] < int(gpu_slots) else half_life * 1.44 * (1 - position[e] / experts)
              for e in range(experts)]
    plan.attach_cold_gsq(store, files, offsets, resident, scores, half_life)
    plan.dynamic = slots   # the slots experts can leave: bounds a call's passes (CpuPlan.forward)
    _layers.append((layer_id, plan, store))
    _start_reporter()
    logger.info('xtier cold: layer %d: %d of %d experts in RAM, the rest in %d checkpoint files on the drive '
                '(%d GB read at load) in %.1f s', layer_id, slots, experts, len(index),
                sum(locations[e][role][2] for e in resident for role in ROLES) / 1e9, time.perf_counter() - t0)
    return slots


_reporter = None


def _start_reporter():
    """XTIER_COLD_LOG=60: every this many seconds, what the pool read from the drive."""
    global _reporter
    if _reporter is not None:
        return
    every = int(os.environ.get('XTIER_COLD_LOG', '0') or 0)
    if every <= 0:
        return
    _reporter = threading.Thread(target=_report, args=(every,), name='xtier-gsq-cold-log', daemon=True)
    _reporter.start()


def _report(every):
    last = None
    while True:
        time.sleep(every)
        now = [0.0] * 6
        for _, plan, _ in _layers:
            st = plan.cold_stats()
            for i in range(6):
                now[i] += st[i]
        if last is not None:
            d = [a - b for a, b in zip(now, last)]
            if d[0] or d[1]:
                logger.info('xtier cold: the last %d s: %d forwards read %d experts (%.2f per forward) in %.1f s',
                            every, d[0], d[1], d[1] / d[0] if d[0] else 0.0, d[2])
        last = now


def stats(plan):
    """(forwards, experts read from the drive, seconds spent reading; the same for the decode steps)."""
    return plan.cold_stats()
