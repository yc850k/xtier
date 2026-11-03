"""The hot expert cache of a GSQ (IQ-quantized GGUF) layer: the hot experts' bytes in the VRAM, decoded on the GPU.

The FP4 pool (xtier/cache/pool.py) keeps the hot experts' packed weights in the VRAM and computes them there while
the CPU engine computes the rest: the routes of the cached experts get the id -1 for the CPU and the GPU's part is
added to the output. The GSQ experts are IQ blocks, so the same shape: the hot experts' bytes (as the checkpoint has
them) go into the VRAM and the kernels decode the blocks they need (xtier/backend/gsq_gpu.py, the engine's decode).

The arena: a slot holds one expert's three tensors (the gate, the up, the down) as the checkpoint has them, so the
GPU decodes the same bytes the engine reads. The pack quantizes every role on its own (the down is often finer than
the gate), so the layout of a slot is computed from the layer's experts: the three tensors' offsets in the slot and
the slot's stride (the largest expert of the layer). The experts of a layer share their role's type, but nothing here
assumes it: the type of every expert is looked up.

The routing is the FP4 pool's: the packing kernel marks the routes of the cached experts for the CPU (-1, the engine
skips them) and gives the GPU their slot, the GPU computes them and the CPU's output is the sum of the two halves.
The residency is static here (the hot experts are chosen at load time; the FP4 pool's adaptive swaps copy from the
engine's RAM shards, which the GSQ path does not have: its experts live in the checkpoint file).

Numerics: the decode is FP32 (the block scales are FP16 and the index arithmetic must not round), the dot takes the
decoded values in FP16 (the card's tensor cores) with an FP32 accumulator, and the intermediate is FP16: the GPU's
expert differs from the engine's by the FP16 rounding of the weights and of the intermediate (the check measures it:
tests/gsq_pool_check.py). Like the FP4 pool, the decode is per route: the routes of a call run in parallel, so the
GPU's part of a call costs about one expert's decode.
"""

import ctypes as C
import logging
import os

import torch
import triton

from xtier import arch, config
from xtier.backend.gsq_gpu import down, down_tiles, gate_up, gate_up_tiles, tables
from xtier.backend.kernels import Pointer
from xtier.backend.prefill import _check, _reduce, _runtime
from xtier.engine import _P
from xtier.cache.pool import (BLOCK, MAX_LAYERS, MAX_M, RING, _commit, _combine, _glob, _pack, _scratch,
                          _tl_dtype, _unpack)

logger = logging.getLogger(__name__)

ROLES = ('w1', 'w3', 'w2')     # the gate, the up, the down (the pack's names)
IQ_TYPES = (16, 17, 19, 29)   # the IQ formats the engine and the GPU decode (csrc/engine/gsq.h)
ALIGN = 256                  # the tensors start at a 256-byte boundary in the slot (the uint16 view stays aligned)
BN = 16                      # the rows of a tensor a program decodes (the gate/up tile and the down tile)


def _align(n):
    """n rounded up to a multiple of ALIGN."""
    return (n + ALIGN - 1) // ALIGN * ALIGN


class GsqPool:
    """The hot experts of one GSQ layer in the VRAM: their IQ bytes, decoded and multiplied by the GPU.

    pack: the checkpoint's pack (the experts' bytes and their types, xtier/backend/gsq.py); layer_id: the model's
    layer; slots: the number of hot experts the layer holds; mailbox: the CPU-GPU mailbox of the layer's plan.
    The expert is computed whole: the tensor-parallel split (a rank's part of the intermediate) is not supported
    (the GSQ path runs one rank).
    """

    layers = []

    def __init__(self, pack, layer_id, slots, mailbox, dtype=None, swiglu_limit=0.0, swiglu_alpha=1.0,
                 swiglu_beta=0.0):
        self.pack, self.layer_id, self.mailbox = pack, int(layer_id), mailbox
        self.E, self.H, self.I = int(pack.experts), int(pack.hidden), int(pack.inter)
        self.S = min(int(slots), self.E)
        self.full = self.S >= self.E   # every expert of the layer is in the VRAM (the FP4 pool's flag)
        self.dtype = dtype if dtype is not None else arch.dtype()
        self.limit = float(swiglu_limit)
        self.alpha, self.beta = float(swiglu_alpha), float(swiglu_beta)
        if self.H % 256 or self.I % 256:
            raise ValueError(f'xtier cache: the GSQ layer {self.layer_id} has H={self.H} and I={self.I}, which are not '
                             'multiples of 256 (the values of an IQ block)')
        if self.H % BN or self.I % BN:
            raise ValueError(f'xtier cache: the GSQ layer {self.layer_id} has H={self.H} and I={self.I}, which are not '
                             f'multiples of the tile ({BN})')
        # the layout of the experts' bytes: the three tensors of an expert in its slot, and the slot's stride
        locations = pack.expert_locations(self.layer_id)
        if len(locations) != self.E:
            raise ValueError(f'xtier cache: the pack has {len(locations)} experts of the layer {self.layer_id}, '
                             f'not {self.E}')
        types, offsets, sizes = [], [], []
        stride = 0
        for roles in locations:
            size = [int(roles[r][2]) for r in ROLES]
            kind = [int(roles[r][3]) for r in ROLES]
            if any(k not in IQ_TYPES for k in kind):
                raise ValueError(f'xtier cache: the layer {self.layer_id} has an expert in a format the GPU cannot '
                                 f'decide ({kind})')
            off = [0, _align(size[0]), _align(size[0] + size[1])]
            types.append(kind)
            offsets.append(off)
            sizes.append(size)
            stride = max(stride, off[2] + size[2])
        self.slot_bytes = _align(stride)
        dev = torch.device('cuda', torch.cuda.current_device())
        self.arena = torch.empty((self.S, self.slot_bytes), dtype=torch.uint8, device=dev)
        self.arena16 = self.arena.view(torch.uint16)
        self.types = torch.tensor(types, dtype=torch.int32, device=dev)
        self.offsets = offsets                       # the host copy (the uploads)
        self.offs = torch.tensor(offsets, dtype=torch.int32, device=dev)
        self.sizes = sizes                           # the bytes of every expert's three tensors
        # The residency tables are rows of the global ones (the FP4 pool's): the adaptive policy's swaps are applied
        # to them by the commit kernel (the ring's commands name the row).
        if not _glob:
            _glob['table'] = torch.full((MAX_LAYERS, self.E), -1, dtype=torch.int32, device=dev)   # expert -> slot
            _glob['owner'] = torch.zeros((MAX_LAYERS, self.E), dtype=torch.int32, device=dev)      # slot -> expert
            _glob['state'] = torch.zeros(2, dtype=torch.int32, device=dev)   # commands evicted / published
            _glob['E'], _glob['rows'] = self.E, 0
            ptr = (C.c_void_p * 2)()   # the mapped command ring must exist before the decode graphs are captured
            if self.mailbox.lib.xtier_mailbox_pool_ring(C.cast(ptr, C.c_void_p)) < 0:
                raise RuntimeError(self.mailbox.error())
            _glob['ring'], _glob['seqs'] = ptr[0], ptr[1]
        if _glob['E'] != self.E:
            raise ValueError('xtier cache: every partly cached layer needs the same expert count')
        self.index = _glob['rows']
        if self.index >= MAX_LAYERS:
            raise ValueError('xtier cache: too many layers')
        _glob['rows'] += 1
        self.table = _glob['table'][self.index]     # expert -> slot
        self.owner = _glob['owner'][self.index]     # slot -> expert
        self.used = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.rank = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.resident = [-1] * self.S
        self.filled = False
        self.scratch = _scratch(self.H, self.I, self.dtype, dev)
        self.pinned = []          # the RAM slots registered with CUDA (pin_slots)
        self.tables = None
        self._pool_on = False
        GsqPool.layers.append(self)

    def bytes(self):
        return self.arena.numel()

    def fill(self, experts, stream):
        """Upload the hot experts' bytes into the slots 0.. and publish them (the residency is static).

        The bytes come from the checkpoint's own files (the pack's mapping), so nothing is unpacked: the GPU reads
        the IQ blocks where the engine reads them.
        """
        experts = [e for e in experts if 0 <= e < self.E][:self.S]
        cudart = _runtime()
        cudart.cudaMemcpyAsync.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
        cudart.cudaMemcpyAsync.restype = C.c_int
        with torch.cuda.stream(stream):
            for j, e in enumerate(experts):
                for role, off, want in zip(ROLES, self.offsets[e], self.sizes[e]):
                    address, size, _ = self.pack.expert_bytes(role, self.layer_id, e)
                    if size != want:
                        raise ValueError(f'xtier cache: the expert {e} of the layer {self.layer_id} has {size} bytes '
                                         f'of {role}, the layout says {want}')
                    dst = self.arena.data_ptr() + j * self.slot_bytes + off
                    _check(cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(address), C.c_size_t(size), 1,
                                                 C.c_void_p(stream.cuda_stream)), 'cudaMemcpyAsync')
        table = torch.full((self.E,), -1, dtype=torch.int32)
        owner = torch.zeros((self.E,), dtype=torch.int32)   # the global row: slot -> expert (E entries)
        for j, e in enumerate(experts):
            table[e] = j
            owner[j] = e
        self.table.copy_(table)
        self.owner.copy_(owner)
        self.resident = experts + [-1] * (self.S - len(experts))
        self.filled = True
        return len(experts)

    def _route(self, m, k, x, ids, routing, act, staging):
        """Stage the call's inputs into the mailbox slot and read the residency table: the routes of the cached
        experts get the id -1 for the CPU and their slot for the GPU (both halves from the same read)."""
        sx, sids, sroute, sorig = staging
        sc = self.scratch
        nk = m * k
        if torch.cuda.is_current_stream_capturing():
            _glob['captured'] = True   # the adaptive policy starts only after the decode graphs exist
        if self.index == 0 and _glob.get('adapt'):
            # once per decode window (anywhere in the window is safe: a slot is overwritten only after this
            # kernel acknowledged its eviction): apply the posted swaps
            _commit[(1,)](Pointer(_glob['ring'], torch.int32), Pointer(_glob['seqs'], torch.int32), _glob['state'],
                          _glob['table'], _glob['owner'], self.E, RING)
        _pack[(triton.cdiv(m * self.H, BLOCK),)](
            Pointer(x, self.dtype), Pointer(ids, torch.int32), Pointer(routing, torch.int32),
            Pointer(act or ids, torch.int32), self.table, Pointer(sx, torch.bfloat16), Pointer(sids, torch.int32),
            Pointer(sroute, torch.int32), Pointer(sorig, torch.int32), self.used, self.rank, sc['rslot'],
            sc['slot_list'], sc['slot_exp'], self.owner, m * self.H, nk, k, self.S, bool(act),
            triton.next_power_of_2(max(self.S, 1)), BLOCK)

    def _kernels(self, m, k, x, routing, gout):
        """The GPU part: the gate/up and the down over the used slots, then the per-token sum of the GPU routes."""
        if self.tables is None:
            self.tables = tables(torch.device('cuda', torch.cuda.current_device()))
        t = self.tables
        sc = self.scratch
        H, I = self.H, self.I
        nk = m * k
        gate_up[(nk, I // BN)](
            Pointer(x, self.dtype), sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.types, self.offs,
            sc['mid'], m, H, I, self.limit, self.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'], t['ksigns'],
            MAX_M, BN, ALPHA=self.alpha, BETA=self.beta, DT=_tl_dtype(self.dtype), num_warps=4, num_stages=2)
        down[(nk, H // BN)](
            sc['mid'], sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.types, self.offs, sc['part'],
            m, H, I, self.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'], t['ksigns'], MAX_M, BN,
            DT=_tl_dtype(self.dtype), num_warps=4, num_stages=2)
        _combine[(m, triton.cdiv(H, 1024))](sc['part'], sc['rslot'], Pointer(routing, torch.float32), gout, H, k,
                                            MAX_M, 1024)

    def decode(self, stream, m, k, x, ids, routing, out, act=0, dtype=None):
        """One decode call of up to 8 tokens on the current stream (device addresses; out: FP32 [m, H]).

        The routes of the cached experts go to the CPU with the id -1 (the engine skips them) and are computed here;
        the GPU's part is added to the CPU's output.
        """
        if dtype is not None and dtype != self.dtype:
            raise TypeError(f'xtier cache: the activations are {dtype}, this layer was built for {self.dtype}')
        if m > 8:
            raise ValueError('xtier cache: decode slot too large')
        sc = self.scratch
        if not self._pool_on:   # the mailbox counts the routes the GPU served
            if self.mailbox.lib.xtier_mailbox_pool_enable(self.mailbox.bridge, 1) < 0:
                raise RuntimeError(self.mailbox.error())
            self._pool_on = True
        sx, sids, sroute, sout, sorig = self.mailbox.slot(stream, m, k, with_routes=True)
        self._route(m, k, x, ids, routing, act, (sx, sids, sroute, sorig))
        if config.debug():
            import ctypes as C
            cudart = _runtime()
            host = torch.empty(m * k, dtype=torch.float32, device='cpu')
            cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
            cudart.cudaMemcpy.restype = C.c_int
            cudart.cudaMemcpy(C.c_void_p(host.data_ptr()), C.c_void_p(sroute), m * k * 4, 2, 0)
            ids_host = torch.empty(m * k, dtype=torch.int32, device='cpu')
            cudart.cudaMemcpy(C.c_void_p(ids_host.data_ptr()), C.c_void_p(sids), m * k * 4, 2, 0)
            torch.cuda.synchronize()
            logger.error('xtier debug: layer %d m %d: the staged routing %s the staged ids %s',
                         self.layer_id, m, host[:6].tolist(), ids_host[:6].tolist())
            torch.cuda.current_stream().synchronize()
        self.mailbox.signal(stream, m, k, 1)
        self._kernels(m, k, x, routing, sc['gout'])
        self.mailbox.signal(stream, m, k, 2)
        _unpack[(triton.cdiv(m * self.H, BLOCK),)](Pointer(sout, torch.float32), sc['gout'], Pointer(out, torch.float32),
                                                   m * self.H, BLOCK)


_scratch_arenas = {}      # (device, stride) -> the arena the prefills upload the experts into
_pinned = {}              # (device, size) -> the pinned buffer the upload stages the bytes in


def _scratch_arena(pool, slots):
    """The arena a prefill uploads the routed experts into (shared by the layers: they take turns with it).

    A layer's experts are ~10 MB of IQ bytes, so the arena is small (the slots of the deepest layer's stride). The
    card may not have the room: the arena then shrinks (the experts that do not fit are computed by the engine).
    """
    key = (pool.arena.device, pool.slot_bytes)
    arena = _scratch_arenas.get(key)
    if arena is not None:
        return arena
    while slots >= 8:
        try:
            arena = torch.empty((slots, pool.slot_bytes), dtype=torch.uint8, device=pool.arena.device)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            slots //= 2
    if arena is None:
        return None
    _scratch_arenas[key] = arena
    logger.info('xtier cache: the scratch arena of the batched prefill: %d experts (%.1f GiB)', slots,
                arena.numel() / 2 ** 30)
    return arena


def _upload(pool, plan, experts, arena, stream):
    """The experts' bytes into the scratch arena over PCIe: the engine's RAM slot when the expert is there (the
    cold pool's RAM is fast), else the checkpoint's files (the pack's mapping).

    This is the batched prefill's trade: the bytes of every expert the batch routes to are moved once and decoded on
    the GPU, instead of being read and decoded by the CPU engine. The cost is the transfer (PCIe), the gain is the
    GPU's decode (about a thousand times faster than the engine's).

    The bytes are staged in pinned memory first: a copy from a mapped file (the pack's mapping) is a pageable one,
    and the driver stages those in small chunks (about a tenth of the bandwidth).
    """
    cudart = _runtime()
    cudart.cudaMemcpyAsync.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
    cudart.cudaMemcpyAsync.restype = C.c_int
    lib = getattr(plan, 'lib', None)
    handle = getattr(plan, 'plan', None)
    with torch.cuda.stream(stream):
        for j, e in enumerate(experts):
            e = int(e)
            # the expert's bytes: the engine's RAM slot when it is there (the cold pool's RAM is fast), else the
            # checkpoint's files (the pack's mapping)
            ram = lib.xtier_slot_pin(handle, e) if lib is not None and handle is not None \
                and hasattr(lib, 'xtier_slot_pin') else -1
            sources = None
            base = None
            if ram >= 0:
                ptrs = (C.c_void_p * 3)()
                sizes = (C.c_size_t * 3)()
                if lib.xtier_gsq_slot_pieces(handle, ram, C.cast(ptrs, C.c_void_p),
                                             C.cast(sizes, C.c_void_p)) == 0:
                    base = ptrs[0]
                    sources = [(ptrs[r], int(sizes[r])) for r in range(3)]
            try:
                for role, off, want in zip(ROLES, pool.offsets[e], pool.sizes[e]):
                    if sources is not None:
                        address, size = sources[ROLES.index(role)]
                    else:
                        address, size, _ = pool.pack.expert_bytes(role, pool.layer_id, e)
                    if size != want:
                        raise ValueError(f'xtier cache: the expert {e} of the layer {pool.layer_id} has {size} bytes of '
                                         f'{role}, the layout says {want}')
                    dst = arena.data_ptr() + j * pool.slot_bytes + off
                    pinned = base is not None and any(b <= address < b + n for b, n in pool.pinned)
                    try:
                        if pinned:
                            # the RAM slot is registered with CUDA: the driver copies it straight to the card
                            rc = cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(address), C.c_size_t(size), 1,
                                                        C.c_void_p(stream.cuda_stream))
                            if rc != 0:
                                cudart.cudaGetLastError()   # clear it: the staged copy below may still work
                                pinned = False
                        if not pinned:
                            key = (pool.arena.device, size)
                            staged = _pinned.get(key)
                            if staged is None:
                                staged = _pinned[key] = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                            C.memmove(staged.data_ptr(), address, size)     # the bytes -> the pinned buffer
                            _check(cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(staged.data_ptr()),
                                                         C.c_size_t(size), 1,
                                                         C.c_void_p(stream.cuda_stream)), 'cudaMemcpyAsync')
                        stream.synchronize()   # the next expert stages over the same buffer
                    except Exception as exc:   # noqa: BLE001  a failed upload must not take the server down
                        cudart.cudaGetLastError()
                        logger.warning('xtier cache: the upload of the expert %d of the layer %d failed (%s): the '
                                       'engine computes its routes', e, pool.layer_id, exc)
                        failed.add(e)
            finally:
                if ram >= 0:
                    lib.xtier_slot_unpin(handle, ram)


def pin_slots(pool, plan, count):
    """The engine's RAM slots registered with CUDA: the pages stay where they are (no swap) and the GPU can read
    them at the full PCIe speed.

    The upload of a batched prefill copies the experts' bytes from there (the cold pool's RAM), and a pageable copy
    goes through the driver's staging buffer at a tenth of the bandwidth. The FP4 path does the same for its shards
    (backend/prefill.py: ensure_registered).

    count: the engine's RAM slots of this layer (the cold pool's resident experts).
    """
    if os.environ.get('XTIER_GSQ_PIN', '1') == '0':
        return 0
    cap = int(float(os.environ.get('XTIER_GSQ_PIN_GIB', '64')) * 2 ** 30)
    cudart = _runtime()
    cudart.cudaHostRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
    cudart.cudaHostRegister.restype = C.c_int
    cudart.cudaHostUnregister.argtypes = [C.c_void_p]
    cudart.cudaHostUnregister.restype = C.c_int
    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = C.c_int
    lib = plan.lib
    ptrs = (C.c_void_p * 3)()
    sizes = (C.c_size_t * 3)()
    done = 0
    for slot in range(int(count)):
        if lib.xtier_gsq_slot_pieces(plan.plan, slot, C.cast(ptrs, C.c_void_p), C.cast(sizes, C.c_void_p)) != 0:
            break
        base = ptrs[0]
        # the slot's three regions are apart (the tensors are 256-byte aligned): the span is from the first to the
        # last one's end, so that every region is inside the registered range
        end = max(ptrs[r] + sizes[r] for r in range(3))
        total = int(end - base)
        if done + total > cap:
            logger.info('xtier cache: layer %d: the RAM registration stops here (%.1f GiB: the driver\'s pinned '
                        'memory is scarce, and the upload copies through its staging buffer)', pool.layer_id,
                        cap / 2 ** 30)
            break
        if cudart.cudaHostRegister(C.c_void_p(base), C.c_size_t(total), 0) != 0:
            # a failed registration leaves a sticky CUDA error: clear it, or every later call fails with it
            cudart.cudaGetLastError()
            logger.warning('xtier cache: the RAM slots of layer %d could not be registered with CUDA (the upload '
                           'stays pageable)', pool.layer_id)
            break
        pool.pinned.append((int(base), total))
        done += total
    if done:
        logger.info('xtier cache: layer %d: %.1f GiB of the engine\'s RAM registered with CUDA (the GPU reads it '
                    'at the full PCIe speed)', pool.layer_id, done / 2 ** 30)
    return done


def prefill(pool, plan, x, ids, weights, cpu_batch=1024):
    """The prefill of a partly cached layer: the routes the GPU computes (the IQ blocks decoded in the tile kernels),
    the rest at the same time on the CPU engine (FP32); the two partial sums are added in FP32.

    x: the activations [m, H] in the model's dtype, ids: int32 [m, k], weights: FP32 [m, k] (device tensors, current
    stream) -> FP32 [m, H]. The residency is read once at the start (the swaps are applied by the decode windows,
    which do not run meanwhile).

    Where the GPU's part comes from: the hot experts are in the VRAM (their bytes were uploaded once, at the load),
    and the experts the batch routes to that are not in the VRAM are **uploaded over PCIe** (the checkpoint's files
    into the scratch arena) and decoded here. That is the trade of a batched prefill: the bytes move once and the
    GPU decodes them (about a thousand times faster than the engine), so the CPU engine only sees the routes of the
    experts that did not fit in the scratch (XTIER_GSQ_PREFILL_UPLOAD slots).
    """
    import threading
    import time
    import time as _time

    H, I, E = pool.H, pool.I, pool.E
    m, k = ids.shape
    if pool.full:
        raise ValueError('xtier cache: the GSQ prefill is for a partly cached layer')
    if pool.tables is None:
        pool.tables = tables(torch.device('cuda', torch.cuda.current_device()))
    t = pool.tables
    di = ids.reshape(-1)
    counts_all = torch.bincount(di, minlength=E)
    order = torch.argsort(di, stable=True)          # the routes sorted by the expert
    hot_expert = pool.table >= 0                    # the experts in the VRAM
    hot = hot_expert[di]
    # the experts the batch routes to that are not in the VRAM: they are uploaded and computed on the GPU
    want = ((counts_all > 0) & ~hot_expert).nonzero().flatten()
    upload_slots = int(os.environ.get('XTIER_GSQ_PREFILL_UPLOAD', '32'))
    arena, experts, uploaded = None, [], torch.zeros(E, dtype=torch.bool, device=di.device)
    upload_ms = 0.0
    _t_stage = time.perf_counter()
    if want.numel() and upload_slots > 0:
        arena = _scratch_arena(pool, upload_slots)
        # the arena may have shrunk (the card had no room): only the experts that fit in it are uploaded
        experts = want[:min(int(want.numel()), upload_slots, arena.shape[0] if arena is not None else 0)].tolist()
        _t_up = time.perf_counter()
        if arena is not None:
            _upload(pool, plan, experts, arena, torch.cuda.current_stream())
            upload_ms = (time.perf_counter() - _t_up) * 1e3
            uploaded[want[:len(experts)]] = True
    stage_ms = (time.perf_counter() - _t_stage) * 1e3
    on_gpu = hot | uploaded[di]                     # the routes the GPU computes
    # the routes of each part, in the expert order (the tiles cut them per expert)
    hot_routes = order[hot[order]]
    up_routes = order[uploaded[di[order]]]
    n_gpu = int(hot_routes.numel()) + int(up_routes.numel())
    # the CPU part: the routes of the experts that are neither in the VRAM nor uploaded (the engine skips the
    # others, the id -1)
    x_h = torch.empty((m, H), dtype=torch.bfloat16, pin_memory=True)
    ids_h = torch.empty((m, k), dtype=torch.int32, pin_memory=True)
    w_h = torch.empty((m, k), dtype=torch.float32, pin_memory=True)
    out_h = torch.empty((m, H), dtype=torch.float32, pin_memory=True)
    x_h.copy_(x, non_blocking=True)
    ids_h.copy_(torch.where(on_gpu.reshape(m, k), -1, ids).to(torch.int32), non_blocking=True)
    w_h.copy_(weights, non_blocking=True)
    errors = []
    cpu_done = []

    def cpu_part():
        try:
            for s0 in range(0, m, cpu_batch):
                n = min(cpu_batch, m - s0)
                plan.forward(n, k, x_h.data_ptr() + s0 * H * 2, ids_h.data_ptr() + s0 * k * 4,
                             w_h.data_ptr() + s0 * k * 4, out_h.data_ptr() + s0 * H * 4)
        except BaseException as e:   # re-raised by the caller after the join
            errors.append(e)
        cpu_done.append(_time.perf_counter())

    worker = threading.Thread(target=cpu_part, name='xtier-gsq-prefill')
    worker.start()
    _t0 = _time.perf_counter()
    _t_pre = _time.perf_counter()
    try:
        out_g = torch.zeros((m, H), dtype=torch.float32, device=x.device)
        if n_gpu:
            # the tiles of one expert's routes, BM at a time
            bm = int(os.environ.get('XTIER_GSQ_PREFILL_TILE_M', '32'))
            bn = int(os.environ.get('XTIER_GSQ_PREFILL_TILE_N', '16'))
            counts = counts_all.cpu().tolist()
            hot_slots = pool.table.cpu().tolist()
            scratch_slots = [-1] * E
            for j, e in enumerate(experts):
                scratch_slots[e] = j
            routes = torch.zeros((m * k, H), dtype=torch.float32, device=x.device)
            for part, order_part, slot_of, arena_part in (
                    ('hot', hot_routes, hot_slots, pool.arena),
                    ('uploaded', up_routes, scratch_slots, arena)):
                if order_part.numel() == 0:
                    continue
                tiles, offset = [], 0
                for e in range(E):
                    c = counts[e]
                    if not c or slot_of[e] < 0:
                        continue
                    for q in range(0, c, bm):
                        tiles.append((e, slot_of[e], offset + q, offset + c))
                    offset += c
                if not tiles:
                    continue
                meta = torch.tensor(tiles, dtype=torch.int32).T.contiguous().to(x.device, non_blocking=True)
                mid = torch.empty((int(order_part.numel()), I), dtype=pool.dtype, device=x.device)
                nt = len(tiles)
                gate_up_tiles[(nt, I // bn)](
                    x, order_part, meta[0], meta[1], meta[2], meta[3], arena_part, arena_part.view(torch.uint16),
                    pool.types, pool.offs, mid, k, H, I, pool.limit, pool.slot_bytes, t['iq1s'], t['iq2xs'],
                    t['iq2xxs'], t['ksigns'], bm, bn, ALPHA=pool.alpha, BETA=pool.beta, DT=_tl_dtype(pool.dtype),
                    num_warps=4, num_stages=2)
                down_tiles[(nt, H // bn)](
                    mid, order_part, meta[0], meta[1], meta[2], meta[3], arena_part, arena_part.view(torch.uint16),
                    pool.types, pool.offs, routes, H, I, pool.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'],
                    t['ksigns'], bm, bn, DT=_tl_dtype(pool.dtype), num_warps=4, num_stages=2)
            # the per-token sum of the GPU's routes (the CPU's routes have no partial)
            w_gpu = torch.where(on_gpu.reshape(m, k), weights, 0.0).reshape(-1)
            _reduce[(m, triton.cdiv(H, 256))](routes, w_gpu, out_g, k, 256, H)
        _t_gpu = _time.perf_counter()
    finally:   # before the caller may run this layer again (elastic cache): the CPU plan takes one caller
        worker.join()
    if os.environ.get('XTIER_TIMING', '0') == '1':
        logger.info('xtier: the GSQ prefill of layer %d: %d tokens, %d of %d routes on the GPU (%d in the VRAM, %d '
                    'uploaded): the GPU part %.1f ms, the CPU part %.1f ms, the upload %.1f ms, the staging %.1f ms',
                    pool.layer_id, m, n_gpu, m * k, int(hot_expert.sum()), len(experts), (_t_gpu - _t0) * 1e3,
                    ((cpu_done[0] - _t0) * 1e3 if cpu_done else 0.0), upload_ms, stage_ms)
    if errors:
        raise errors[0]
    return out_g + out_h.to(x.device, non_blocking=True)


def fill_all():
    """Fill the GSQ pools' slots with the hot experts (the pack's usage profile's ranking) and let the adaptive
    policy follow the routes (the swaps copy the experts from the engine's RAM shards)."""
    todo = [p for p in GsqPool.layers if not p.filled]
    if not todo:
        return 0
    stream = torch.cuda.Stream()
    n = 0
    for pool in todo:
        n += pool.fill(pool.pack.ranking(pool.layer_id), stream)
    partial = [p for p in todo if not p.full]
    if partial and _glob.get('prepared'):
        logger.warning('xtier cache: %d GSQ layers added after the adaptive policy started keep their initial '
                       'residency', len(partial))
        partial = []
    if partial:
        _glob['prepared'] = True
    if os.environ.get('XTIER_CACHE_ADAPT', '1') != '0' and partial:
        every, budget = int(os.environ.get('XTIER_CACHE_EVERY', '4')), int(os.environ.get('XTIER_CACHE_SWAPS', '16'))
        decay = float(os.environ.get('XTIER_CACHE_DECAY', '0.7'))
        gain = float(os.environ.get('XTIER_CACHE_MIN_GAIN', '1.5'))
        lib = partial[0].mailbox.lib
        for pool in partial:
            offsets = (C.c_int32 * (pool.E * 3))(*[o for e in range(pool.E) for o in pool.offsets[e]])
            sizes = (C.c_size_t * (pool.E * 3))(*[s for e in range(pool.E) for s in pool.sizes[e]])
            resident = (C.c_int32 * pool.S)(*pool.resident)
            if lib.xtier_mailbox_pool_attach_gsq(pool.mailbox.bridge, pool.index, pool.S, pool.S, pool.E,
                                                 C.c_void_p(pool.arena.data_ptr()), C.c_size_t(pool.slot_bytes),
                                                 offsets, sizes, resident) < 0:
                raise RuntimeError(pool.mailbox.error())
        _glob['adapt'] = (lib, every, budget, decay, gain)
    return n


def maybe_start():
    """At any MoE call outside capture: start the adaptive policy once the decode graphs have been captured, so its
    copies never overlap a capture."""
    a = _glob.get('adapt')
    if not a or _glob.get('started') or not _glob.get('captured') or torch.cuda.is_current_stream_capturing():
        return
    lib, every, budget, decay, gain = a
    if lib.xtier_mailbox_pool_start(every, budget, C.c_float(decay), C.c_float(gain)) < 0:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    _glob['started'] = True
    logger.info('xtier cache: the GSQ pools\' adaptive residency started: every %d windows, up to %d swaps per '
                'window, decay %s, min gain %s', every, budget, decay, gain)
