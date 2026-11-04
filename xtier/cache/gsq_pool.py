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

import collections
import ctypes as C
import logging
import os
import time

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
BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 29: 56}   # the bytes of a 256-value block (gsq_gpu._block_bytes)
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
                 swiglu_beta=0.0, tp_rank=0, tp_part=0):
        self.pack, self.layer_id, self.mailbox = pack, int(layer_id), mailbox
        self.E, self.H, self.I = int(pack.experts), int(pack.hidden), int(pack.inter)
        # The tensor-parallel split: the rank computes its own slice of the intermediate (the rows of the gate and
        # the up, the columns of the down) and SGLang adds the ranks' partial outputs up. The expert's bytes stay
        # whole in the VRAM (the kernels read the rank's part of them).
        self.part = int(tp_part) if tp_part else self.I
        self.ioff = int(tp_rank) * self.part
        self.IP = self.part             # the width of the intermediate the rank keeps
        self.S = min(int(slots), self.E)
        self.full = self.S >= self.E   # every expert of the layer is in the VRAM (the FP4 pool's flag)
        self.dtype = dtype if dtype is not None else arch.dtype()
        self.limit = float(swiglu_limit)
        self.alpha, self.beta = float(swiglu_alpha), float(swiglu_beta)
        if self.H % 256 or self.I % 256:
            raise ValueError(f'xtier cache: the GSQ layer {self.layer_id} has H={self.H} and I={self.I}, which are not '
                             'multiples of 256 (the values of an IQ block)')
        if self.H % BN or self.I % BN or self.part % BN:
            raise ValueError(f'xtier cache: the GSQ layer {self.layer_id} has H={self.H} and I={self.I}, which are not '
                             f'multiples of the tile ({BN})')
        if self.ioff + self.part > self.I:
            raise ValueError(f'xtier cache: the GSQ layer {self.layer_id}: the rank {tp_rank} takes the rows '
                             f'{self.ioff}..{self.ioff + self.part} of an intermediate of {self.I}')
        # the layout of the experts' bytes: the three tensors of an expert in its slot, and the slot's stride
        locations = pack.expert_locations(self.layer_id)
        if len(locations) != self.E:
            raise ValueError(f'xtier cache: the pack has {len(locations)} experts of the layer {self.layer_id}, '
                             f'not {self.E}')
        types, offsets, sizes, tensor_sizes, slice_off = [], [], [], [], []
        stride = 0
        for roles in locations:
            size = [int(roles[r][2]) for r in ROLES]
            kind = [int(roles[r][3]) for r in ROLES]
            if any(k not in IQ_TYPES for k in kind):
                raise ValueError(f'xtier cache: the layer {self.layer_id} has an expert in a format the GPU cannot '
                                 f'decide ({kind})')
            # The rank's slice of the intermediate: the rows of the gate and the up (the intermediate's rows, a
            # contiguous range inside the tensor), so only those rows travel to the card. The down's columns are
            # strided inside a row, so the whole down travels.
            row_bytes = [(self.H // 256) * BLOCK_BYTES[k] for k in kind]
            cut = [self.ioff * row_bytes[0], self.ioff * row_bytes[1], 0]
            part = [self.IP * row_bytes[0], self.IP * row_bytes[1], size[2]]
            off = [0, _align(part[0]), _align(part[0] + part[1])]
            types.append(kind)
            offsets.append(off)
            sizes.append(part)
            tensor_sizes.append(size)
            slice_off.append(cut)
            stride = max(stride, off[2] + part[2])
        self.slot_bytes = _align(stride)
        dev = torch.device('cuda', torch.cuda.current_device())
        self.arena = torch.empty((self.S, self.slot_bytes), dtype=torch.uint8, device=dev)
        self.arena16 = self.arena.view(torch.uint16)
        self.types = torch.tensor(types, dtype=torch.int32, device=dev)
        self.offsets = offsets                       # the host copy (the uploads)
        self.offs = torch.tensor(offsets, dtype=torch.int32, device=dev)
        self.sizes = sizes                           # the bytes of every expert's three tensors (the rank's slice)
        self.tensor_sizes = tensor_sizes             # the bytes of the whole tensors (the checkpoint's)
        self.slice_off = slice_off                   # where the slice starts inside each tensor
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
        self.owner = _glob['owner'][self.index]     # slot -> expert (the uploaded slots follow the hot ones)
        # the slot space: the hot experts and the ones a call uploads (their slots are after the hot ones)
        self.upload_slots = int(os.environ.get('XTIER_GSQ_PREFILL_UPLOAD', '64'))
        self.used = torch.empty(max(self.S + self.upload_slots, 1), dtype=torch.int32, device=dev)
        self.rank = torch.empty(max(self.S + self.upload_slots, 1), dtype=torch.int32, device=dev)
        self.resident = [-1] * self.S
        self.stat = [0, 0, 0, 0]    # the residency counters (routed, hits, uploaded, bytes)
        self.upload_bytes = [0, 0]  # the uploaded bytes: the pinned RAM and the checkpoint's files
        self.routes = collections.Counter()   # the routed experts (the diagnostic)
        self.calls = 0                # this layer's decode calls (the residency log)
        self.gpu_ms = 0.0             # the GPU's kernel time (XTIER_TIMING)
        self.wall_ms = 0.0            # the wall time of the calls (XTIER_TIMING)
        self.up_ms = 0.0              # the uploads' wall time (XTIER_TIMING)
        self.sync_ms = 0.0            # the wait for the routed ids (XTIER_TIMING)
        self.copy_ms = 0.0            # the copies themselves (XTIER_TIMING)
        self.pre_ms = 0.0             # the routed experts' selection (XTIER_TIMING)
        self.window_calls = 0         # the calls in the timing window (XTIER_TIMING)
        self._pending = []            # the events of the calls (the GPU's kernel time, read at the log point)
        self.last_call = time.perf_counter()
        self.filled = False
        self.scratch = _scratch(self.H, self.IP, self.dtype, dev)
        self.pinned = []          # the RAM slots registered with CUDA (pin_slots)
        self.uploaded = torch.full((self.E,), -1, dtype=torch.int32,
                                   device=dev)   # the experts uploaded for the current call (their slot)
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
                for ri, (role, off, want) in enumerate(zip(ROLES, self.offsets[e], self.sizes[e])):
                    address, size, _ = self.pack.expert_bytes(role, self.layer_id, e)
                    if size != self.tensor_sizes[e][ri]:
                        raise ValueError(f'xtier cache: the expert {e} of the layer {self.layer_id} has {size} bytes '
                                         f'of {role}, the layout says {self.tensor_sizes[e][ri]}')
                    address += self.slice_off[e][ri]   # the rank's slice of the tensor
                    dst = self.arena.data_ptr() + j * self.slot_bytes + off
                    _check(cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(address), C.c_size_t(want), 1,
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
        # the adaptive policy starts once the decode is running outside a capture: this path has no CUDA graphs
        # (the launcher turns them off on this card), so the swaps can start with the first decode call
        _glob['captured'] = True
        _calls[0] += 1
        self.calls += 1
        self.window_calls += 1
        if self.calls % 64 == 1:
            logger.info('xtier cache: layer %d: %d of %d experts in the VRAM (the calls %d, the table %s); '
                        'the last %d calls routed %d experts, %d were in the VRAM and %d were uploaded '
                        '(%.1f MiB over PCIe: %.1f from the pinned RAM, %.1f from the files)',
                        self.layer_id, int((self.table >= 0).sum()), self.E, self.calls,
                        self.table[self.table >= 0][:6].tolist(), self.calls - 1, self.stat[0], self.stat[1],
                        self.stat[2], self.stat[3] / 2 ** 20, self.upload_bytes[0] / 2 ** 20, self.upload_bytes[1] / 2 ** 20)
            if config.timing():
                n = max(self.window_calls, 1)
                for ev0, ev1 in self._pending:   # the GPU's kernel time of the calls (the events have landed)
                    ev1.synchronize()
                    self.gpu_ms += ev0.elapsed_time(ev1)
                self._pending.clear()
                logger.info('xtier timing: layer %d: the last %d calls: the GPU %.2f ms a call, the wall %.2f ms a '
                            'call, the uploads %.2f ms a call (the selection %.2f, the ids %.2f, the copies %.2f)',
                            self.layer_id, n, self.gpu_ms / n, self.wall_ms / n, self.up_ms / n, self.pre_ms / n,
                            self.sync_ms / n, self.copy_ms / n)
                self.gpu_ms = self.wall_ms = self.up_ms = 0.0
                self.sync_ms = self.copy_ms = self.pre_ms = 0.0
                self.window_calls = 0
            self.stat = [0, 0, 0, 0]
            self.upload_bytes = [0, 0]
        if os.environ.get('XTIER_ROUTE_STATS') and self.calls % 16 == 0:
            counts = sorted(self.routes.values(), reverse=True)
            total = sum(counts)
            top = sum(counts[:64])
            logger.info('xtier routes: layer %d: %d distinct experts, %d routes, the top 64 take %.1f%%, '
                        'the top 14 take %.1f%%', self.layer_id, len(counts), total,
                        100.0 * top / max(total, 1), 100.0 * sum(counts[:14]) / max(total, 1))
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
            triton.next_power_of_2(max(self.S + self.upload_slots, 1)), BLOCK,
            self.uploaded, self.S, self.S + self.upload_slots)

    def _kernels(self, m, k, x, routing, gout):
        """The GPU part: the gate/up and the down over the used slots, then the per-token sum of the GPU routes."""
        if self.tables is None:
            self.tables = tables(torch.device('cuda', torch.cuda.current_device()))
        t = self.tables
        sc = self.scratch
        H, I = self.H, self.I
        nk = m * k
        scratch = _scratch_arena(self, int(os.environ.get('XTIER_GSQ_PREFILL_UPLOAD', '64')))
        arena2 = scratch if scratch is not None else self.arena
        arena16_2 = arena2.view(torch.uint16)
        gate_up[(nk, self.IP // BN)](
            Pointer(x, self.dtype), sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.types, self.offs,
            sc['mid'], sc['gate'], m, H, I, self.limit, self.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'],
            t['ksigns'], MAX_M, BN, ALPHA=self.alpha, BETA=self.beta, DT=_tl_dtype(self.dtype), num_warps=4,
            num_stages=2, HOT=self.S, ARENA2=arena2, ARENA16_2=arena16_2, ROLE=0, IOFF=self.ioff, IP=self.IP)
        gate_up[(nk, self.IP // BN)](
            Pointer(x, self.dtype), sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.types, self.offs,
            sc['mid'], sc['gate'], m, H, I, self.limit, self.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'],
            t['ksigns'], MAX_M, BN, ALPHA=self.alpha, BETA=self.beta, DT=_tl_dtype(self.dtype), num_warps=4,
            num_stages=2, HOT=self.S, ARENA2=arena2, ARENA16_2=arena16_2, ROLE=1, IOFF=self.ioff, IP=self.IP)
        down[(nk, H // BN)](
            sc['mid'], sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.types, self.offs, sc['part'],
            m, H, I, self.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'], t['ksigns'], MAX_M, BN,
            DT=_tl_dtype(self.dtype), num_warps=4, num_stages=2,
            HOT=self.S, ARENA2=arena2, ARENA16_2=arena16_2, IOFF=self.ioff, IP=self.IP)
        _combine[(m, triton.cdiv(H, 1024))](sc['part'], sc['rslot'], Pointer(routing, torch.float32), gout, H, k,
                                            MAX_M, 1024)

    def _upload_routed(self, ids):
        """The experts this call routes to that are not in the VRAM: uploaded over PCIe into the scratch arena, so
        the GPU decodes them and the engine does not see them.

        The bytes come from the engine's RAM (the cold pool's resident set) when the expert is there, else from the
        checkpoint's files. The upload is what makes the decode GPU-first: the card decodes the IQ blocks about a
        hundred times faster than the engine's AVX2 kernels.
        """
        t0 = time.perf_counter()
        self.uploaded.fill_(-1)
        flat = ids.reshape(-1)
        flat = flat[flat >= 0]
        if flat.numel() == 0:
            return 0
        uniq = torch.unique(flat)
        want = uniq[self.table[uniq] < 0]         # the ones not in the VRAM
        t_sync = time.perf_counter()
        if config.timing():
            self.pre_ms += (t_sync - t0) * 1e3
        if os.environ.get('XTIER_ROUTE_STATS'):
            self.routes.update(uniq.tolist())
            if self.calls == 16:
                import json
                with open(f'/tmp/routes_{os.environ["XTIER_ROUTE_STATS"]}_{self.layer_id}.json', 'w') as f:
                    json.dump(sorted(self.routes), f)
        self.stat[0] += int(uniq.numel())
        self.stat[1] += int(uniq.numel()) - int(want.numel())
        if want.numel() == 0:
            return 0
        slots = int(os.environ.get('XTIER_GSQ_PREFILL_UPLOAD', '64'))
        arena = _scratch_arena(self, slots)
        if arena is None:
            return 0
        take = min(int(want.numel()), int(arena.shape[0]))
        experts = want[:take].tolist()
        if config.timing():
            self.sync_ms += (time.perf_counter() - t_sync) * 1e3
        # The cold pool: the experts this call routes to are read into the engine's RAM (the coldest resident
        # leaves), so the upload reads their bytes from the pinned RAM instead of the checkpoint's files, and the
        # RAM's contents follow the routes the GPU is really asked for.
        plan = self.mailbox.plan
        lib = getattr(plan, 'lib', None)
        if lib is not None and hasattr(lib, 'xtier_gsq_cold_resident'):
            wanted = (C.c_int32 * len(experts))(*experts)
            lib.xtier_gsq_cold_resident(plan.plan, 1, len(experts), wanted)
        _upload(self, self.mailbox.plan, experts, arena, torch.cuda.current_stream())
        if config.timing():
            self.copy_ms += (time.perf_counter() - t_sync) * 1e3 - self.sync_ms
        self.uploaded[want[:take]] = torch.arange(take, dtype=torch.int32, device=ids.device)
        # the packing kernel reads the expert of every slot from the owner table: the uploaded slots follow the hot ones
        self.owner[self.S:self.S + take] = want[:take]
        # the counters of the residency: the uploaded experts and the bytes they cost
        self.stat[2] += take
        self.stat[3] += sum(self.sizes[e][r] for e in experts for r in range(3))
        return take

    def decode(self, stream, m, k, x, ids, routing, out, act=0, dtype=None, ids_tensor=None):
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
        if ids_tensor is not None:
            t0 = time.perf_counter()
            self._upload_routed(ids_tensor)
            self.up_ms += (time.perf_counter() - t0) * 1e3
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
        if config.timing():
            self._ev0 = torch.cuda.Event(enable_timing=True)
            self._ev0.record()
        self._kernels(m, k, x, routing, sc['gout'])
        if config.timing():
            self._ev1 = torch.cuda.Event(enable_timing=True)
            self._ev1.record()
            self._pending.append((self._ev0, self._ev1))
        self.mailbox.signal(stream, m, k, 2)
        _unpack[(triton.cdiv(m * self.H, BLOCK),)](Pointer(sout, torch.float32), sc['gout'], Pointer(out, torch.float32),
                                                   m * self.H, BLOCK)
        if config.timing():
            # the wall time of the call (the uploads are synchronous, so the wall time carries them). The GPU's
            # kernel time is read at the log point: waiting for the events here would drain the pipeline.
            now = time.perf_counter()
            self.wall_ms += (now - self.last_call) * 1e3
            self.last_call = now


_calls = [0]             # the decode calls (the residency log)
_scratch_arenas = {}      # (device, stride) -> the arena the prefills upload the experts into
_pinned = {}              # (device, size) -> the pinned buffers the upload stages the bytes in
_copy_streams = {}        # device -> the stream the uploads copy on (the decode stream is not drained by them)


def _copy_stream(device):
    """The stream the uploads copy on: the decode stream waits for the copies with an event, so the copies do not
    drain the decode pipeline (a synchronization per expert drained it sixteen times a layer)."""
    key = torch.device(device)
    stream = _copy_streams.get(key)
    if stream is None:
        stream = _copy_streams[key] = torch.cuda.Stream(device=key)
    return stream


def _stage(device, size):
    """A pinned buffer to stage the bytes of an expert in, and the event that says when its copy has landed (the
    buffer is reused after that). A ring of them: the staging of the next expert overlaps the copy of this one."""
    key = (torch.device(device), size)
    ring = _pinned.get(key)
    if ring is None:
        ring = _pinned[key] = []
    # a free buffer: its copy has landed (an event that was never recorded is complete)
    for i, (buf, done) in enumerate(ring):
        if done.query():
            ring.append(ring.pop(i))
            return buf, done
    if len(ring) < 4:
        buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        done = torch.cuda.Event()
        ring.append((buf, done))
        return buf, done
    # every buffer is busy: wait for the oldest one and reuse it
    buf, done = ring.pop(0)
    done.synchronize()
    ring.append((buf, done))
    return buf, done


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
    cold pool's resident set) else the checkpoint's files (the pack's mapping).

    This is the batched prefill's trade: the bytes of every expert the batch routes to are moved once and decoded on
    the GPU, instead of being read and decoded by the CPU engine. The cost is the transfer (PCIe), the gain is the
    GPU's decode (about a thousand times faster than the engine's).

    The bytes are staged in pinned memory first: a copy from a mapped file (the pack's mapping) is a pageable one,
    and the driver stages those in small chunks (about a tenth of the bandwidth).

    The copies run on their own stream: the decode stream is not drained by them (a synchronization per expert
    drains the whole decode pipeline, and the decode calls it sixteen times a layer). The caller's stream waits for
    the copies with an event, and the copy stream waits for the caller's stream before it overwrites the arena.
    """
    cudart = _runtime()
    cudart.cudaMemcpyAsync.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
    cudart.cudaMemcpyAsync.restype = C.c_int
    lib = getattr(plan, 'lib', None)
    handle = getattr(plan, 'plan', None)
    copy = _copy_stream(pool.arena.device)
    # the arena's slots are reused by the next call: the copies wait for the decode that reads them
    copy.wait_stream(stream)
    failed = set()
    with torch.cuda.stream(copy):
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
                for ri, (role, off, want) in enumerate(zip(ROLES, pool.offsets[e], pool.sizes[e])):
                    if sources is not None:
                        address, size = sources[ri]
                    else:
                        address, size, _ = pool.pack.expert_bytes(role, pool.layer_id, e)
                    if size != pool.tensor_sizes[e][ri]:
                        raise ValueError(f'xtier cache: the expert {e} of the layer {pool.layer_id} has {size} bytes of '
                                         f'{role}, the layout says {pool.tensor_sizes[e][ri]}')
                    # the rank's slice of the tensor (the rows of the gate and the up)
                    address += pool.slice_off[e][ri]
                    dst = arena.data_ptr() + j * pool.slot_bytes + off
                    pinned = base is not None and any(b <= address < b + n for b, n in pool.pinned)
                    pool.upload_bytes[0 if pinned else 1] += want
                    try:
                        if pinned:
                            # the RAM slot is registered with CUDA: the driver copies it straight to the card
                            rc = cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(address), C.c_size_t(want), 1,
                                                        C.c_void_p(copy.cuda_stream))
                            if rc != 0:
                                cudart.cudaGetLastError()   # clear it: the staged copy below may still work
                                pinned = False
                        if not pinned:
                            staged, done = _stage(pool.arena.device, want)
                            C.memmove(staged.data_ptr(), address, want)     # the bytes -> the pinned buffer
                            _check(cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(staged.data_ptr()),
                                                         C.c_size_t(want), 1,
                                                         C.c_void_p(copy.cuda_stream)), 'cudaMemcpyAsync')
                            done.record(copy)   # the buffer is free when the copy has landed
                    except Exception as exc:   # noqa: BLE001  a failed upload must not take the server down
                        cudart.cudaGetLastError()
                        logger.warning('xtier cache: the upload of the expert %d of the layer %d failed (%s): the '
                                       'engine computes its routes', e, pool.layer_id, exc)
                        failed.add(e)
            finally:
                if ram >= 0:
                    lib.xtier_slot_unpin(handle, ram)
    # the decode stream waits for the copies (an event, not a drain of the whole pipeline)
    landed = torch.cuda.Event()
    landed.record(copy)
    stream.wait_event(landed)
    return failed


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
                mid = torch.empty((int(order_part.numel()), pool.IP), dtype=pool.dtype, device=x.device)
                gate_buf = torch.empty((int(order_part.numel()), pool.IP), dtype=torch.float32, device=x.device)
                nt = len(tiles)
                gate_up_tiles[(nt, pool.IP // bn)](
                    x, order_part, meta[0], meta[1], meta[2], meta[3], arena_part, arena_part.view(torch.uint16),
                    pool.types, pool.offs, mid, gate_buf, k, H, I, pool.limit, pool.slot_bytes, t['iq1s'], t['iq2xs'],
                    t['iq2xxs'], t['ksigns'], bm, bn, ALPHA=pool.alpha, BETA=pool.beta, DT=_tl_dtype(pool.dtype),
                    num_warps=4, num_stages=2, ROLE=0, IOFF=pool.ioff, IP=pool.IP)
                gate_up_tiles[(nt, pool.IP // bn)](
                    x, order_part, meta[0], meta[1], meta[2], meta[3], arena_part, arena_part.view(torch.uint16),
                    pool.types, pool.offs, mid, gate_buf, k, H, I, pool.limit, pool.slot_bytes, t['iq1s'], t['iq2xs'],
                    t['iq2xxs'], t['ksigns'], bm, bn, ALPHA=pool.alpha, BETA=pool.beta, DT=_tl_dtype(pool.dtype),
                    num_warps=4, num_stages=2, ROLE=1, IOFF=pool.ioff, IP=pool.IP)
                down_tiles[(nt, H // bn)](
                    mid, order_part, meta[0], meta[1], meta[2], meta[3], arena_part, arena_part.view(torch.uint16),
                    pool.types, pool.offs, routes, H, I, pool.slot_bytes, t['iq1s'], t['iq2xs'], t['iq2xxs'],
                    t['ksigns'], bm, bn, DT=_tl_dtype(pool.dtype), num_warps=4, num_stages=2,
                    IOFF=pool.ioff, IP=pool.IP)
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
    stream.synchronize()   # the experts' bytes are in the VRAM before the first decode reads them
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
            # the experts' bytes in the checkpoint's own mapping: a swap of an expert that is not in the engine's RAM
            # (a cold expert) reads them from there. The source offsets say where the rank's slice starts in each
            # tensor (the rows of the gate and the up).
            sources = (C.c_void_p * (pool.E * 3))()
            cuts = (C.c_int64 * (pool.E * 3))()
            for e in range(pool.E):
                for r, role in enumerate(ROLES):
                    address, size, _ = pool.pack.expert_bytes(role, pool.layer_id, e)
                    if size != pool.tensor_sizes[e][r]:
                        raise ValueError(f'xtier cache: the expert {e} of the layer {pool.layer_id} has {size} bytes '
                                         f'of {role}, the layout says {pool.tensor_sizes[e][r]}')
                    sources[e * 3 + r] = C.c_void_p(address)
                    cuts[e * 3 + r] = pool.slice_off[e][r]
            if lib.xtier_mailbox_pool_attach_gsq(pool.mailbox.bridge, pool.index, pool.S, pool.S, pool.E,
                                                 C.c_void_p(pool.arena.data_ptr()), C.c_size_t(pool.slot_bytes),
                                                 offsets, sizes, resident, sources, cuts) < 0:
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
