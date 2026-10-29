"""The hot expert cache of a GSQ (IQ-quantized GGUF) layer: the hot experts' bytes in the VRAM, decoded on the GPU.

The FP4 pool (xtier/cache/pool.py) keeps the hot experts' packed weights in the VRAM and decodes them in the kernel.
The GSQ experts are IQ blocks, so the same shape: the hot experts' bytes (as in the checkpoint) go into the VRAM and
the kernels decode the blocks they need (xtier/backend/gsq_gpu.py, the engine's decode).

The decode's arithmetic is FP32 (the block scales are FP16 and the index arithmetic must not round); the weights go
into the dot as FP16, like the FP4 pool's (its scales multiply in FP16 too).

The routing is the FP4 pool's: the packing kernel marks the routes of the cached experts for the CPU (-1) and gives
the GPU their slot, the GPU computes them and the CPU's output is the sum. The residency is static here (the hot
experts are chosen at load time; the FP4 pool's adaptive swaps copy from the engine's RAM shards, which the GSQ path
does not have: its experts live in the checkpoint file).
"""

import ctypes as C

import torch
import triton
import triton.language as tl

from xtier.backend.gsq_gpu import _iq_vals
from xtier.cache.pool import BLOCK, MAX_M, MAX_ROUTES, Profile, _combine, _pack, _scratch, _unpack
from xtier.backend.prefill import _check, _runtime

# The IQ block sizes (csrc/engine/gsq.h): the bytes of one 256-value block, per type.
BLOCK_BYTES = {16: 66, 17: 74, 19: 50, 29: 56}


@triton.jit
def _gsq_gate_up(X, SLOT_LIST, SLOT_EXP, ARENA, ARENA16, ETYPE, MID, M, H: tl.constexpr, I: tl.constexpr,
                 LIMIT, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0, DT: tl.constexpr = tl.float16):
    """The gate and the up of the cached expert: its IQ blocks decoded and multiplied by the tokens.

    The expert's gate is I rows of H values (H/256 blocks a row) and its up the same; the down is H rows of I values.
    """
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    typ = tl.load(ETYPE + e)
    bb = tl.where(typ == 16, 66, tl.where(typ == 17, 74, tl.where(typ == 19, 50, 56))).to(tl.int64)
    tok = tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    base = slot.to(tl.int64) * (GATE_CAP + UP_CAP + (H * (I // 256) * 74))
    rstride = (H // 256) * bb
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(H // 256):
        ke = kb * 256 + tl.arange(0, 256)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0).to(DT)
        bg = base + n.to(tl.int64) * rstride + kb * bb
        bu = base + GATE_CAP + n.to(tl.int64) * rstride + kb * bb
        wg = _iq_vals(typ, ARENA, ARENA16, bg, G1, G2, G3, KS, BN).to(DT)
        wu = _iq_vals(typ, ARENA, ARENA16, bu, G1, G2, G3, KS, BN).to(DT)
        accg = tl.dot(xe, tl.trans(wg), accg)
        accu = tl.dot(xe, tl.trans(wu), accu)
    g = accg
    u = accu
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
        val = g / (1.0 + tl.exp(-ALPHA * g)) * (u + BETA)
    else:
        val = g / (1.0 + tl.exp(-g)) * u
    tl.store(MID + (row * BM + tok[:, None]) * I + n[None, :], val.to(DT), mask=tok[:, None] < M)


@triton.jit
def _gsq_down(MID, SLOT_LIST, SLOT_EXP, ARENA, ARENA16, ETYPE, PART, M, H: tl.constexpr, I: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr,
               DT: tl.constexpr = tl.float16):
    """The down of the cached expert: its IQ blocks decoded and multiplied by the intermediate."""
    p = tl.program_id(0)
    slot = tl.load(SLOT_LIST + p)
    if slot < 0:
        return
    row = p
    e = tl.load(SLOT_EXP + p).to(tl.int64)
    typ = tl.load(ETYPE + e)
    bb = tl.where(typ == 16, 66, tl.where(typ == 17, 74, tl.where(typ == 19, 50, 56))).to(tl.int64)
    tok = tl.arange(0, BM)
    h = tl.program_id(1) * BN + tl.arange(0, BN)
    base = slot.to(tl.int64) * (GATE_CAP + UP_CAP + (H * (I // 256) * 74))
    dbase = base + GATE_CAP + UP_CAP
    rstride = (I // 256) * bb
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(I // 256):
        ke = kb * 256 + tl.arange(0, 256)
        xe = tl.load(MID + (row * BM + tok[:, None]) * I + ke[None, :], mask=tok[:, None] < M, other=0.0).to(DT)
        bd = dbase + h.to(tl.int64) * rstride + kb * bb
        w = _iq_vals(typ, ARENA, ARENA16, bd, G1, G2, G3, KS, BN).to(DT)
        acc = tl.dot(xe, tl.trans(w), acc)
    tl.store(PART + (row * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


class GsqPool:
    """The hot experts of one GSQ layer in the VRAM: their IQ bytes, decoded and multiplied by the GPU."""

    def __init__(self, plan, pack, layer_id, slots, mailbox, dtype, hidden, inter, experts, types):
        self.plan, self.pack, self.layer_id = plan, pack, layer_id
        self.E, self.H, self.I = int(experts), int(hidden), int(inter)
        self.S = min(int(slots), self.E)
        self.dtype = dtype
        self.mailbox = mailbox
        self.limit = float(plan.swiglu_limit)
        self.alpha, self.beta = float(plan.swiglu_alpha), float(plan.swiglu_beta)
        # the arena: the three matrices of every slot (the gate, the up, the down), each as big as the largest block
        # size of the layer's experts (the experts have different quant types)
        bb = max(BLOCK_BYTES[t] for t in types)
        self.gate_cap = self.I * (self.H // 256) * bb
        self.up_cap = self.gate_cap
        self.down_cap = self.H * (self.I // 256) * bb
        self.slot_bytes = self.gate_cap + self.up_cap + self.down_cap
        dev = torch.device('cuda', torch.cuda.current_device())
        self.arena = torch.empty((self.S, self.slot_bytes), dtype=torch.uint8, device=dev)
        self.arena16 = self.arena.view(torch.uint16)
        self.etype = torch.tensor(list(types), dtype=torch.int32, device=dev)
        self.table = torch.full((self.E,), -1, dtype=torch.int32, device=dev)   # expert -> slot
        self.owner = torch.zeros((self.E,), dtype=torch.int32, device=dev)      # slot -> expert
        self.used = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.rank = torch.empty(max(self.S, 1), dtype=torch.int32, device=dev)
        self.resident = [-1] * self.S
        self.filled = False
        self.scratch = _scratch(self.H, self.I, dtype, dev)
        self.tables = None

    def bytes(self):
        return self.arena[:self.S].numel()

    def fill(self, experts, stream):
        """Upload the hot experts' bytes into the slots 0.. and publish them (the residency is static)."""
        experts = [e for e in experts if 0 <= e < self.E][:self.S]
        cudart = _runtime()
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for j, e in enumerate(experts):
                for role, off in (('w1', 0), ('w3', self.gate_cap), ('w2', self.gate_cap + self.up_cap)):
                    arr = self.pack.expert_bytes(role, self.layer_id, e)
                    dst = self.arena.data_ptr() + j * self.slot_bytes + off
                    _check(cudart.cudaMemcpyAsync(C.c_void_p(dst), C.c_void_p(arr.ctypes.data), C.c_size_t(arr.nbytes),
                                                 1, C.c_void_p(stream)), 'cudaMemcpyAsync')
        table = torch.full((self.E,), -1, dtype=torch.int32)
        owner = torch.zeros((self.E,), dtype=torch.int32)
        for j, e in enumerate(experts):
            table[e] = j
            owner[j] = e
        self.table.copy_(table, non_blocking=False)
        self.owner.copy_(owner, non_blocking=False)
        self.resident = experts + [-1] * (self.S - len(experts))
        self.filled = True
        return len(experts)

    def _route(self, m, k, x, ids, routing, act, staging):
        sx, sids, sroute, sorig = staging
        sc = self.scratch
        nk = m * k
        _pack[(triton.cdiv(m * self.H, BLOCK),)](
            _Pointer(x, self.dtype), _Pointer(ids, torch.int32), _Pointer(routing, torch.int32),
            _Pointer(act or ids, torch.int32), self.table, _Pointer(sx, torch.bfloat16), _Pointer(sids, torch.int32),
            _Pointer(sroute, torch.int32), _Pointer(sorig, torch.int32), self.used, self.rank, sc['rslot'],
            sc['slot_list'], sc['slot_exp'], self.owner, m * self.H, nk, k, self.S, bool(act),
            triton.next_power_of_2(self.S), BLOCK)

    def _kernels(self, m, k, x, routing, gout):
        from xtier.backend.gsq_gpu import tables
        if self.tables is None:
            self.tables = tables(torch.device('cuda', torch.cuda.current_device()))
        t = self.tables
        sc = self.scratch
        H, I = self.H, self.I
        nk = m * k
        bn = 32
        bk = 256
        _gsq_gate_up[(nk, I // bn)](
            _Pointer(x, self.dtype), sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.etype,
            sc['mid'], m, H, I, self.limit, self.gate_cap, self.up_cap, MAX_M, bn,
            ALPHA=self.alpha, BETA=self.beta, DT=_tl_dtype(self.dtype), G1=t['iq1s'], G2=t['iq2xs'], G3=t['iq2xxs'],
            KS=t['ksigns'], num_warps=4, num_stages=2)
        _gsq_down[(nk, H // bn)](
            sc['mid'], sc['slot_list'], sc['slot_exp'], self.arena, self.arena16, self.etype, sc['part'], m, H, I,
            MAX_M, bn, self.gate_cap, self.up_cap, DT=_tl_dtype(self.dtype), G1=t['iq1s'], G2=t['iq2xs'], G3=t['iq2xxs'],
            KS=t['ksigns'], num_warps=4, num_stages=2)
        _combine[(m, triton.cdiv(H, 1024))](sc['part'], sc['rslot'], _Pointer(routing, torch.float32), gout,
                                            H, k, MAX_M, 1024)

    def decode(self, stream, m, k, x, ids, routing, out, act=0, dtype=None):
        """One decode call of up to 8 tokens on the current stream (device addresses; out: FP32 [m, H])."""
        sc = self.scratch
        sx, sids, sroute, sout, sorig = self.mailbox.slot(stream, m, k, with_routes=True)
        self._route(m, k, x, ids, routing, act, (sx, sids, sroute, sorig))
        self.mailbox.signal(stream, m, k, 1)
        self._kernels(m, k, x, routing, sc['gout'])
        self.mailbox.signal(stream, m, k, 2)
        _unpack[(triton.cdiv(m * self.H, BLOCK),)](_Pointer(sout, torch.float32), sc['gout'], _Pointer(out, torch.float32),
                                                   m * self.H, BLOCK)


class _Pointer:
    """A raw device address as a Triton pointer argument."""

    def __init__(self, address, dtype):
        self.address = address
        self.dtype = dtype

    def data_ptr(self):
        return self.address


def _tl_dtype(dtype):
    """The Triton dtype of a 16-bit float dtype: the kernels' dot operands."""
    return tl.float16 if dtype == torch.float16 else tl.bfloat16


def fill_all():
    """Fill the GSQ pools' slots with the hot experts (the profile's ranking, then the expert order)."""
    from xtier.cache import gsq_pool
    pools = [p for p in gsq_pool.GsqPool.layers if not p.filled]
    if not pools:
        return 0
    stream = torch.cuda.Stream()
    n = 0
    for pool in pools:
        ranking = Profile.ranking(pool.layer_id, pool.E)
        n += pool.fill(ranking or list(range(pool.E)), stream)
    return n
