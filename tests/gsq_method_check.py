"""The GSQ layer through the backend's glue (xtier/backend/method.py): the pack, the CPU engine, the GPU's hot
expert cache and the cold pool, as the server runs them.

  python tests/gsq_method_check.py

The server builds the layer's method (create_weights), hands the experts over (_convert_gsq: the engine's plan, the
mailbox, the GPU cache, the cold pool), fills the caches with the other layers (_register_all) and decodes
(_decode). This test runs that sequence on a synthetic pack (a real file on disk: the cold pool reads it with
O_DIRECT) and checks the layer's output against the engine's own.

The SGLang parts are the ones the glue touches: the MoE method's class (the format's detection), the layer's runner
config and the tensor-parallel rank.
"""
import ctypes as C
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


E, H, I, K, M = 10, 256, 256, 1, 8        # the engine's minimum is 10 experts; H and I: multiples of 256
THREADS = 4
CACHED = 4                                # the GPU cache's slots (the pack's usage profile is empty: the order)
BLOCK = {16: 66, 17: 74, 19: 50, 29: 56}     # the bytes of one block, per type (csrc/engine/gsq.h)
ROW = {t: H // 256 * b for t, b in BLOCK.items()}
ROW_DOWN = {t: I // 256 * b for t, b in BLOCK.items()}
types = np.array([29, 17, 16, 19, 29, 17, 16, 19, 29, 16], dtype=np.uint8)         # the gate and the up
types_down = np.array([16, 19, 29, 17, 16, 19, 29, 17, 16, 19], dtype=np.uint8)     # the down
NAMES = {16: 'IQ2_XXS', 17: 'IQ2_XS', 19: 'IQ1_S', 29: 'IQ1_M'}
TOL = 5e-3

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / 'model'
    pack_dir = model / 'maya-ds4-gsq'
    pack_dir.mkdir(parents=True)
    # the checkpoint file: the experts' bytes (the store reads them with O_DIRECT, the pack maps them)
    size = sum(ROW.get(int(types[e]), 0) * I for e in range(E)) * 2 + \
        sum(ROW_DOWN.get(int(types_down[e]), 0) * H for e in range(E))
    raw = bytearray(np.random.default_rng(7).integers(0, 256, size, dtype=np.uint8).tobytes())
    # the scale fields of the blocks: a finite fp16 (random bytes would give NaNs). The scale is 2^-8, so the weights
    # are O(1) and the dots stay in the FP16 range.
    d16 = np.frombuffer(np.float16(2 ** -8).tobytes(), dtype=np.uint8)
    iq1m = np.frombuffer(np.uint16([0x0000, 0x0000, 0xC000, 0x1000]).tobytes(), dtype=np.uint8)
    offsets = {}
    at = 0
    for role in ('w1', 'w3', 'w2'):
        for e in range(E):
            t = int(types_down[e]) if role == 'w2' else int(types[e])
            rows = H if role == 'w2' else I
            row_bytes = ROW_DOWN[t] if role == 'w2' else ROW[t]
            block = BLOCK[t]
            for i in range(rows * (row_bytes // block)):
                off = at + i * block
                if t == 29:   # IQ1_M: the scales at 48 (the block's d comes from their high nibbles)
                    raw[off + 48:off + 56] = iq1m.tobytes()
                else:        # the fp16 d at the start of every block
                    raw[off:off + 2] = d16.tobytes()
            offsets[(role, e)] = at
            at += row_bytes * rows
    (model / 'model.gguf').write_bytes(bytes(raw))
    (pack_dir / 'ds4.txt').write_text(
        f'vocab_size 129280\nhidden_size {H}\nmoe_intermediate_size {I}\nnum_hidden_layers 2\n'
        f'n_routed_experts {E}\nnum_experts_per_tok 6\nformat gguf\nmodel_dir {model}\n')
    lines = ['file 0 model.gguf']
    for e in range(E):
        for role, rows, cols in (('w1', I, H), ('w3', I, H), ('w2', H, I)):
            t = int(types_down[e]) if role == 'w2' else int(types[e])
            row_bytes = ROW_DOWN[t] if role == 'w2' else ROW[t]
            lines.append(f't layers.0.ffn.experts.{e}.{role}.weight {NAMES[t]} 0 '
                         f'{offsets[(role, e)]} {row_bytes * rows} {rows} {cols}')
    (pack_dir / 'tensors.txt').write_text('\n'.join(lines) + '\n')
    os.environ['XTIER_GSQ_PACK'] = str(pack_dir)
    os.environ['XTIER_CACHE_SLOTS'] = str(CACHED)
    os.environ['XTIER_COLD'] = '1'
    os.environ['XTIER_COLD_FRACTION'] = '0.6'      # 6 of the 10 experts stay on the drive (4 in RAM)

    from xtier.backend.method import LayerConfig, XTierMoEMethod   # noqa: E402

    class GGUFMoEMethod:
        """SGLang's GGUF MoE method (its class name is how the format is detected)."""

    class RunnerConfig:
        activation = 'silu'
        is_gated = True
        apply_router_weight_on_input = False
        swiglu_limit = 10.0
        gemm1_alpha = None
        gemm1_beta = None
        routed_scaling_factor = None

    class Layer:
        moe_runner_config = RunnerConfig()
        num_fused_shared_experts = 0
        moe_ep_size = 1

    layer = Layer()
    method = XTierMoEMethod(GGUFMoEMethod(), LayerConfig(layer_id=0, threads=THREADS, cache_slots=CACHED))
    method.create_weights(layer, E, H, I, torch.float16)
    check(method.gsq, 'the layer is a GSQ layer (the experts stay in the checkpoint file)')
    check(method in XTierMoEMethod.layers, 'the layer registered itself (its cache is filled with the other layers\')')

    method._convert_gsq(layer)
    check(method.plan is not None and method.mailbox is not None, 'the engine\'s plan and the mailbox are up')
    check(method.pool is not None, f'the GPU cache holds the layer\'s hot experts ({method.pool.S} slots)')
    check(method.pool.bytes() > 0, f'the GPU cache\'s arena is allocated ({method.pool.bytes()} bytes)')

    # the caches are filled with the layers' (the server's first MoE call)
    XTierMoEMethod._register_all()
    check(method.pool.filled, 'the GPU cache was filled (the hot experts are in the VRAM)')

    # the cold pool: the experts that are not in RAM are read from the checkpoint file on the drive
    table = method.plan.slot_table()
    check(table.count(-1) > 0, f'the experts that are not in RAM stay on the drive ({table.count(-1)} of {E})')

    # the layer's output: the GPU computes the cached experts, the engine the rest
    rng = np.random.default_rng(20261029)
    x = rng.integers(-256, 256, (M, H)).astype(np.float32) / 256.0
    # the routes: the GPU's experts (0..3) and one on the drive (8): the engine's slots hold 4 experts, so a call
    # must not name more distinct experts than that (the FP4 path's rule)
    ids = np.array([[e % 4 if e < 4 else 8] for e in range(M)], dtype=np.int32)
    rw = (0.5 + 0.5 * rng.random((M, K))).astype(np.float32)
    x16 = (C.c_uint16 * (M * H))(*(x.view(np.uint32) >> 16).astype(np.uint16).reshape(-1).tolist())
    ids32 = (C.c_int32 * (M * K))(*ids.reshape(-1).tolist())
    rw32 = (C.c_float * (M * K))(*rw.reshape(-1).tolist())
    out = (C.c_float * (M * H))()
    rc = method.plan.forward(M, K, C.addressof(x16), C.addressof(ids32), C.addressof(rw32), C.addressof(out))
    ref = np.frombuffer(out, dtype=np.float32).reshape(M, H).copy()
    check(rc >= 0, f'the engine computed the layer (rc {rc})')

    xd = torch.from_numpy(x).to(torch.float16).cuda()
    ids_d = torch.from_numpy(ids).int().cuda()
    rw_d = torch.from_numpy(rw).float().cuda()
    got = method._decode(xd, ids_d, rw_d, M, K).cpu().numpy()
    errs = [float(np.linalg.norm(got[t] - ref[t]) / max(np.linalg.norm(ref[t]), 1e-30)) for t in range(M)]
    check(max(errs) < TOL, f'the layer through the glue against the engine (relative L2 {max(errs):.2e}, '
                         f'tolerance {TOL:.0e})')
    loads = method.plan.cold_stats()[1]
    check(loads > 0, f'the experts that were not in RAM were read from the drive ({loads} reads)')

    # the server's call: apply() with the dispatcher's output (the decode path and the prefill path)
    class TopK:
        def __init__(self, ids, weights):
            self.topk_ids, self.topk_weights = ids, weights

    class DispatchOutput:
        def __init__(self, hidden_states, ids, weights):
            self.hidden_states, self.topk_output = hidden_states, TopK(ids, weights)

    def run(m):
        """The layer's output for m tokens through apply() (what the server calls)."""
        xd = torch.from_numpy(np.repeat(x[:1], m, axis=0)).to(torch.float16).cuda()
        ids_t = torch.from_numpy(np.repeat(ids[:1], m, axis=0)).int().cuda()
        rw_t = torch.from_numpy(np.repeat(rw[:1], m, axis=0)).float().cuda()
        out = method.apply(layer, DispatchOutput(xd, ids_t, rw_t))
        return out.hidden_states.float().cpu().numpy()

    got = run(4)      # the same token and route four times: the GPU's part alone (the expert is cached)
    want = np.repeat(ref[:1], 4, axis=0)
    err = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
    check(err < TOL, f'the decode through apply() against the engine (relative L2 {err:.2e})')
    got = run(128)    # more than the decode path takes (XTIER_DECODE_MAX): the prefill branch
    check(np.isfinite(got).all() and np.abs(got).sum() > 0,
          f'the prefill through apply() (the CPU engine does it all, {got.shape[0]} tokens)')
    check(method.calls['decode'] > 0 and method.calls['cpu_prefill'] > 0,
          f'the calls were counted ({method.calls})')

    # the adaptive residency: the GPU cache's set follows the routes. The expert below is in the engine's RAM (the
    # cold pool's resident set) and not in the VRAM: after a few windows of routes to it, the policy swaps it in.
    from xtier.cache import gsq_pool
    from xtier.cache.pool import _glob
    _glob['captured'] = True   # the policy starts once the decode graphs exist (this test captures none)
    gsq_pool.maybe_start()

    def run_to(expert, m):
        """m tokens, all routed to one expert."""
        xd = torch.from_numpy(np.repeat(x[:1], m, axis=0)).to(torch.float16).cuda()
        ids_t = torch.full((m, 1), expert, dtype=torch.int32).cuda()
        rw_t = torch.full((m, 1), 1.0, dtype=torch.float32).cuda()
        out = method.apply(layer, DispatchOutput(xd, ids_t, rw_t))
        return out.hidden_states.float().cpu().numpy()

    target = 4      # in the engine's RAM (the cold pool's resident set), not in the GPU cache
    check(method.pool.table[target].item() < 0, 'the expert is not in the GPU cache yet')
    for _ in range(32):
        run_to(target, 4)
    check(method.pool.table[target].item() >= 0,
          f'the expert was swapped into the GPU cache (slot {method.pool.table[target].item()})')
    got = run_to(target, 4)
    # the engine's own output for the same routes
    n_ref = 4
    ids_ref = (C.c_int32 * n_ref)(*([target] * n_ref))
    rw_ref = (C.c_float * n_ref)(*([1.0] * n_ref))
    out_ref = (C.c_float * (n_ref * H))()
    # the same activations the GPU's call used (the first token, four times)
    x_ref = (C.c_uint16 * (n_ref * H))(*(np.repeat(x[:1], n_ref, axis=0).view(np.uint32) >> 16).astype(np.uint16)
                                       .reshape(-1).tolist())
    rc = method.plan.forward(n_ref, 1, C.addressof(x_ref), C.addressof(ids_ref), C.addressof(rw_ref),
                             C.addressof(out_ref))
    want = np.frombuffer(out_ref, dtype=np.float32).reshape(n_ref, H).copy()
    err = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
    check(rc >= 0 and err < TOL, f'the swapped expert computes the same as the engine (relative L2 {err:.2e})')

    # a prefill that routes to more experts than the RAM slots hold: the engine runs it in passes (each expert read
    # from the drive once), and the result is the engine's own
    def run_many(m, k):
        """m tokens, k experts each: every expert of the layer is routed to (a prefill: the engine runs it in
        passes, each expert read from the drive once)."""
        xd = torch.from_numpy(np.repeat(x[:1], m, axis=0)).to(torch.float16).cuda()
        ids_t = torch.from_numpy(np.tile(np.arange(E, dtype=np.int32), (m, 1))[:, :k].copy()).int().cuda()
        rw_t = torch.from_numpy(np.full((m, k), 1.0 / k, dtype=np.float32)).cuda()
        out = method.apply(layer, DispatchOutput(xd, ids_t, rw_t))
        return out.hidden_states.float().cpu().numpy()

    plan = method.plan
    K2 = 6
    ids_all = np.tile(np.arange(E, dtype=np.int32), (128, 1))[:, :K2].copy()
    x16b = (C.c_uint16 * (128 * H))(*(np.repeat(x[:1], 128, axis=0).view(np.uint32) >> 16).astype(np.uint16)
                                        .reshape(-1).tolist())
    ids_c = (C.c_int32 * (128 * K2))(*ids_all.reshape(-1).tolist())
    rw_c = (C.c_float * (128 * K2))(*np.full(128 * K2, 1.0 / K2, dtype=np.float32).tolist())
    out = (C.c_float * (128 * H))()
    rc = plan.forward(128, K2, C.addressof(x16b), C.addressof(ids_c), C.addressof(rw_c), C.addressof(out))
    ref_all = np.frombuffer(out, dtype=np.float32).reshape(128, H).copy()
    got = run_many(128, 6)
    err = float(np.linalg.norm(got - ref_all) / max(np.linalg.norm(ref_all), 1e-30))
    check(rc >= 0 and err < TOL, f'the prefill over every expert against the engine (relative L2 {err:.2e})')

print('GSQ-METHOD-CHECK', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
