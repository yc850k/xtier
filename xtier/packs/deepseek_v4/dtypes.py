"""The DeepSeek V4.1 model's dtypes on a card whose kernels do not all take the model's dtype.

The attention's low-rank output projection (``wo_a``) keeps its weight in BF16 whenever the FP8 path is off (the
weight's dtype is fixed at build time), and the DSA indexer and the KV compressor keep BF16 weights whatever the
model's dtype is. The grouped matmul and the plain linears take one dtype for both operands, so the pack casts the
activations to the weights' dtype (and the result back where the model's dtype has to be restored).

On a host where every one of them matches the model, the patch does nothing.
"""

import logging
import os

import torch

from xtier import config

logger = logging.getLogger(__name__)

MODELS = 'sglang.srt.models.deepseek_v4'
APPLY = f'{MODELS}._apply_wo_a_bf16_matmul'
RMS_NORM = 'sglang.srt.layers.layernorm.RMSNorm.forward_cuda'
MIX_STATS = 'sglang.srt.models.deepseek_v4_mhc.mix_stats'
DOWNGRADE = 'sglang.srt.model_executor.model_runner.maybe_downgrade_dtype_for_legacy_gpu'
MHC_NORM_CACHE = f'{MODELS}.DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache'
MHC_POST = 'sglang.srt.models.deepseek_v4_mhc.post'
MHC_COMBINE = 'sglang.srt.models.deepseek_v4_mhc.combine'
MLP_FWD = 'sglang.srt.models.deepseek_v2.DeepseekV2MLP.forward'
CLAMP = 'sglang.kernels.ops.moe.dsv4.silu_and_mul_clamp'
INDEXER = 'sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Indexer'
INDEXER_QUERIES = f'{INDEXER}.queries'
INDEXER_HEAD_WEIGHTS = f'{INDEXER}.head_weights_raw'
INDEXER_WK = f'{INDEXER}.forward_wk'
COMPRESSOR = 'sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Compressor'
COMPRESSOR_PROJECT = f'{COMPRESSOR}.project'
C4_INDEXER = 'sglang.srt.layers.attention.dsv4.indexer.C4Indexer'
C4_COMPUTE_Q = f'{C4_INDEXER}.compute_q'
C4_COMPUTE_WEIGHTS = f'{C4_INDEXER}.compute_weights'
LINEAR_BF16_FP32 = 'sglang.kernels.ops.gemm.bf16_fp32._linear_bf16_fp32_cublas'


def _apply_wo_a(original, o, wo_a, *args, **kwargs):
    """The wo_a matmul: the activations take the weight's dtype, the result the activations'."""
    if o.dtype == wo_a.dtype:
        return original(o, wo_a, *args, **kwargs)
    out = original(o.to(wo_a.dtype), wo_a, *args, **kwargs)
    if isinstance(out, torch.Tensor):
        return out.to(o.dtype)
    return out


def _indexer_bf16(original, self, *args, **kwargs):
    """The indexer's projections: the activations take the BF16 weights' dtype.

    The indexer (the DSA side attention) keeps its weights in BF16 upstream, whatever the model's dtype is. On a
    card that computes in FP16 the projections would mix the dtypes ("expected mat1 and mat2 to have the same
    dtype"), so the activations are cast; the index keys and the queries stay BF16, as the stored keys are.
    """
    if args and torch.is_tensor(args[0]) and args[0].dtype != torch.bfloat16:
        args = (args[0].to(torch.bfloat16),) + args[1:]
    return original(self, *args, **kwargs)


def _compressor_project(original, self, x, *args, **kwargs):
    """The compressor's ratio-1 projection: a plain Linear with a BF16 weight.

    The ratio-2 projection goes through ``linear_bf16_fp32``, which takes the activation in the model's dtype (its
    BF16 path needs a card with BF16 tensor cores), so only the ratio-1 path casts.
    """
    if self.compress_ratio == 1 and torch.is_tensor(x) and x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return original(self, x, *args, **kwargs)


def _linear_bf16_fp32(x, y):
    """The BF16 GEMM with an FP32 output, on a card without BF16 tensor cores.

    torch refuses that GEMM below sm_80 ("gemm input type at::BFloat16 and output type float is only supported for
    CUDA devices with compute capability 8.0 or higher"), so the operands are upcast: the same numbers (BF16 is
    exact in FP32), in FP32.
    """
    return torch.mm(x.float(), y.float().t())


def install(registry, hook_type):
    """Cast the activations around the wo_a matmul when the dtypes differ."""
    registry.register(APPLY, _apply_wo_a, hook_type.AROUND)
    registry.register(DOWNGRADE, _keep_dtype, hook_type.REPLACE)
    registry.register(MHC_NORM_CACHE, _mhc_norm_cache, hook_type.AROUND)
    registry.register(INDEXER_QUERIES, _indexer_bf16, hook_type.AROUND)
    registry.register(INDEXER_HEAD_WEIGHTS, _indexer_bf16, hook_type.AROUND)
    registry.register(INDEXER_WK, _indexer_bf16, hook_type.AROUND)
    registry.register(COMPRESSOR_PROJECT, _compressor_project, hook_type.AROUND)
    registry.register(C4_COMPUTE_Q, _indexer_bf16, hook_type.AROUND)
    registry.register(C4_COMPUTE_WEIGHTS, _indexer_bf16, hook_type.AROUND)
    from xtier.packs import supported

    if not supported((8, 0), 'the BF16 GEMM with an FP32 output (deepseek_v4)', 'a BF16 GEMM in FP32'):
        registry.register(LINEAR_BF16_FP32, _linear_bf16_fp32, hook_type.REPLACE)
    if not supported((8, 0), 'the FP16 hyper-connection residual stream (deepseek_v4)', 'an FP32 residual stream'):
        registry.register(MHC_POST, _post_fp32, hook_type.AROUND)
        registry.register(MHC_COMBINE, _combine_fp16, hook_type.AROUND)
        registry.register(RMS_NORM, _norm_fp32, hook_type.AROUND)
    if config.debug():
        registry.register(MIX_STATS, _debug_mix, hook_type.AROUND)
        registry.register(MLP_FWD, _debug_mlp, hook_type.AROUND)
        registry.register(CLAMP, _debug_clamp, hook_type.AROUND)
    logger.info('xtier: the DeepSeek V4.1 wo_a projection casts between the activation and the weight dtype')
    return True


def _keep_dtype(*args, **kwargs):
    """Keep the requested dtype on a card without BF16 tensor cores.

    SGLang downgrades the model to FP16 below sm_80 ("due to lack of bfloat16 support"). The
    hyper-connection needs BF16's range: its residual stream carries the layer outputs and the
    mixed coefficients overflow FP16's 65504, so a model that asks for BF16 keeps it (emulated on
    Turing) and every kernel of the model sees the dtype it was written for.
    """
    return None


def _mhc_norm_cache(original, self, *args, **kwargs):
    """The MHC's cached norm weights, in the model's dtype.

    The model caches them in BF16 (the fused MHC path wants BF16); when the model runs in another dtype the cache
    must hold that dtype, or the fused path would multiply the activations by weights of another dtype.
    """
    out = original(self, *args, **kwargs)
    norm = getattr(self, 'input_layernorm', None)
    weight = getattr(norm, 'weight', None)
    if not torch.is_tensor(weight):
        return out
    for name in ('_input_layernorm_weight_bf16', '_post_attention_layernorm_weight_bf16'):
        w = getattr(self, name, None)
        if torch.is_tensor(w) and w.dtype != weight.dtype:
            setattr(self, name, w.to(weight.dtype))
    return out


_norm_dtype = {}   # the norm's id -> the model's dtype (its weight's dtype)


def _norm_fp32(original, self, x, *args, **kwargs):
    """The RMSNorm with an FP32 input takes the torch path: the fused kernel is 16-bit only.

    The MHC's combine reads the FP32 residual stream, and the fused rmsnorm kernel ("failed to dispatch data type
    Float") only takes FP16 and BF16.
    """
    if x.dtype == torch.float32:
        out = self.forward_native(x, *args, **kwargs)
        if config.debug():
            logger.error('xtier debug: the norm: the input not finite %s the output not finite %s the dtypes %s %s',
                         int((~torch.isfinite(x.float())).sum()), int((~torch.isfinite(out.float())).sum()),
                         x.dtype, out.dtype)
        return out
    return original(self, x, *args, **kwargs)


def _post_fp32(original, cfg, x, residual, post_mix, comb):
    """The MHC's post with the residual stream in FP32.

    The stream carries the layer outputs and grows with the depth (284672 over the 40 layers, past FP16's 65504),
    so it is kept in FP32: the sublayer's output is cast up for the arithmetic and SGLang's own cast back to the
    activation's dtype would overflow.
    """
    out = original(cfg, x.float(), residual, post_mix, comb)
    if config.debug():
        names = ('x', 'residual', 'post_mix', 'comb')
        counts = []
        for i, name in enumerate(names, start=1):
            tensor = (x, residual, post_mix, comb)[i - 1]
            counts.append('%s %s' % (name, int((~torch.isfinite(tensor.float())).sum())))
        logger.error('xtier debug: the post: %s the out %s the out dtype %s',
                     ' '.join(counts), int((~torch.isfinite(out.float())).sum()), out.dtype)
    return out


def _combine_fp16(original, hc, state, quantized=None):
    """The MHC's combine: the residual's rows are FP32, the sublayer's input is the model's dtype."""
    out = original(hc, state, quantized)
    weight = getattr(getattr(hc, 'norm', None), 'weight', None)
    dtype = weight.dtype if torch.is_tensor(weight) else out.dtype
    if out.dtype != dtype:
        out = out.to(dtype)
    if config.debug():
        logger.error('xtier debug: the combine: the rows %s the out %s the dtype %s',
                     int((~torch.isfinite(state.residual.float())).sum()),
                     int((~torch.isfinite(out.float())).sum()), out.dtype)
    return out


def _debug_mlp(original, self, *args, **kwargs):
    """The shared expert's MLP: the weights once, then the input and the output."""
    x = args[0] if args else kwargs.get('x')
    first = id(self) not in _MLP_SEEN
    if first:
        _MLP_SEEN.add(id(self))
        for name in ('gate_up_proj', 'down_proj'):
            mod = getattr(self, name, None)
            weight = getattr(mod, 'weight', None)
            if torch.is_tensor(weight):
                logger.error('xtier debug: the shared MLP %s: the weight %s', name, _stat(weight))
    out = original(self, *args, **kwargs)
    if first or _bad(x) or _bad(out):
        logger.error('xtier debug: the shared MLP: the input %s the output %s', _stat(x), _stat(out))
    return out


def _debug_clamp(original, *args, **kwargs):
    """The clamped SwiGLU: the input, the output buffer it writes and the limit."""
    inp = args[0] if args else None
    out = original(*args, **kwargs)
    logger.error('xtier debug: the clamp: the input %s the output %s the limit %s',
                 _stat(inp), _stat(args[1] if len(args) > 1 else None), args[2] if len(args) > 2 else None)
    return out
