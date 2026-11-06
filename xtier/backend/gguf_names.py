"""The GGUF tensor names of DeepSeek V4.1 -> the model's parameter names.

SGLang's GGUF loader derives the map from the architecture names the ``gguf`` package knows (llama.cpp's) and from
a Transformers dummy model. Neither knows DeepSeek V4.1: the package has no ``deepseek41`` architecture and
Transformers has no ``deepseek_v41`` model. This builder supplies the map, the way SGLang's own
``gguf_name_maps`` does for the architectures outside Transformers.

The checkpoint is a Maya pack of DeepSeek V4.1 (the routed experts in llama.cpp's IQ quants): its shards hold the
llama.cpp names (``blk.N.ffn_gate_exps.weight`` and the rest), and the model wants the SGLang parameter names.
The routed experts are not in the map: XTier's GSQ path reads them from the checkpoint file (the engine's IQ decode),
so the model must not hold them.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The layer tensors: the GGUF (llama.cpp) name -> the model's parameter name (without the layer index).
LAYER = {
    'attn_norm': 'input_layernorm.weight',
    'ffn_norm': 'post_attention_layernorm.weight',
    'attn_sinks': 'self_attn.attn_sink',
    'attn_q_a_norm': 'self_attn.q_norm.weight',
    'attn_kv_a_norm': 'self_attn.kv_norm.weight',
    'attn_q_b': 'self_attn.wq_b.weight',
    'attn_output_a': 'self_attn.wo_a.weight',
    'attn_output_b': 'self_attn.wo_b.weight',
    'hc_attn_fn': 'hc_attn_fn',
    'hc_ffn_fn': 'hc_ffn_fn',
    'hc_attn_base': 'hc_attn_base',
    'hc_ffn_base': 'hc_ffn_base',
    'hc_attn_scale': 'hc_attn_scale',
    'attn_compressor_norm': 'self_attn.compressor.norm.weight',
    'attn_compressor_kv': 'self_attn.compressor.wkv.weight',
    'attn_compressor_gate': 'self_attn.compressor.wgate.weight',
    'hc_ffn_scale': 'hc_ffn_scale',
    'ffn_gate_inp': 'mlp.gate.weight',
    # the shared expert: the gate and the up are stacked into the model's fused gate_up projection
    'ffn_gate_shexp': 'mlp.shared_experts.gate_proj.weight',
    'ffn_up_shexp': 'mlp.shared_experts.up_proj.weight',
    'ffn_down_shexp': 'mlp.shared_experts.down_proj.weight',
}

# The global tensors.
GLOBAL = {
    'token_embd': 'model.embed_tokens.weight',
    'output_norm': 'model.norm.weight',
    'output': 'lm_head.weight',
}

# The indexer's tensors (the DSA layers): the GGUF name -> the model's parameter name.
INDEXER = {
    'indexer.attn_k': 'self_attn.indexer.wk.weight',
    'indexer.attn_q_b': 'self_attn.indexer.wq_b.weight',
    'indexer.k_norm': 'self_attn.indexer.k_norm.weight',
    'indexer.proj': 'self_attn.indexer.weights_proj.weight',
}


def _model_names(config):
    """The model's parameter names (the model on the meta device: no memory)."""
    import torch
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    with torch.device('meta'):
        model = DeepseekV4ForCausalLM(config)
    return list(model.state_dict())


def build_deepseek_v41_name_map(config):
    """{the GGUF tensor name: the model's parameter name} for DeepSeek V4.1."""
    dump = os.environ.get('XTIER_GGUF_NAME_DUMP')
    if dump:
        import torch
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
        with torch.device('meta'):
            model = DeepseekV4ForCausalLM(config)
        with open(dump, 'w') as f:
            for name, param in model.named_parameters():
                f.write(f'{name}\t{tuple(param.shape)}\t{param.dtype}\n')
        logger.info('xtier: the model\'s parameter names written to %s', dump)
    out = {}
    for gguf, hf in GLOBAL.items():
        out[f'{gguf}.weight'] = hf
    layers = config.num_hidden_layers
    for layer in range(layers):
        for gguf, hf in LAYER.items():
            out[f'blk.{layer}.{gguf}.weight'] = f'model.layers.{layer}.{hf}'
        for gguf, hf in INDEXER.items():
            out[f'blk.{layer}.{gguf}.weight'] = f'model.layers.{layer}.{hf}'
        # the router's bias
        out[f'blk.{layer}.exp_probs_b.bias'] = f'model.layers.{layer}.mlp.gate.e_score_correction_bias'
        # the attention's low-rank projections: the model wants the q and the kv apart (it fuses them itself)
        out[f'blk.{layer}.attn_q_a.weight'] = f'model.layers.{layer}.self_attn.wq_a.weight'
        out[f'blk.{layer}.attn_kv.weight'] = f'model.layers.{layer}.self_attn.wkv.weight'
    logger.info('xtier: the GGUF name map for DeepSeek V4.1: %d tensors', len(out))
    return out
