"""The GGUF checkpoints of a model SGLang runs: the loader's method and its weights iterator.

SGLang's GGUF support keeps the checkpoint's tensors in llama.cpp's formats (the ``gguf`` package's reader reads one
file, and the loader maps the names with the architecture the package and Transformers know). A Maya pack of DeepSeek
V4.1 is a split checkpoint of an architecture neither knows, and three things do not fit:

* the split: the parts are read as one file (xtier.backend.gguf_split);
* the names: the map from the checkpoint's names to the model's parameters (xtier.backend.gguf_names);
* the formats: the model's attention wants the weights dequantized (the MLA's kernels read the model's dtype, not
  llama.cpp's blocks), so the Linears keep the standard method and the loader dequantizes what it hands them. The
  routed experts are not in the model at all: XTier's GSQ path reads them from the checkpoint file (the engine's IQ
  decode), so the model must not hold them.
"""

import logging

logger = logging.getLogger(__name__)

# The model's parameters that hold llama.cpp's quantized blocks: the attention's (the MLA's kernels read the model's
# dtype). The name map says which checkpoint tensor goes where, so the rule is on the model's name.
DEQUANTIZED = ('self_attn.',)


def _is_quantized(tensor_type):
    from gguf.constants import GGMLQuantizationType

    return tensor_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16)


def install_method():
    """The GGUF config's method: the standard one for the layers whose kernels read the model's dtype.

    The MoE keeps the GGUF method (XTier's GSQ path takes the layer over and the experts stay in the file).
    """
    from sglang.srt.layers.quantization import gguf

    if getattr(gguf.GGUFConfig, '_xtier_method', False):
        return False
    original = gguf.GGUFConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            return original(self, layer, prefix)
        # the Linears and the embeddings: the model's kernels read the model's dtype, so the weights are dequantized
        from sglang.srt.layers.linear import (RowParallelLinear, ColumnParallelLinear, QKVParallelLinear,
                                              MergedColumnParallelLinear, ReplicatedLinear)
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead

        if isinstance(layer, (RowParallelLinear, ColumnParallelLinear, QKVParallelLinear, MergedColumnParallelLinear,
                              ReplicatedLinear, VocabParallelEmbedding, ParallelLMHead)):
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod, UnquantizedEmbeddingMethod

            if isinstance(layer, (VocabParallelEmbedding, ParallelLMHead)):
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()
        return original(self, layer, prefix)

    gguf.GGUFConfig.get_quant_method = get_quant_method
    gguf.GGUFConfig._xtier_method = True
    gguf.GGUFConfig._xtier_original = original
    logger.info('xtier: the GGUF loader dequantizes the weights (the model\'s kernels read its dtype)')
    return True


def install_iterator():
    """The loader's weights iterator: the checkpoint's tensors dequantized, the model's parameters named."""
    import torch
    from sglang.srt.model_loader import weight_utils

    if getattr(weight_utils.gguf_quant_weights_iterator, '_xtier_gguf', False):
        return False
    original = weight_utils.gguf_quant_weights_iterator

    def iterator(gguf_file, gguf_to_hf_name_map):
        import gguf
        from gguf.constants import GGMLQuantizationType

        reader = gguf.GGUFReader(gguf_file)
        for tensor in reader.tensors:
            name = gguf_to_hf_name_map.get(tensor.name)
            if name is None:
                continue   # not in the model (the routed experts: the engine reads them from the file)
            kind = tensor.tensor_type
            if kind == GGMLQuantizationType.BF16:
                # the gguf package's reader has no BF16 view: the bytes are the 16-bit values
                shape = tuple(tensor.data.shape[:-1]) + (tensor.data.shape[-1] // 2,)
                data = torch.frombuffer(tensor.data.tobytes(), dtype=torch.bfloat16).reshape(shape)
            elif kind == GGMLQuantizationType.F16:
                data = torch.from_numpy(tensor.data)
            elif kind == GGMLQuantizationType.F32:
                data = torch.from_numpy(tensor.data)
            else:
                data = torch.from_numpy(gguf.quants.dequantize(tensor.data, kind))
            if data.dim() == 3:
                # llama.cpp keeps some weights in groups (the attention's output: 8 groups of the model's rows)
                data = data.reshape(-1, data.shape[-1])
            yield name, data

    iterator._xtier_gguf = True
    iterator._xtier_original = original
    weight_utils.gguf_quant_weights_iterator = iterator
    # the loader imported the function by name: its own reference has to be replaced too
    from sglang.srt.model_loader import loader as loader_module
    loader_module.gguf_quant_weights_iterator = iterator
    logger.info('xtier: the GGUF loader reads the llama.cpp-native checkpoints (the attention dequantized)')
    return True
