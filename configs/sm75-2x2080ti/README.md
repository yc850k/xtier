# sm_75 host: 2× RTX 2080 Ti (Turing)

The branch `feat/sm75` runs XTier on a Turing card: 2× RTX 2080 Ti (sm_75, 11 GB each), 2× Xeon E5-2680 v4
(Broadwell: AVX2, no AVX-512 BF16), 141 GiB of RAM, 2 NUMA nodes. What the card can and cannot do is in
[docs/sm75.md](../../docs/sm75.md); the launcher is [run-xtier-sm75.sh](run-xtier-sm75.sh).

```bash
MODEL=/path/to/checkpoint PORT=18190 configs/sm75-2x2080ti/run-xtier-sm75.sh [SGLang flags]
```

## The model

XTier's CPU engine reads the experts as they are in the checkpoint: NVFP4 (E2M1 weights, E4M3 scales per 16) or
MXFP4 (UE8M0 scales per 32). The rest of the model (attention, the dense layers, the embeddings) runs on the GPU,
so it has to work on Turing:

- **FP16, not BF16**: Turing has no BF16 tensor cores (`--dtype float16`; the launcher sets it). A checkpoint whose
  dense layers are BF16 is cast by SGLang at load time; a checkpoint with FP8 dense layers needs a model that
  SGLang can load in FP16 (XTier does not re-quantize anything).
- **An attention backend that works on sm_75 in FP16**: FlashAttention 2 and the MLA backends need sm_80 and
  newer, so the model needs SGLang's Triton attention (or another sm_75 backend).
- The routed experts themselves are FP4 in the checkpoint and stay that way: the engine unpacks them on the CPU.

## What is off on this card

| Feature | Why |
|---|---|
| the MXFP4 fused MoE of the GPU prefill (`XTIER_PREFILL_FP4`) | the SM120 FP4 tensor cores; the packed weights are unpacked in software instead |
| the packs' FP8 features (GLM's FP8 weights, DeepSeek's FP8 draft head and W8A8 configs) | the FP8 tensor cores (sm_89) |
| the Qwen hyper-connection mix overlap | the TMA bulk prefetch (sm_90) |
| the Qwen NVFP4 KV cache reader | the FP8 conversions of the block scales (sm_89) |
| the MiniMax M3.1 pack | the M3.1 kernels are SM100 only |

Each of them is reported once in the log when it is off.

## The engine and the GPU

- The CPU engine runs the AVX2 build (`XTIER_ISA=avx2`): Broadwell has no AVX-512 BF16. The experts' FP4 unpacking
  is the same on both builds.
- The GPU prefill's tiles are smaller than on the Blackwell hosts (Turing has 64 KiB of shared memory per SM, a
  quarter of the RTX PRO 5000's): 64 wide with 2 stages.
- The hot expert cache holds few experts: an 11 GB card has no room for the 155 slots per layer of the RTX PRO 5000
  (`XTIER_CACHE_SLOTS=8`).

## What was checked here

The kernels on the card itself: `python tests/sm75_gpu_check.py` (the software FP4 decode and the grouped GPU
prefill in FP16 against an FP32 reference) and `python tests/arch_check.py` (the capability gates). An end-to-end
SGLang run was not made on this host: it has no SGLang installation and no checkpoint for a Turing card.
