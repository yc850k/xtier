# sm_75 host: 2× RTX 2080 Ti (Turing)

The branch `feat/sm75` runs XTier on a Turing card: 2× RTX 2080 Ti (sm_75, 11 GB each), 2× Xeon E5-2680 v4
(Broadwell: AVX2, no AVX-512 BF16), 141 GiB of RAM, 2 NUMA nodes. What the card can and cannot do is in
[docs/sm75.md](../../docs/sm75.md); the launcher is [run-xtier-sm75.sh](run-xtier-sm75.sh), the console card is
[console.py](console.py).

```bash
MODEL=/home/userroot/work/model/ds41-gguf PORT=18200 configs/sm75-2x2080ti/run-xtier-sm75.sh [SGLang flags]
```

## The model directory

`/home/userroot/work/model/ds41-gguf/` is a small wrapper around the checkpoint on the Optane drive
(`/mnt/optane/DeepSeek-V4.1-Flash/`, 245 GiB: the four model shards `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-*` and
the Engram table `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`):

| in the directory | what it is |
|---|---|
| `config.json` | the original `deepseek-ai/DeepSeek-V4.1-Flash-1017` config (HF mirror) with the compressed variant's fields from the owner's `maya-ds4-gsq/ds4.txt` (compress ratios, Engram/index/kv layer ids, DSpark) |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.jinja` | the tokenizer (the real `tokenizer.json` from the model repo; the owner's `chat_template.jinja`) |
| `*.gguf` | symlinks to the four shards and the Engram table (the 138 GiB stay on the Optane drive) |

SGLang reads the GGUF with `--quantization gguf --load-format gguf` (the launcher sets both).

## Where SGLang stops on this checkpoint

SGLang 0.5.22 has two GGUF paths, and this checkpoint fits neither:

- the generic path (`--load-format gguf`) maps the tensor names with llama.cpp's architecture table, which knows
  `deepseek` and `deepseek2` but not DeepSeek V4/V4.1 → `Unknown gguf model_type: deepseek_v41`;
- the DeepSeek 4 path (`--load-format expert_pack`) wants the model type `deepseek_v4` (whose config class rejects
  the V4.1 compression ratios: `KeyError: 2`), a **single-file** source GGUF and a prepared expert pack
  (`sglang/srt/model_loader/expert_pack/build.py`, hours over 138 GiB).

So the owner's own loader (`maya-ds4-gsq`: the per-expert mixed quantization, the Engram in its own file, the
expert profile, the slots) remains the way to run this checkpoint; the config above is ready for a checkpoint SGLang
can read.

## XTier and this checkpoint

The CPU engine reads NVFP4 and MXFP4 experts. A GGUF checkpoint's experts (IQ2_XXS / IQ1_M / IQ2_XS per expert) are
not one of those formats, so the MoE layers stay with SGLang's own GGUF kernels: the load reports
`MoE layer N stays with SGLang: no CPU engine format for the MoE method GGUFMoEMethod` per layer, and the load's
self-check says that no layer uses a format the engine can read. XTier's hybrid path needs an NVFP4/MXFP4 checkpoint.

## What is off on this card

| Feature | Why |
|---|---|
| the MXFP4 fused MoE of the GPU prefill (`XTIER_PREFILL_FP4`) | the SM120 FP4 tensor cores; the packed weights are unpacked in software instead |
| the packs' FP8 features (GLM's FP8 weights, DeepSeek's FP8 draft head and W8A8 configs) | the FP8 tensor cores (sm_89) |
| the Qwen hyper-connection mix overlap | the TMA bulk prefetch (sm_90) |
| the Qwen NVFP4 KV cache reader | the FP8 conversions of the block scales (sm_89) |
| the MiniMax M3.1 pack | the M3.1 kernels are SM100 only |

Each of them is reported once in the log when it is off.

## Fast load

`XTIER_FAST_LOAD=1` (the launcher sets it): the experts go straight from the checkpoint files into the CPU engine
instead of through SGLang's loader, and the server still reports ready only when every expert is installed
(DeepSeek V4.1 on the production host: 472 → 186 s). It needs a checkpoint whose expert tensors XTier reads
(NVFP4/MXFP4) — a GGUF checkpoint goes through SGLang's own loader.

## The engine and the GPU

- The CPU engine runs the AVX2 build (`XTIER_ISA=avx2`): Broadwell has no AVX-512 BF16. The experts' FP4 unpacking
  is the same on both builds.
- The GPU prefill's tiles are smaller than on the Blackwell hosts (Turing has 64 KiB of shared memory per SM, the
  RTX PRO 5000 has 100): 64 wide with 2 stages.
- The hot expert cache holds few experts: an 11 GB card has no room for the 155 slots per layer of the RTX PRO 5000
  (`XTIER_CACHE_SLOTS=8`).

## What was checked here

The kernels on the card itself: `python tests/sm75_gpu_check.py` (the software FP4 decode and the grouped GPU prefill
in FP16 against an FP32 reference) and `python tests/arch_check.py` (the capability gates). The CPU engine builds and
runs here (`tools/build_engine.sh /usr/local/cuda`, the AVX2 build). An end-to-end SGLang run was not made: the
checkpoint above is not one SGLang can read (see above).
