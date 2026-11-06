#!/usr/bin/env bash
# XTier on a Turing host (2x RTX 2080 Ti, sm_75): upstream SGLang main + XTier on the GGUF checkpoint on the Optane
# drive (the four model shards + the Engram table). The checkpoint and the server arguments come from the caller:
#
#   MODEL=/mnt/optane/DeepSeek-V4.1-Flash/ PORT=18200 configs/sm75-2x2080ti/run-xtier-sm75.sh [SGLang flags]
#
# The GGUF shards hold the dense weights; the MoE experts come from the GSQ pack on the Optane drive
# (XTIER_GSQ_PACK), which the CPU engine reads (the load log names the layers it took).
#
# What sm_75 means here (docs/sm75.md):
# - no BF16 tensor cores: the model runs FP16, the dtype the card's tensor cores take (the dot products are FP16).
#   The hyper-connection's residual stream needs more range than FP16 has (it reaches 284672 over the 40 layers,
#   past FP16's 65504), so the pack keeps that stream in FP32 (packs/deepseek_v4/dtypes.py). The checkpoint's
#   *weights* are dequantized into what the card computes: the GGUF's IQ blocks at load time, the FP8 and FP4 ones
#   in the kernels, with bit math (the same numbers as the hardware conversions). The CPU engine keeps its BF16
#   contract (the activations are converted when they are staged for it);
# - no FP8 and no FP4 tensor cores: the packs' FP8 features are off and the packed FP4 experts are decoded in
#   software (the E2M1 values, the same numbers as the hardware conversion);
# - no TMA: the Qwen hyper-connection mix overlap stays off;
# - 64 KiB of shared memory per SM: the GPU prefill's tiles are smaller than on the Blackwell hosts.
set -eo pipefail
M=${MODEL:-/home/userroot/work/model/ds41-gguf}   # the GGUF checkpoint: config.json + tokenizer + the GGUF
                                               # symlinks to the shards and the Engram table on the Optane drive
D=${XTIER_ENV:-/home/userroot/work/.venvs/xtier-sm75}   # the python environment with SGLang and XTier
PY=${XTIER_PYTHON:-$D/bin/python}
PORT=${PORT:-18190}
# The GSQ pack (the MoE experts in the format the CPU engine reads) on the Optane drive
export XTIER_GSQ_PACK=${XTIER_GSQ_PACK:-/mnt/optane/DeepSeek-V4.1-Flash/maya-ds4-gsq}
CUDA=${XTIER_CUDA_ROOT:-$($PY - <<'EOF'
# The JIT compilers (FlashInfer, TVM) need a toolkit whose compiler and headers are the same version: the
# environment's nvidia-cu13 package can hold a 13.4 compiler with 13.0 headers, and the CCCL headers refuse
# that mix ("CUDA compiler and CUDA toolkit headers are incompatible"). The first consistent toolkit wins.
import glob, os, re, site, subprocess, sys
roots = sorted(glob.glob('/usr/local/cuda-*'), reverse=True)
roots += [os.path.join(p, 'nvidia', 'cu13') for p in site.getsitepackages()]
for root in roots:
    nvcc, header = os.path.join(root, 'bin', 'nvcc'), os.path.join(root, 'include', 'cuda_runtime_api.h')
    if not (os.access(nvcc, os.X_OK) and os.path.exists(header)):
        continue
    cc = re.search(r'release ([0-9.]+)', subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout)
    ver = re.search(r'define CUDART_VERSION\s+([0-9]+)', open(header).read())
    if not (cc and ver):
        continue
    v = int(ver.group(1))
    if cc.group(1) == f'{v // 1000}.{(v % 1000) // 10}':
        print(root)
        break
EOF
)}
[ -n "$CUDA" ] || CUDA=/usr/local/cuda
export CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib:${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1
# Both cards compute (the GPU pool and the prefill can take the second one). The display is on the
# AMD FirePro W4100, so the NVIDIA cards carry no output; the DRM path is off (gui-freeze-fix.sh).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# FlashInfer and TileLang compile for this device (7.5 on a Turing card, 12.0 on the RTX PRO 5000)
ARCH=${XTIER_CUDA_ARCH:-$($PY -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}.{c[1]}")' 2>/dev/null || echo 12.0)}
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-$ARCH}
export TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=${TVM_FFI_CUDA_ARCH_LIST:-$ARCH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# The MHC's TileLang kernels (the pre-mix, the post-mix) and the DSA indexer's need the tensor cores of sm_80 and
# newer: they compile for this card and then fail to load. SGLang's own paths take over (the MHC's pre-mix in
# PyTorch, the indexer in Triton). The pack sets these too, but only in the process that installs the hooks: with
# TP the model worker is a separate process, so the switches are set here, where every process inherits them.
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-0}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-0}
export SGLANG_OPT_USE_TILELANG_INDEXER=${SGLANG_OPT_USE_TILELANG_INDEXER:-0}
export XTIER=${XTIER:-1}
# Fast load: the experts go straight from the checkpoint files into the CPU engine (still all installed before the
# server reports ready; the loader's own copy of them is skipped)
export XTIER_FAST_LOAD=${XTIER_FAST_LOAD:-1}
# The CPU engine: Broadwell (AVX2 only, no AVX-512 BF16), 2 NUMA nodes of 14 cores
# The engine's workers take all but a few CPUs: one for the mailbox dispatcher and a few for the process's own
# threads (the scheduler, torch). All of them would leave the dispatcher without a CPU.
export XTIER_ISA=${XTIER_ISA:-avx2} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
# Tensor parallelism over the two cards (TP=2): each rank holds half of the dense weights, of the KV cache and of
# the hot experts, and its own CPU engine over its own group of NUMA nodes. The threads are the budget per rank
# (tools/cpu_budget.py --tp N: the scheduler's CPUs and the ranks' share of the rest).
TP=${TP:-2}
if [ -z "${XTIER_THREADS:-}" ]; then
  XTIER_THREADS=$($PY "$(dirname "$0")/../../tools/cpu_budget.py" --tp $TP)
fi
export XTIER_THREADS
if [ "$TP" -gt 1 ]; then export XTIER_TP_NODES=${XTIER_TP_NODES:-auto}; fi
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576}
# The GPU prefill: the tiles fit Turing's 64 KiB of shared memory (the Blackwell hosts run 128 x 64 with 3 stages)
export XTIER_GPU_PREFILL_CHUNK=${XTIER_GPU_PREFILL_CHUNK:-4096} XTIER_PREFILL_EXPERT_CHUNK=${XTIER_PREFILL_EXPERT_CHUNK:-8}
export XTIER_PREFILL_PREFETCH=${XTIER_PREFILL_PREFETCH:-1} XTIER_PREFILL_RELEASE_WORKSPACE=${XTIER_PREFILL_RELEASE_WORKSPACE:-1}
export XTIER_PREFILL_ROUTE_ACCUM=${XTIER_PREFILL_ROUTE_ACCUM:-1} XTIER_PREFILL_UNPACK=${XTIER_PREFILL_UNPACK:-1}
export XTIER_GATE_BN=${XTIER_GATE_BN:-64} XTIER_GATE_BK=${XTIER_GATE_BK:-64} XTIER_GATE_LAUNCH=${XTIER_GATE_LAUNCH:-4,2}
export XTIER_DOWN_BN=${XTIER_DOWN_BN:-64} XTIER_DOWN_BK=${XTIER_DOWN_BK:-32} XTIER_DOWN_LAUNCH=${XTIER_DOWN_LAUNCH:-4,2}
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=${XTIER_GPU_PREFILL_MIN:-1024}
# The hot experts in the VRAM: as many as fit, the KV cache takes the rest (the context on this host is short).
# A rank's slice of a GSQ expert (the rows of the gate and the up it owns, the down whole) is 5.2 MB of IQ bytes, so
# a slot per layer costs 5.2 MB x the layers: 26 slots a layer are 5.4 GB. The dense weights take 12.1 GB of the
# 22.5 GB of a card (measured at the load), so the KV cache (the context here is a few hundred tokens) and the hot
# experts share what is left.
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-26} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-8} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-16}
# The adaptive residency: the counts decay slowly and the gain bar is low, so the slots follow the routes the decode
# really makes (the profile's ranking is the production traffic's, a different traffic needs the swaps).
export XTIER_CACHE_DECAY=${XTIER_CACHE_DECAY:-0.95} XTIER_CACHE_MIN_GAIN=${XTIER_CACHE_MIN_GAIN:-1.05}
# The batched prefill uploads the experts the batch routes to (the ones that are not in the VRAM) over PCIe and
# decodes them on the GPU: the scratch arena they go into (shared by the layers) is this many slots.
# The scratch arena the batched prefill uploads the routed experts into (shared by the layers): 128 slots of a rank's
# slice (5.2 MB) are 660 MB.
# The card's VRAM after the weights (12.1 GB) and the KV cache (3.3 GB at the minimum mem-fraction) is about 6 GB:
# 26 hot slots a layer (5.4 GB) and the scratch (0.7 GB) fit in it, the rest is the CUDA context.
export XTIER_GSQ_PREFILL_UPLOAD=${XTIER_GSQ_PREFILL_UPLOAD:-128}
# The prefill's tiles: the routes of one expert at a time (the tile's rows of routes). A small tile keeps the kernel's
# registers low (the decoded tile of 32 routes and 16 rows does not fit), which is worth twice the prefill's speed here
# (the time to the first token: 47 s -> 26 s).
export XTIER_GSQ_PREFILL_TILE_M=${XTIER_GSQ_PREFILL_TILE_M:-8}
# The cold pool: the experts of the RAM slots, the rest read from the Optane drive when they are routed to (the
# checkpoint's own files are the store, read with O_DIRECT). Each rank keeps its own copy of the experts it holds in
# RAM, so with two ranks the pool needs twice as much memory.
export XTIER_COLD=${XTIER_COLD:-1} XTIER_COLD_THREADS=${XTIER_COLD_THREADS:-8}
# The cold pool's share of the experts that are NOT in RAM: the RAM holds the rest, and the GPU's traffic touches
# 116-125 experts of a layer, so the pool needs a bit more than that to stop its LRU from churning (the reads from the
# drive are 1.8 ms a layer call at 0.55, 2.9 ms at 0.65). The slices make the slots two thirds of the bytes, so the
# pool's RAM is 71 GB of the unit's 141 (the page cache keeps the rest). With one rank the whole set fits in the RAM.
if [ -z "${XTIER_COLD_FRACTION:-}" ]; then
  if [ "$TP" -gt 1 ]; then export XTIER_COLD_FRACTION=0.55; else export XTIER_COLD_FRACTION=0; fi
fi
# The cold pool's RAM holds the hot experts (the engine reads them there), the rest are read from the Optane drive
# when a forward routes to them (the store's own files, O_DIRECT).
#
# The budget, measured: the whole expert set is 118.9 GB (384 experts x 40 layers x 7.74 MB) and the host has 141 GB
# of RAM. Each rank holds its own copy of the experts it computes, so the RAM holds the set as many times as there are
# ranks that compute the same experts: with one rank the whole set fits (the cold fraction 0: nothing on the drive but
# the store's copy), with two ranks (TP=2) each rank holds a share of it. The slots hold the rank's slice of an expert
# (5.2 MB), so the fraction 0.55 (173 of 384 experts a layer, 36 GB a rank, 71 GB for the two) leaves room for the
# engine's own buffers and the page cache (37 GB on this host): the fraction 0.3 (269 experts a rank, 167 GB for the
# two) took the server with the OOM killer.
# The engine's RAM registered with CUDA (the upload of a batched prefill copies the experts' bytes from there): the
# pages are locked, so the limit has to allow for them. 18.5 GB is the default here; the whole set needs more.
ulimit -l unlimited 2>/dev/null || true
exec $PY -m sglang.launch_server --model-path $M --served-model-name ${SERVED_NAME:-$(basename $M)} \
  --host 127.0.0.1 --port $PORT --tp-size $TP --trust-remote-code --skip-server-warmup \
  --cuda-graph-backend-prefill disabled --dtype float16 --max-running-requests 2 \
  --quantization gguf --load-format gguf \
  --mem-fraction-static ${XTIER_MEM_FRACTION:-0.727} \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size ${XTIER_CHUNKED_PREFILL:-256} \
  --disable-cuda-graph \
  --weight-loader-drop-cache-after-load "$@"
