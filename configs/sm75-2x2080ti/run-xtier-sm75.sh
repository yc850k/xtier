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
# - no BF16 tensor cores: the model keeps the checkpoint's BF16, the format its kernels were written for. The
#   hyper-connection's residual stream needs BF16's range (it reaches 284672 over the 40 layers, past FP16's 65504),
#   so the card's BF16 dots are emulated. The checkpoint's *weights* are dequantized into what the card computes:
#   the GGUF's IQ blocks at load time, the FP8 and FP4 ones in the kernels, with bit math (the same numbers as the
#   hardware conversions). The CPU engine keeps its BF16 contract (the activations are converted when they are
#   staged for it);
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
# An 11 GB card holds few hot experts: a few slots per layer instead of the 155 of the RTX PRO 5000
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-8} XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-8} XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500}
# The cold pool: the experts of the RAM slots, the rest read from the Optane drive when they are routed to (the
# checkpoint's own files are the store, read with O_DIRECT)
export XTIER_COLD=${XTIER_COLD:-1} XTIER_COLD_FRACTION=${XTIER_COLD_FRACTION:-0.3} XTIER_COLD_THREADS=${XTIER_COLD_THREADS:-8}
exec $PY -m sglang.launch_server --model-path $M --served-model-name ${SERVED_NAME:-$(basename $M)} \
  --host 127.0.0.1 --port $PORT --tp-size $TP --trust-remote-code --skip-server-warmup \
  --cuda-graph-backend-prefill disabled --dtype float16 --max-running-requests 2 \
  --quantization gguf --load-format gguf \
  --mem-fraction-static ${XTIER_MEM_FRACTION:-0.92} \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size ${XTIER_CHUNKED_PREFILL:-256} \
  --disable-cuda-graph \
  --weight-loader-drop-cache-after-load "$@"
