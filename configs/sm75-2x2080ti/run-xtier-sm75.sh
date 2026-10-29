#!/usr/bin/env bash
# XTier on a Turing host (2x RTX 2080 Ti, sm_75): upstream SGLang main + XTier, the experts on the CPU engine and the
# hot ones on the GPU. The checkpoint and the server arguments come from the caller:
#
#   MODEL=/path/to/checkpoint PORT=18190 configs/sm75-2x2080ti/run-xtier-sm75.sh [SGLang flags]
#
# What sm_75 means here (docs/sm75.md):
# - no BF16 tensor cores: the model runs in FP16 (--dtype float16) and the GPU kernels take FP16 operands; the CPU
#   engine keeps its BF16 contract, so the activations are converted when they are staged for it;
# - no FP8 and no FP4 tensor cores: the packs' FP8 features are off and the packed FP4 experts are decoded in
#   software (the E2M1 values, the same numbers as the hardware conversion);
# - no TMA: the Qwen hyper-connection mix overlap stays off;
# - 64 KiB of shared memory per SM: the GPU prefill's tiles are smaller than on the Blackwell hosts.
set -eo pipefail
M=${MODEL:?set MODEL to the checkpoint}
D=${XTIER_ENV:-/home/userroot/work/llm-env}          # the python environment with SGLang
PY=${XTIER_PYTHON:-$D/bin/python}
PORT=${PORT:-18190}
# FlashInfer and TileLang compile for this device (7.5 on a Turing card, 12.0 on the RTX PRO 5000)
ARCH=${XTIER_CUDA_ARCH:-$($PY -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}.{c[1]}")' 2>/dev/null || echo 12.0)}
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-$ARCH}
export TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=${TVM_FFI_CUDA_ARCH_LIST:-$ARCH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export XTIER=${XTIER:-1}
# The CPU engine: Broadwell (AVX2 only, no AVX-512 BF16), 2 NUMA nodes of 14 cores
export XTIER_ISA=${XTIER_ISA:-avx2} XTIER_THREADS=${XTIER_THREADS:-24} XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
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
exec $PY -m sglang.launch_server --model-path $M --served-model-name ${SERVED_NAME:-$(basename $M)} \
  --host 127.0.0.1 --port $PORT --tp-size 1 --trust-remote-code --skip-server-warmup \
  --cuda-graph-backend-prefill disabled --dtype float16 --max-running-requests 2 \
  --weight-loader-drop-cache-after-load "$@"
