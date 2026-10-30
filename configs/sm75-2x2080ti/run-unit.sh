#!/usr/bin/env bash
# The sm_75 server as a transient user unit (no root, no password): the memory is capped (the kernel stops the unit
# at the limit instead of killing the host's other processes), the swap is off and the unit's OOM score is raised.
#
#   configs/sm75-2x2080ti/run-unit.sh [MODEL] [PORT]
#
# The logs: journalctl --user -u xtier-sm75 -f
#
# The properties mirror the Strata engine's unit (Strata/scripts/restart_strata.sh): MemoryHigh == MemoryMax (the
# cgroup v2 soft cap is effectively disabled, so the mmap'd checkpoint pages the engine reads are not evicted under
# it; the hard cap is the only safety net), MemorySwapMax=0, OOMScoreAdjust=500 and the NUMA policy preferred (the
# engine's shards are localized per node).
set -eo pipefail
M=${1:-/home/userroot/work/model/ds41-gguf}
PORT=${2:-18190}
UNIT=xtier-sm75
systemctl --user reset-failed $UNIT 2>/dev/null || true
systemctl --user stop $UNIT 2>/dev/null || true
sleep 2
exec systemd-run --user --unit=$UNIT --collect \
  --description="XTier DeepSeek V4.1 Flash on sm_75 (the Turing host)" \
  --property=MemoryHigh=120G \
  --property=MemoryMax=128G \
  --property=MemorySwapMax=0 \
  --property=OOMScoreAdjust=500 \
  --property=LimitMEMLOCK=16G \
  --property=NUMAPolicy=preferred \
  --property=NUMAMask=0 \
  --working-directory=/home/userroot/work/xtier \
  --setenv=PATH="/home/userroot/.bun/bin:/usr/local/cuda-13.2/bin:/home/userroot/.npm-global/bin:/home/userroot/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --setenv=MODEL=$M --setenv=PORT=$PORT --setenv=XTIER_DEBUG_ROUTE=${XTIER_DEBUG_ROUTE:-0} \
  --setenv=XTIER_COLD_FRACTION=${XTIER_COLD_FRACTION:-0.3} \
  --setenv=XTIER_COLD_LOG=${XTIER_COLD_LOG:-60} \
  --setenv=XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-500} \
  /home/userroot/work/xtier/configs/sm75-2x2080ti/run-xtier-sm75.sh
