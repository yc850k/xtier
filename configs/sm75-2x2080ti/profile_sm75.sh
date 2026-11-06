#!/usr/bin/env bash
# Where the time goes on the Turing host: one request with the timing and the debug logs on, so the log shows the
# MoE layers' kinds and times and the attention's path.
#
#   bash configs/sm75-2x2080ti/profile_sm75.sh
set -uo pipefail
cd /home/userroot/work/xtier
PORT=18190
PROMPT='{"model":"ds41-gguf","messages":[{"role":"user","content":"Write a short paragraph about the history of the printing press in Europe, including the Gutenberg press, the Reformation and the printing of scientific journals."}],"max_tokens":64,"temperature":0}'

export XTIER_TIMING=1
export XTIER_COLD_LOG=60
configs/sm75-2x2080ti/run-unit.sh >/tmp/unit-profile.log 2>&1
for i in $(seq 1 90); do
  if journalctl --user -u xtier-sm75 --since "-3min" --no-pager 2>/dev/null | grep -q "fired up"; then break; fi
  sleep 5
done
sleep 3
T0=$(date +%s.%N)
curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PROMPT" >/dev/null
T1=$(date +%s.%N)
echo "the request: $(echo "$T1 - $T0" | bc) s"
echo "=== the MoE layers' times ==="
journalctl --user -u xtier-sm75 --since "-10min" --no-pager 2>/dev/null | grep -oE "xtier timing:.*" | tail -6
echo "=== the cold pool and the cache ==="
journalctl --user -u xtier-sm75 --since "-10min" --no-pager 2>/dev/null | grep -oE "xtier (cold|cache):[^,]{0,90}" | tail -8
echo "=== the GPU's share ==="
journalctl --user -u xtier-sm75 --since "-10min" --no-pager 2>/dev/null | grep -oE "xtier[^,]{0,100}(cache|slots|prefill)[^,]{0,60}" | tail -8
