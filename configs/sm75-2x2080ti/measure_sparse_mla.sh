#!/usr/bin/env bash
# The two attention paths on the Turing host: the Triton sparse MLA (upstream's below-SM90 kernel) and the PyTorch
# path (the fallback). One server start per configuration, two requests each (the first is cold: the experts come
# from the drive).
#
#   bash configs/sm75-2x2080ti/measure_sparse_mla.sh
set -uo pipefail
cd /home/userroot/work/xtier
PORT=18190
PROMPT='{"model":"ds41-gguf","messages":[{"role":"user","content":"Write a short paragraph about the history of the printing press in Europe, including the Gutenberg press, the Reformation and the printing of scientific journals."}],"max_tokens":64,"temperature":0}'

req() {
  local t0 t1
  t0=$(date +%s.%N)
  OUT=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PROMPT")
  t1=$(date +%s.%N)
  echo "$(echo "$t1 - $t0" | bc)"
}

start() {
  configs/sm75-2x2080ti/run-unit.sh >/tmp/unit-$1.log 2>&1
  for i in $(seq 1 90); do
    if journalctl --user -u xtier-sm75 --since "-3min" --no-pager 2>/dev/null | grep -q "fired up"; then return 0; fi
    sleep 5
  done
  echo "the server did not come up" >&2
  return 1
}

for mode in 1 0; do
  name=$([ "$mode" = 1 ] && echo triton || echo pytorch)
  export XTIER_SPARSE_MLA=$mode
  start "$name" || exit 1
  sleep 3
  for n in 1 2; do
    t=$(req)
    echo "$name request $n: ${t} s"
  done
  echo "$name: the Triton kernel calls: $(journalctl --user -u xtier-sm75 --since '-20min' --no-pager 2>/dev/null | grep -c 'Triton sparse MLA took')"
  systemctl --user stop xtier-sm75.service 2>/dev/null
  sleep 5
done
