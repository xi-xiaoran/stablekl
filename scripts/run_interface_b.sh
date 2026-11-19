#!/usr/bin/env bash
# Interface B (torch.nn.functional.kl_div) on the released frozen data.
#   bash scripts/run_interface_b.sh <log_target 0|1> <fp16|fp32> <K> [device]
# 1. scores the released official outputs against the certified truth and writes report/ for the configuration;
# 2. runs the repaired kl_div (methods C, D32, D64) on <device>, scores it and writes the acceptance verdict.
# The score stage of step 1 writes score/ inside the configuration directory, so DATA must be writable
# (use a copy of data/interface_b).  Environment: DATA (default data/interface_b), OUT (default runs/interface_b).
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
LT=${1:?0 or 1}; DT=${2:?fp16 or fp32}; K=${3:?K}; DEV=${4:-cpu}
DATA=${DATA:-data/interface_b}; OUT=${OUT:-runs/interface_b}
CFG="lt$([ "$LT" = 1 ] && echo T || echo F)_${DT}_K${K}"; A=(--lt "$LT" --dt "$DT" --K "$K")
python -m kldivbench.pipeline score --root "$DATA" "${A[@]}"
python -m kldivbench.aggregate --root "$DATA" --cfg "$CFG" --out "$OUT/$CFG/report_official"
python benchmark/interface_b/config_report.py "$DATA" "$CFG" "$OUT/$CFG/report_official"
for st in run2 score2 pre2 h1b; do
  python -m kldivbench2.pipeline2 "$st" --root "$DATA" --root2 "$OUT" "${A[@]}" --device "$DEV"
done
python -m kldivbench2.verdict2 --root2 "$OUT" --cfg "$CFG"
