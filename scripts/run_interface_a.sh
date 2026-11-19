#!/usr/bin/env bash
# Interface A (torch.distributions.kl_divergence) on the released frozen data.
#   bash scripts/run_interface_a.sh <config> [device] [workers]
# Recomputes the official implementation (A; B32/B64 = the official formula evaluated one or two precisions higher),
# the same-precision rewrite (C) and the repair (D32/D64) on <device>, then scores every output against the
# certified truth.  The released official CPU outputs are scored in the same pass (suffix _cpu).
# Environment: DATA (default data), OUT (default runs/interface_a).
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
CFG=${1:?config name, see scripts/configs_a.txt}; DEV=${2:-cpu}; W=${3:-1}
DATA=${DATA:-data}; OUT=${OUT:-runs/interface_a}
python benchmark/evaluate_a.py compute --name "$CFG" --root "$DATA" --out "$OUT" --device "$DEV" --cpu-compare 0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python benchmark/evaluate_a.py score --name "$CFG" --root "$DATA" --out "$OUT" --workers "$W"
python benchmark/evaluate_a_bins.py --name "$CFG" --root "$DATA" --gdir "$OUT/outputs/$CFG" --out "$OUT" --workers "$W"
