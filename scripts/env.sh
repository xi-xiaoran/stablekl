# Source this file from the repository root:  source scripts/env.sh
# Puts the package and the two benchmark harnesses on PYTHONPATH (no installation needed for the benchmark).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO:$REPO/benchmark/interface_a:$REPO/benchmark/interface_b${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore
