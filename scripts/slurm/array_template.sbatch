#!/bin/bash
# Template: one array task per configuration.  Fill in the placeholders for your cluster, then from the
# repository root:   sbatch --array=0-32 scripts/slurm/array_template.sbatch a     (interface A, 33 configurations)
#                    sbatch --array=0-19 scripts/slurm/array_template.sbatch b     (interface B, 20 configurations)
#SBATCH --job-name=stablekl
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
## For a GPU run add:  #SBATCH --gres=gpu:1   and set DEVICE=cuda below.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# source <your environment activation>      # e.g. a virtual environment with torch, numpy, mpmath, python-flint
DEVICE=${DEVICE:-cpu}
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
case "${1:?a or b}" in
  a) bash scripts/run_interface_a.sh "$(sed -n "${LINE}p" scripts/configs_a.txt)" "$DEVICE" "${SLURM_CPUS_PER_TASK:-1}" ;;
  b) bash scripts/run_interface_b.sh $(sed -n "${LINE}p" scripts/configs_b.txt) "$DEVICE" ;;
esac
