#!/bin/bash
# Submit the legacy L0-L3 chain (train 40 -> eval 10 -> summarize).
# Run from $HOME/DCR on the login node.
set -euo pipefail
cd "$HOME/DCR"
SB=$(command -v sbatch || echo /opt/slurm/bin/sbatch)
TRAIN=$("$SB" --parsable --array=0-39%12 scripts/slurm/randommerge_od_legacy_train.slurm)
EVAL=$("$SB" --parsable --array=0-9 --dependency=afterok:"$TRAIN" \
  scripts/slurm/randommerge_od_legacy_eval.slurm)
SUM=$("$SB" --parsable --dependency=afterok:"$EVAL" \
  scripts/slurm/randommerge_od_legacy_summary.slurm)
cat <<EOF
train=$TRAIN (array 0-39%12)
eval=$EVAL (array 0-9)
summarize=$SUM
EOF
