#!/bin/bash
# Submit the Part B chain (shared -> features -> train -> eval -> summarize)
# with afterok dependencies.  Run from $HOME/DCR on the login node.
#   bash scripts/slurm/randommerge_pool_submit.sh
# Each stage is idempotent: existing heads/features/signals are reused unless
# the stage is re-run with --force.
set -euo pipefail
cd "$HOME/DCR"
SB=$(command -v sbatch || echo /opt/slurm/bin/sbatch)
SHARED=$("$SB" --parsable scripts/slurm/randommerge_pool_shared.slurm)
FEAT=$("$SB" --parsable --array=0-9 --dependency=afterok:"$SHARED" \
  scripts/slurm/randommerge_pool_features.slurm)
TRAIN=$("$SB" --parsable --array=0-69%12 --dependency=afterok:"$FEAT" \
  scripts/slurm/randommerge_pool_train.slurm)
EVAL=$("$SB" --parsable --array=0-9 --dependency=afterok:"$TRAIN" \
  scripts/slurm/randommerge_pool_eval.slurm)
SUM=$("$SB" --parsable --dependency=afterok:"$EVAL" \
  scripts/slurm/randommerge_pool_summarize.slurm)
cat <<EOF
shared=$SHARED
features=$FEAT (array 0-9)
train=$TRAIN (array 0-69%12)
eval=$EVAL (array 0-9)
summarize=$SUM
watch:  squeue -u $USER
EOF
