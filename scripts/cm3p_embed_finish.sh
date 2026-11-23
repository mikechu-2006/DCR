#!/bin/bash
# Embedding + assembly only (audio is already staged on the HPC).
set -uo pipefail
cd /hpc2hdd/home/aiaa3111_l01_008/DCR || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
export PATH=/opt/slurm/bin:$PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
say() { echo; echo "=== $(date '+%F %T') $*"; }
say "submit embedding array (resume)"
jobid=$(sbatch scripts/slurm/cm3p_embed_todo.slurm | awk '{print $NF}')
echo "array job: $jobid"
while squeue -h -j "$jobid" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; do
  sleep 120
done
say "assemble"
"$PY" scripts/cm3p_embed_todo.py --assemble \
    --ids-csv data/processed/cm3p_ids_todo_2786.csv \
    --in-dir results/cm3p_todo_official/shards \
    --out-parquet data/processed/chart_content_cm3p_local_audio.parquet
say "done"
