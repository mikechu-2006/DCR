#!/bin/bash
# =============================================================================
#  Slurm job 1/2 - download the .osu files.  NEEDS INTERNET.
#
#  Submit from the repo root:
#      sbatch scripts/slurm/cm3p_fetch.slurm
#
#  Compute nodes on many clusters have no outbound network.  Test first:
#      srun -n 1 --time=00:05:00 bash -c \
#        'curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 https://osu.ppy.sh/osu/222593'
#  200 -> submit this job.  Anything else -> run the same command on the LOGIN node
#  inside tmux/nohup instead (it is light: ~10.3k small GETs, ~40 min):
#      tmux new -s fetch
#      python scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_todo_1k.csv \
#             --out-dir data/raw/osu_cache --workers 4
#      # Ctrl-b d to detach
#
#  Resumable: a file whose md5 already matches is skipped, so re-submitting after
#  a timeout only fetches what is still missing.
# =============================================================================
#SBATCH -J cm3p-fetch
#SBATCH -o slurm_cm3p_fetch_%j.out
#SBATCH -e slurm_cm3p_fetch_%j.err
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#  ^ no -p on purpose: this is a CPU job, let the cluster's default queue take it.
#    If your site requires one, add e.g.  #SBATCH -p <cpu-partition>  (see: sinfo)

set -uo pipefail
# see cm3p_embed.slurm: "$0" is Slurm's spooled copy, so use SLURM_SUBMIT_DIR instead
# REPO can be passed explicitly at submit time, which is the only reliable override if you
# are not standing in the repo when you call sbatch:
#     REPO=/scratch/$USER/DSR sbatch scripts/slurm/cm3p_embed.slurm
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
echo "job starts in: $(pwd)"
if [ ! -f scripts/step0d_fetch_osu.py ]; then
  echo "ERROR: submit this job from the repo root (currently in $REPO)." >&2
  exit 1
fi

# module load anaconda3        # <- uncomment / adapt for your site
PY="${PY:-python}"
OSU_CACHE="${OSU_CACHE:-$REPO/data/raw/osu_cache}"
WORKERS="${WORKERS:-4}"

echo "host=$(hostname)  repo=$REPO  python=$($PY -V 2>&1)"
echo "cache=$OSU_CACHE"
mkdir -p "$OSU_CACHE"

$PY scripts/step0d_fetch_osu.py \
    --ids-file manifests/cm3p_todo_1k.csv \
    --out-dir  "$OSU_CACHE" --workers "$WORKERS" --delay 0.25 || exit 1

$PY scripts/step0d_fetch_osu.py \
    --ids-file manifests/cm3p_probe_1k.csv \
    --out-dir  "$OSU_CACHE" --workers "$WORKERS" --delay 0.25 || exit 1

echo
echo "files now in the cache: $(ls "$OSU_CACHE"/*.osu 2>/dev/null | wc -l)"
echo "next: submit scripts/slurm/cm3p_embed.slurm"
