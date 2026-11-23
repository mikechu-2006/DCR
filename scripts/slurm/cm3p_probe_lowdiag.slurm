#!/bin/bash
# =============================================================================
#  Diagnostics for the charts whose cosine fell below 0.99.
#
#  Both low-cos charts have tok_max=2000, i.e. at least one 16 s window hit the
#  max_length truncation.  These runs isolate that:
#
#      F  pad=longest    max_length=2000   baseline (same as the full probe)
#      G  pad=max_length max_length=2000   literal upstream padding
#      H  pad=max_length max_length=4000   no truncation at 2000 (hypothesis test)
#      I  pad=longest    max_length=2000, ffmpeg decode
#
#  Submit:  sbatch -p debug --time=00:25:00 scripts/slurm/cm3p_probe_lowdiag.slurm
# =============================================================================
#SBATCH -p debug
#SBATCH -J cm3p-lowdiag
#SBATCH -o slurm_cm3p_lowdiag_%j.out
#SBATCH -e slurm_cm3p_lowdiag_%j.err
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --time=00:25:00

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
SAMPLES=${SAMPLE:-results/cm3p_probe_official/sample_low.csv}
AUDIO_DIR=${AUDIO_DIR:-data/probe_audio}
MODEL_DIR=${MODEL_DIR:-data/raw/cm3p_model_26cc694}
OUTDIR=${OUTDIR:-results/cm3p_probe_official/lowdiag}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export USE_TF=0 USE_FLAX=0
CONDA_LIB=${CONDA_LIB:-/hpc2ssd/softwares/anaconda3/lib}
[ -d "$CONDA_LIB" ] && export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
mkdir -p "$OUTDIR"

run() {
  local name="$1"; shift
  echo; echo "--- $name: $*"
  "$PY" scripts/cm3p_official_probe.py \
      --sample "$SAMPLES" --audio-dir "$AUDIO_DIR" --model-dir "$MODEL_DIR" \
      --official-table data/processed/chart_content_cm3p.parquet \
      --osu-dir data/osu_cache --device cpu --crop legacy \
      --out "$OUTDIR/$name.jsonl" --summary "$OUTDIR/$name.json" \
      "$@" > "$OUTDIR/$name.log" 2>&1
  echo "  rc=$?"
  "$PY" - "$OUTDIR/$name.json" <<'EOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
print("  ", {r["beatmap_id"]: round(r["cos_official"], 5)
            for r in map(json.loads, open(d["out"])) if r.get("cos_official")})
EOF
}

run F_padlongest_ml2000 --audio on --audio-loader librosa --dtype float32 --pad-mode longest --max-length 2000
run G_padmax_ml2000     --audio on --audio-loader librosa --dtype float32 --pad-mode max_length --max-length 2000
run H_padmax_ml4000     --audio on --audio-loader librosa --dtype float32 --pad-mode max_length --max-length 4000
run I_ffmpeg_ml2000     --audio on --audio-loader ffmpeg  --dtype float32 --pad-mode longest --max-length 2000

echo; echo "=== done ==="; date
