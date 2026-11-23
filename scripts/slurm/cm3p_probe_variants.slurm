#!/bin/bash
# =============================================================================
#  Diagnostic variants of the official-embedding probe, on a small pilot sample.
#
#  Purpose: identify the residual difference between our re-encode (cos ~0.998)
#  and the published vectors.  Each variant re-encodes the SAME pilot charts and
#  writes its own jsonl/summary; nothing is shared except the audio and the table.
#
#      A  librosa   fp32  longest     <- baseline (matches the first pilot)
#      B  ffmpeg    fp32  longest     <- newer transformers decode path
#      C  librosa   bf16  longest     <- the dtype the published table used (GPU)
#      D  librosa   fp32  max_length  <- literal upstream padding (control)
#      E  (audio off)                 <- domain-shift control, expect a low cos
#
#  Submit:  sbatch -p debug --time=00:25:00 scripts/slurm/cm3p_probe_variants.slurm
# =============================================================================
#SBATCH -p debug
#SBATCH -J cm3p-probe-var
#SBATCH -o slurm_cm3p_probe_var_%j.out
#SBATCH -e slurm_cm3p_probe_var_%j.err
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --time=00:25:00

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
SAMPLES=${SAMPLE:-results/cm3p_probe_official/sample_pilot5.csv}
AUDIO_DIR=${AUDIO_DIR:-data/probe_audio}
MODEL_DIR=${MODEL_DIR:-data/raw/cm3p_model_26cc694}
OUTDIR=${OUTDIR:-results/cm3p_probe_official/variants}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export USE_TF=0 USE_FLAX=0
CONDA_LIB=${CONDA_LIB:-/hpc2ssd/softwares/anaconda3/lib}
[ -d "$CONDA_LIB" ] && export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}

mkdir -p "$OUTDIR"
echo "=== variants on $(hostname) ==="
date
"$PY" - <<'EOF'
import transformers, torch
print("transformers", transformers.__version__, "torch", torch.__version__)
EOF

run() {  # name, extra args...
  local name="$1"; shift
  echo
  echo "--- variant $name: $*"
  "$PY" scripts/cm3p_official_probe.py \
      --sample "$SAMPLES" --audio-dir "$AUDIO_DIR" --model-dir "$MODEL_DIR" \
      --official-table data/processed/chart_content_cm3p.parquet \
      --osu-dir data/osu_cache --device cpu --crop legacy \
      --out "$OUTDIR/$name.jsonl" --summary "$OUTDIR/$name.json" \
      "$@" > "$OUTDIR/$name.log" 2>&1
  local rc=$?
  /usr/bin/env python3 - "$OUTDIR/$name.json" "$rc" <<'EOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    c = d.get("cos", {})
    print(f"  rc={sys.argv[2]} n_ok={d.get('n_ok')} mean={c.get('mean')} "
          f"min={c.get('min')} n>=0.99={c.get('n_ge_0_99')}")
except Exception as exc:
    print("  no summary:", exc)
EOF
}

run A_librosa_fp32_longest --audio on --audio-loader librosa --dtype float32 --pad-mode longest
run B_ffmpeg_fp32_longest  --audio on --audio-loader ffmpeg  --dtype float32 --pad-mode longest
run C_librosa_bf16_longest  --audio on --audio-loader librosa --dtype bfloat16 --pad-mode longest
run D_librosa_fp32_maxlen   --audio on --audio-loader librosa --dtype float32 --pad-mode max_length
run E_noaudio               --audio off --dtype float32 --pad-mode longest

echo
echo "=== done ==="
date
