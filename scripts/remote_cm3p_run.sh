#!/usr/bin/env bash
# =============================================================================
#  Remote one-shot: CM3P embeddings for the REMAINING whitelist charts.
#
#  Run from the DSR repo root on the GPU box:
#      bash scripts/remote_cm3p_run.sh 2>&1 | tee logs/cm3p_remote_$(date +%F).log
#
#  It is resumable and idempotent at every stage:
#    * an already-correct .osu is never re-downloaded (md5-checked);
#    * step0c only computes ids that are absent from its --out table;
#    * the probe gate runs FIRST, so you learn whether the no-audio rows may be
#      mixed with the published table before spending hours on the full run.
#
#  Knobs (env vars):
#      SKIP_INSTALL=1   do not touch pip
#      SKIP_FETCH=1     assume the .osu cache is already populated
#      SKIP_PROBE=1     skip the audio-gap gate (not recommended)
#      WORKERS=4        download concurrency
# =============================================================================
set -euo pipefail

CM3P_DIR=${CM3P_DIR:-external/cm3p}
OSU_CACHE=${OSU_CACHE:-data/raw/osu_cache}
TODO=${TODO:-manifests/cm3p_todo_1k.csv}
PROBE=${PROBE:-manifests/cm3p_probe_1k.csv}
OUT_ADD=${OUT_ADD:-data/processed/chart_content_cm3p_add.parquet}
OUT_PROBE=${OUT_PROBE:-data/processed/chart_content_cm3p_probe.parquet}
WORKERS=${WORKERS:-4}
PY=${PY:-python}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "0/6 environment"
$PY - <<'EOF'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
else:
    print("WARNING: no CUDA -- 22 layers x up to 4000 tokens per 16 s window on CPU is a "
          "multi-day job for 10k charts.  Strongly consider a GPU box.")
EOF

say "1/6 CM3P checkout"
if [ -d "$CM3P_DIR/cm3p" ]; then
  echo "$CM3P_DIR already present"
else
  mkdir -p "$(dirname "$CM3P_DIR")"
  git clone --depth 1 https://github.com/OliBomby/CM3P.git "$CM3P_DIR"
fi

say "2/6 dependencies"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo "SKIP_INSTALL=1 -- not touching pip"
else
  # deliberately NOT 'pip install -r requirements.txt': that pins a CPU torch, and this box
  # needs the CUDA build.  Only the inference-side deps are installed here.
  $PY -m pip install --upgrade "transformers>=4.48" "huggingface_hub>=0.26" accelerate
  $PY -m pip install numpy pandas pyarrow
  # slider: NOT from GitHub (the cluster cannot reach it) and NOT "pip install slider"
  # (PyPI's slider is a different project).  The wheel is vendored in vendor/.
  $PY -m pip install --user --no-index --no-deps vendor/slider-*.whl \
    || $PY -c "import zipfile,glob;zipfile.ZipFile(glob.glob('vendor/slider-*.whl')[0]).extractall('vendor/_unpacked')"
fi

say "3/6 fetch .osu files (resumable, md5-checked)"
if [ "${SKIP_FETCH:-0}" = "1" ]; then
  echo "SKIP_FETCH=1 -- using whatever is in $OSU_CACHE"
else
  $PY scripts/step0d_fetch_osu.py --ids-file "$TODO"  --out-dir "$OSU_CACHE" --workers "$WORKERS"
  $PY scripts/step0d_fetch_osu.py --ids-file "$PROBE" --out-dir "$OSU_CACHE" --workers "$WORKERS"
fi

if [ "${SKIP_PROBE:-0}" != "1" ]; then
  say "4/6 PROBE: re-encode 256 already-published charts with the no-audio path"
  $PY scripts/step0c_chart_content.py --source osu-dir --ids-file "$PROBE" \
      --osu-dir "$OSU_CACHE" --audio off --out "$OUT_PROBE" --infer-batch 8
  say "5/6 PROBE GATE: may the two tables be mixed?"
  $PY scripts/step0e_validate_mix.py --computed "$OUT_PROBE" \
      --json data/processed/cm3p_mix_report.json
  echo
  echo ">>> read the VERDICT above.  'ok'/'caution' -> continue; 'do-not-mix' -> stop and"
  echo ">>> either get audio (.osz) or keep the two subsets in separate experiments."
else
  say "4-5/6 probe skipped (SKIP_PROBE=1)"
fi

say "6/6 full run: remaining whitelist"
$PY scripts/step0c_chart_content.py --source osu-dir --ids-file "$TODO" \
    --osu-dir "$OSU_CACHE" --audio off --out "$OUT_ADD" --infer-batch 8

say "done"
echo "bring the result home (from the LOCAL machine):"
echo "    scp <user>@<host>:<repo>/$OUT_ADD  <repo>/data/processed/"
echo "    scp <user>@<host>:<repo>/data/processed/cm3p_mix_report.json <repo>/data/processed/"
echo "then merge locally:"
echo "    python scripts/step0c_chart_content.py --source table --in $OUT_ADD --tag 1k"
