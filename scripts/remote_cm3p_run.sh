#!/usr/bin/env bash
# =============================================================================
#  Remote one-shot: CM3P embeddings for the REMAINING whitelist charts.
#
#  Run from the DSR repo root on the GPU box:
#      mkdir -p logs && bash scripts/remote_cm3p_run.sh 2>&1 | tee logs/cm3p_remote_$(date +%F).log
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
#      FETCH=...        manifest to download into the cache
#                       (default manifests/whitelist_1k.csv = the whole whitelist)
#      EMBED=...        manifest to encode
#                       (default manifests/whitelist_1k.csv = ALL 21,949 charts,
#                        one no-audio convention; EMBED=manifests/cm3p_todo_1k.csv
#                        encodes only what the published HF table misses)
#      MERGE=1          fold the encoded rows into the canonical content table
#
#  The cache holds .osu TEXT BODIES only -- no .osz, no audio, no images.
# =============================================================================
set -euo pipefail

CM3P_DIR=${CM3P_DIR:-external/cm3p}
OSU_CACHE=${OSU_CACHE:-data/osu_cache}            # .osu text only: no .osz/audio/images
FETCH=${FETCH:-manifests/whitelist_1k.csv}        # what the cache should hold
EMBED=${EMBED:-manifests/whitelist_1k.csv}        # what gets encoded (all 21,949)
PROBE=${PROBE:-manifests/cm3p_probe_1k.csv}
TAG=${TAG:-1k}
OUT_LOCAL=${OUT_LOCAL:-data/processed/chart_content_cm3p_local.parquet}
OUT_PROBE=${OUT_PROBE:-data/processed/chart_content_cm3p_probe.parquet}
WORKERS=${WORKERS:-4}
PY=${PY:-python}
# CM3P head is @torch.compile-decorated; Inductor would shell out to nvcc and can die
# with "PermissionError: [Errno 13] Permission denied: 'nvcc'" -- inference is eager.
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

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
  # deliberately NOT 'pip install -r requirements.txt': that pins a CPU torch and belongs to
  # pipeline B.  The inference-side list is requirements-cm3p.txt; its two load-bearing pins:
  #   * transformers -- only 4.42-4.57 has processing_utils.CommonKwargs (<=4.41 lacks it,
  #     >=5.0 removed it, same ImportError either way; the old ">=4.48" here allowed 5.x);
  #   * hydra-core -- CM3P's config.py imports 'hydra' at module level and
  #     utils/mmrs_dataset.py pulls it in, so '--source osu-dir' dies with
  #     "No module named 'hydra'" without it, even though Hydra itself is never used.
  $PY -m pip install --upgrade "transformers==4.55.0" "hydra-core>=1.3,<1.4" \
      "huggingface_hub>=0.26" accelerate
  $PY -m pip install numpy pandas pyarrow
  # slider: NOT from GitHub (the cluster cannot reach it) and NOT "pip install slider"
  # (PyPI's slider is a different project).  The wheel is vendored in vendor/.
  if ! $PY -m pip install --user --no-index --no-deps vendor/slider-*.whl; then
    $PY -c "import zipfile,glob;zipfile.ZipFile(glob.glob('vendor/slider-*.whl')[0]).extractall('vendor/_unpacked')"
    export PYTHONPATH="$PWD/vendor/_unpacked:${PYTHONPATH:-}"
  fi

  # Same imports step0c makes: a dependency check is only useful if it is identical.
  $PY - "$CM3P_DIR" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from cm3p.processing_cm3p import CM3PProcessor                # noqa: F401  (slider, transformers)
from cm3p.modeling_cm3p import CM3PModel                      # noqa: F401
from utils.beatmap_files_dataset import BeatmapFilesDataset   # noqa: F401  (config -> hydra)
print("CM3P imports OK")
EOF
fi

say "3/6 fetch .osu files (resumable, md5-checked)"
if [ "${SKIP_FETCH:-0}" = "1" ]; then
  echo "SKIP_FETCH=1 -- using whatever is in $OSU_CACHE"
else
  # the whole whitelist by default; the 256 probe ids are a subset of it, so one pass
  # covers both.  Charts that are deleted/restricted (404/403/451) simply stay absent.
  $PY scripts/step0d_fetch_osu.py --ids-file "$FETCH" --out-dir "$OSU_CACHE" --workers "$WORKERS"
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

say "6/6 full run: $EMBED"
$PY scripts/step0c_chart_content.py --source osu-dir --ids-file "$EMBED" \
    --osu-dir "$OSU_CACHE" --audio off --out "$OUT_LOCAL" --infer-batch "${INFER_BATCH:-8}"

if [ "${MERGE:-0}" = "1" ]; then
  say "6b/6 merge $OUT_LOCAL -> data/processed/chart_content_cm3p.parquet"
  $PY scripts/step0c_chart_content.py --source table --in "$OUT_LOCAL" --tag "$TAG"
fi

say "done"
echo "bring the result home (from the LOCAL machine):"
echo "    scp <user>@<host>:<repo>/$OUT_LOCAL  <repo>/data/processed/"
echo "    scp <user>@<host>:<repo>/data/processed/cm3p_mix_report.json <repo>/data/processed/"
echo "then either use it directly (step2 --chart-content cm3p --content-path $OUT_LOCAL)"
echo "or merge it into the canonical table:"
echo "    python scripts/step0c_chart_content.py --source table --in $OUT_LOCAL --tag $TAG"
echo
echo "finally update the whitelist table, and let step0 drop the non-cacheable charts:"
echo "    python scripts/step0g_whitelist.py --tag 1k --cache $OSU_CACHE \\"
echo "        --out data/processed/whitelist_1k.csv --manifest-out manifests/whitelist_1k.csv"
echo "    python scripts/step0_preprocess.py --dump-dir <1k dump> \\"
echo "        --whitelist data/processed/whitelist_1k.csv"
