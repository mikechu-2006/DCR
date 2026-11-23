#!/usr/bin/env bash
# Wait for the audio/legacy P2/P4 run to finish, then run the paired and
# unpaired comparisons and the combined summary.  The polling loop lives in
# this background job so the agent does not have to wait in its context.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/p4_multiseed_audio
LOG=$OUT/analysis.log
DONE=$OUT/ANALYSIS_DONE
rm -f "$DONE"
: > "$LOG"
DEADLINE=$(( $(date +%s) + 6*3600 ))

MAIN_PID=$(pgrep -f "results/p4_multiseed_audio/run_all.log" | head -1 || true)
echo "[watch] start $(date -Is) main_pid=$MAIN_PID" >> "$LOG"

while true; do
  if grep -q "=== all done" "$OUT/run_all.log" 2>/dev/null; then
    echo "[watch] completion marker seen $(date -Is)" >> "$LOG"
    break
  fi
  if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "[watch] timeout waiting for completion marker" >> "$LOG"
      exit 1
    fi
    sleep 60
    continue
  fi
  echo "[watch] runner exited without completion marker $(date -Is)" >> "$LOG"
  tail -80 "$OUT/run_all.log" >> "$LOG" 2>/dev/null || true
  exit 1
done

echo "[watch] comparisons start $(date -Is)" >> "$LOG"
python scripts/compare_multiseed_runs.py \
  --baseline "$OUT/legacy" --variant "$OUT/audio" \
  --out "$OUT/audio_vs_legacy.md" >> "$LOG" 2>&1
python scripts/compare_multiseed_unpaired.py \
  --baseline results/p4_multiseed --variant "$OUT/legacy" \
  --out "$OUT/legacy_vs_p4_multiseed.md" >> "$LOG" 2>&1
python scripts/compare_multiseed_unpaired.py \
  --baseline results/p4_multiseed --variant "$OUT/audio" \
  --out "$OUT/audio_vs_p4_multiseed.md" >> "$LOG" 2>&1
python scripts/summarize_p4_multiseed_audio.py \
  --out-prefix "$OUT/combined_summary" >> "$LOG" 2>&1

touch "$DONE"
echo "[watch] done $(date -Is)" >> "$LOG"
