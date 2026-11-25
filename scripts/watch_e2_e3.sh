#!/bin/bash
# Wait for the E2/E3 HPC chain, then print a compact status report.
# This is a background watcher: the agent does not poll; the job completion
# notification wakes it.
set -u
HOST="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
BASES="14116577 14116578 14116579 14116580"
PATTERN="^($(echo "$BASES" | tr " " "|"))(_|$)"
empty=0
for i in $(seq 1 720); do
  if ! active=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" \
      "/opt/slurm/bin/squeue -u \$USER -h -o '%i' 2>/dev/null | grep -E '$PATTERN' || true" 2>/dev/null); then
    echo "[watch] ssh unavailable at $(date +%H:%M:%S); retrying"
    sleep 60
    continue
  fi
  if [ -z "$active" ]; then
    empty=$((empty + 1))
    if [ "$empty" -ge 2 ]; then
      echo "[watch] no active E2/E3 jobs at $(date +%H:%M:%S)"
      break
    fi
  else
    empty=0
  fi
  sleep 60
done

ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" 'cd ~/DCR
echo "===== SACCT ====="
/opt/slurm/bin/sacct -j 14116577,14116578,14116579,14116580 \
  --format=JobID,JobName%28,State,Elapsed,ExitCode -n -P 2>/dev/null | tail -40 || true
echo "===== KEY FILES ====="
for f in data/processed/step0_judge_full.json \
         data/processed/step1_full_pairbest_judge.json \
         data/processed/step1_full_holdout_judge_s1.json \
         results/behavior/e3/teacher_s1/signature.json \
         runs/behavior/e2_fixed/eval/summary.md runs/behavior/e3/eval/summary.md; do
  if [ -f "$f" ]; then echo "--- $f"; tail -40 "$f"; else echo "--- $f MISSING"; fi
done
echo "===== RUN METRICS ====="
for f in runs/behavior/e2_fixed/*/s1/metrics.json runs/behavior/e3/*/s1/metrics.json; do
  [ -f "$f" ] || continue
  echo "--- $f"
  ~/.conda/envs/DCR/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in (\"task\",\"aux\",\"pair_views\",\"best_val_nll\",\"bad_batches\",\"elapsed_sec\")})" "$f" || true
done
echo "===== LOG TAILS ====="
for f in logs/behavior/e2_fixed_*.err logs/behavior/e3_teacher_*.err logs/behavior/e3_train_*.err \
         logs/behavior/e2_fixed_eval_*.err logs/behavior/e3_eval_*.err; do
  [ -f "$f" ] || continue
  echo "--- $f"
  tail -8 "$f"
done
' 2>&1
