#!/bin/bash
# Retry until the HPC login node accepts one session, then print E2/E3 status.
set -u
HOST="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
for i in $(seq 1 180); do
  if timeout 120 ssh -o ConnectTimeout=30 -o BatchMode=yes "$HOST" 'cd ~/DCR
echo "===== DATE ====="; date
echo "===== QUEUE ====="
timeout 20 /opt/slurm/bin/squeue -u $USER -o "%.12i %.20j %.2t %20P %.10M" | head -30 || true
echo "===== TARGET ====="
timeout 20 /opt/slurm/bin/squeue -j 14116577,14116578,14116579,14116580 -o "%.12i %.20j %.2t %20P %.10M" || true
echo "===== SACCT ====="
timeout 20 /opt/slurm/bin/sacct -j 14116577,14116578,14116579,14116580 --format=JobID,JobName%28,State,Elapsed,ExitCode -n -P 2>/dev/null | tail -40 || true
echo "===== FILES ====="
ls -lh results/behavior/e3/teacher_s1/signature.json runs/behavior/e2_fixed/eval/summary.md runs/behavior/e3/eval/summary.md 2>/dev/null || true
echo "===== E2 SUMMARY ====="
cat runs/behavior/e2_fixed/eval/summary.md 2>/dev/null || echo MISSING
echo "===== E3 SUMMARY ====="
cat runs/behavior/e3/eval/summary.md 2>/dev/null || echo MISSING
echo "===== E3 TEACHER ====="
cat results/behavior/e3/teacher_s1/signature.json 2>/dev/null | tail -30 || echo MISSING
echo "===== METRICS ====="
for f in runs/behavior/e2_fixed/*/s1/metrics.json runs/behavior/e3/*/s1/metrics.json; do
  [ -f "$f" ] || continue
  echo "--- $f"
  ~/.conda/envs/DCR/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in (\"task\",\"aux\",\"pair_views\",\"best_val_nll\",\"bad_batches\",\"elapsed_sec\")})" "$f" 2>/dev/null || true
done
' > /tmp/hpc_e2_e3_report.txt 2>/tmp/hpc_e2_e3_report.err; then
    cat /tmp/hpc_e2_e3_report.txt
    exit 0
  fi
  sleep 120
done
echo "[report] HPC still unreachable"
exit 1
