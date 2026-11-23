#!/bin/bash
# Wait for the Slurm array, then pull results and logs to the local repo.
set -uo pipefail
JOB=14031088
HOST=hpc2login.hpc.hkust-gz.edu.cn
while true; do
  n=$(ssh "$HOST" "export PATH=/opt/slurm/bin:\$PATH; squeue -j $JOB -h 2>/dev/null | wc -l" 2>/dev/null || echo 1)
  if [ "$n" = "0" ]; then break; fi
  sleep 30
done
sleep 10
ssh "$HOST" "export PATH=/opt/slurm/bin:\$PATH; sacct -j $JOB -X -o JobID,State,ExitCode -n" > /tmp/ki_sacct.txt 2>&1 || true
mkdir -p results/ki_eval/10k results/ki_eval/rm
rsync -avz --no-perms --omit-dir-times "$HOST:ki_eval/results/ki_eval/" results/ki_eval/ 2>&1 | tail -5
rsync -avz --no-perms --omit-dir-times "$HOST:ki_eval/slurm_ki_*.out" results/ki_eval/ 2>&1 | tail -3 || true
echo "=== sacct ==="
cat /tmp/ki_sacct.txt
