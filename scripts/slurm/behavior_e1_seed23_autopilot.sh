#!/bin/bash
# Autopilot: E1 seeds 2-3, retrying while the GPU QOS submit limit is hit.
# It submits once the limit clears, waits for both arrays, then prints evals.
set -uo pipefail
HPC=hpc2login.hpc.hkust-gz.edu.cn

submit_one() {
  local seed="$1"
  while true; do
    out=$(ssh -o BatchMode=yes "$HPC" "cd ~/DCR && /opt/slurm/bin/sbatch --parsable -p emergency_gpu --export=ALL,SEED=$seed scripts/slurm/behavior_e1_train.slurm" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      echo "SEED$seed_JOB=$out"
      echo "$out"
      return 0
    fi
    echo "[autopilot] seed=$seed submit not allowed yet: $out"
    sleep 600
  done
}

J2=$(submit_one 2 | tail -1)
J3=$(submit_one 3 | tail -1)
echo "[autopilot] waiting jobs $J2 $J3"
while true; do
  active=$(ssh -o BatchMode=yes "$HPC" "export PATH=/opt/slurm/bin:\$PATH; squeue -h -u \$USER -o '%i'" 2>/dev/null | grep -cE "^($J2|$J3)(_[0-9]+)?$" || true)
  if [ "$active" -eq 0 ]; then break; fi
  sleep 300
done
echo "[autopilot] jobs done"
ssh -o BatchMode=yes "$HPC" 'cd ~/DCR && export PATH=/opt/slurm/bin:$PATH; sacct -j '$J2','$J3' --format=JobID,JobName,State,ExitCode,Elapsed -P | head -30; echo ---EVALS---; for f in runs/behavior/e1/*/s2/eval.json runs/behavior/e1/*/s3/eval.json; do echo "### $f"; cat "$f" 2>/dev/null | tail -5; done' 2>&1
echo E1_SEED23_AUTOPILOT_DONE
