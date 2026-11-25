#!/bin/bash
# Local watcher for the Phase 0 HPC jobs.  It blocks in sbatch --wait on a
# sentinel that depends on attach + baselines, so the agent does not poll.
set -uo pipefail
HPC=hpc2login.hpc.hkust-gz.edu.cn
ssh -o BatchMode=yes "$HPC" 'cd ~/DCR && /opt/slurm/bin/sbatch --wait --dependency=afterok:14080080:14080081 scripts/slurm/behavior_phase0_sentinel.slurm; rc=$?; echo SENTINEL_RC=$rc; echo ---SACCT---; /opt/slurm/bin/sacct -j 14080071,14080072,14080079,14080080,14080081 --format=JobID,JobName,State,ExitCode,Elapsed -P | head -60; echo ---LOGS---; for f in logs/behavior/judge_scan_*.out logs/behavior/judge_union_*.out logs/behavior/judge_attach_*.out logs/behavior/h4_split_*.out logs/behavior/h4_base_*.out; do if [ -f "$f" ]; then echo "### $f"; tail -6 "$f"; fi; done' 2>&1
echo "WATCH_DONE"
