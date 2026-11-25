# Attribute-channel 1-seed pilot — final status (2026-11-25 17:10)

## Completed

* M0 implementation + 11-group test suite (local CPU): attributes.py, checkpoint.py,
  configs a12/a20/a27/a20_shuf, train/data/extract/model edits.
* HPC training (A800): S-a0cont/a12/a20/a27 (5 epoch continuation) and
  D-a0d/a12d/a20d/a27d (20 epoch from scratch, mapset-disjoint 16,487 charts).
* Extraction: 8 x 512d content tables (S: 6,828 charts; D: 5,462 charts).
* Probes (CPU):
  * S: P2 / P4 / P4-hold, seed 1, 4 arms (results/d2v_attr/).
  * D: cached set-P4 (5,204 charts, group_overlap=0, 1.16M comparisons), seed 1,
    4 arms + CM3P legacy reference (results/d2v_attr_d_cached/).
* Diagnostics: C1 metadata-only, C3 mapper-ID probe, C4 attribute decodability
  (results/d2v_attr_diag/).
* Report: docs/d2v_attr_results.md; 1-seed table: results/d2v_attr_run/pilot_table.md.

## Main result (1 split seed, no CI)

D protocol (mapset-disjoint): a20d (+tempo/beat/bar/SV/kiai) vs a0d (8ch):
  linear  0.57558 vs 0.60023  (Delta -0.02464)
  +ext    0.56327 vs 0.59242  (Delta -0.02915)
a27d (adds hitsound/volume): 0.59092 / 0.57352 (worse than a20d, better than a0d).
a12d (release only): ~0 effect.
CM3P legacy reference on same split: linear 0.61804 / +ext 0.58889 / MLP+ext 0.58696.

S protocol (5-epoch continuation, transductive/holdout): a20 directionally better
(P2 +ext -0.0055, P4 -0.0017, P4-hold -0.0014) but inside single-seed noise.

## Not completed: shuffle negative control

* 3 jobs (14118322/14118323/14118325) submitted 13:12, waited ~2.5 h in queue,
  then CANCELLED by 0 at 15:41:36 (external; local watcher issued no scancel).
* Re-submission fails with "sbatch: error: AssocGrpSubmitJobsLimit" even though the
  queue is empty and even a 1-minute/1-core trivial job is rejected
  (checked 17:07). This is an account/group-level QOS or submit limit, not queue
  occupancy; needs user or HPC administrator action.
* scripts/hpc_attr_shuffle_retry.sh will submit train->extract->probe automatically
  (retry every 5 min) and pull results into results/d2v_attr_d_cached/a20shuf/
  once the quota recovers.
* Consequence: the a20d gain cannot yet be separated from a channel-count/capacity
  or statistical artefact; see docs/d2v_attr_results.md section 3.3 / section 6.

## Reproduce

scripts/hpc_attr_run3.sh        # S+D training -> extraction -> probes (attempt 3)
scripts/hpc_attr_dprobe2.sh     # D cached set-P4 probe rerun
scripts/hpc_attr_shuffle_retry.sh  # shuffle control once quota allows
scripts/analyze_attr_pilot.py   # aggregated table
scripts/run_attr_diag_all.sh    # C3/C4 for all arms
