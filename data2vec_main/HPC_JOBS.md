# HPC CPU migration（2026-11-23 20:25）

本地 probe 任务已停止，改投 HPC CPU 队列（`i64m512u`，16 核 / 64G）。
本地机器可以重启；下面是重启后要看的 job。

| Job | 分区 | 脚本 | 内容 | 产物 |
|---|---|---|---|---|
| **14076958** | i64m512u | `scripts/hpc_mlp_probes.sbatch` | 1k P4/P2，D2V + CM3P legacy，`mlp,mlp_extra`，5 seeds | `results/d2v_main/p4_*_hpc`、`p2_*_hpc` |
| **14076959** | i64m512u | `scripts/hpc_randommerge_probes.sbatch` | randommerge P2/P4，D2V，`linear,linear_extra`，5 seeds | `results/d2v_main/randommerge_d2v_hpc` |

提交时两个 job 都是 `PENDING (Priority)`。

重启后查看：

~~~bash
ssh aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
cd ~/DCR
export PATH=/opt/slurm/bin:$PATH
squeue -j 14076958,14076959 -o "%.10i %.9P %.8T %.10M %R"
tail -f slurm_d2v_mlp_cpu_14076958.out
tail -f slurm_d2v_rm_cpu_14076959.out
~~~

完整结果在：
- `results/d2v_main/p4_d2v_mlp_hpc/summary.md`
- `results/d2v_main/p4_cm3p_legacy_mlp_hpc/summary.md`
- `results/d2v_main/p2_d2v_mlp_hpc/summary.md`
- `results/d2v_main/p2_cm3p_legacy_mlp_hpc/summary.md`
- `results/d2v_main/randommerge_d2v_hpc/summary.md`

注意：randommerge 的 captured 不可引用（官方文档：ceiling 不是上界，captured >1）；只看 NLL / Pearson / Spearman。
