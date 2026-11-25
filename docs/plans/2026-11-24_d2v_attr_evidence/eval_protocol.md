# D2V 新内容表端到端评测协议（只读盘点）

> 2026-11-24，subagent eval-proto，task-2。**只读调查，未改源码、未跑训练。**
> 目的：给“D2V 8ch front-end 扩展到结构化属性通道”的计划提供 (a) 基线表 + provenance、
> (b) 逐协议命令矩阵、(c) content parquet 契约、(d) 分组划分可用性 verdict、
> (e) 功效建议、(f) 坑。
> 本地 checkout 与 HPC（$HOME/DCR）不同步：标注 [HPC-only] 的产物本地不存在；
> 本文件里本地可复现的命令均已给出现有输入；没有的输入给“需先在 HPC/本机重新生成”。

---

## 0. 结论速览

1. **新属性臂必须在 5 个口径上逐对比较**，每个口径的 8ch-D2V 基线见 §1：
   主确认口径 = **P4（6,828 图、5 split seeds、linear+ext 与 MLP+ext）+ randommerge（大样本）**；
   **P4-holdout（1,365 图）只用于“编码器未见图”的干净检查，功效只有 ~0.012 nats**，
   不能用来否定 <0.01 的效应；P2 是次要口径（单 split，无 5 seeds 的现成线性基线）；
   VSRG Dan benchmark 是外部粗门（MAE 量级，MDE ~0.02–0.03）。
2. **8ch D2V 目前打不过最强 CM3P 基线**：P4 上 legacy MLP+ext 0.57343/0.57373，
   D2V+ext 0.58561，**+0.0123 ± 0.0016 更差（5/5）**；D2V 只赢 canon linear 与 legacy linear。
   新属性通道的合理期望值可参考 canon 消融：去掉 hitsound/volume + kiai + SV 使 CM3P+ext
   在 P4 变差 **+0.0185（linear_extra）/ +0.0208（MLP_extra）nats** —— 属性信息量级 ~0.005–0.02。
3. **mapset-grouped split 已经是“已实现且已执行”**：scripts/step1_p4_split_sets.py；
   Experiment 2b 已跑（docs/audio_newset_results.md、results/audio_newset/p4sets/）。
   但 **split parquet 本地缺失（HPC-only）**，且必须重新生成；**mapper-grouped split 没有实现**，
   只在数据上可行（mapper_id 100% 覆盖）。默认 P2/P4 **不是** set-disjoint。
4. **功效红线**：5 split seeds 下 95% 可检测 = 1.24 × sd(per-seed paired Δ)：
   P4 主口径 ≈ **0.005**、P4-holdout ≈ **0.012**、randommerge ≈ **0.001–0.004**、
   P2 单 split ≈ **0.006**、P2-holdout 单 split ≈ **0.015**、VSRG ≈ 0.02–0.03 MAE。
   5/5 符号检验单侧 p=0.031，**5 seeds 本身不足以做强确认**；预注册判据建议
   “mean Δ ≤ -0.005 + 5/5 为负 + 每 seed 配对 CI 方向”或加 seed 到 8–10。

---

## 1. (a) 现有基线：新臂必须超过的数字 + provenance

所有 NLL 都是 rank-NLL（越低越好，nats/comparison）。**绝对值不可跨数据集/协议/ceiling 比较；
只看同一 pair 集上的 paired ΔNLL。** 表中“Δ”均为 5 seeds 逐 seed 配对（负 = 内容表更好）。

### 1.1 协议定义（同一套 step1/pairbest 单位）

| 协议 | 定义 | 输入表 | 评测设置 | n（test） |
|---|---|---|---|---|
| **P2-all** | 原 pair-random 70/30 split + 5 折 chart-disjoint 拟合 | step1_1k_pairbest.parquet | --split test --ceiling-fit all --folds 2 --bootstrap 300（**无** strict-pairs/common-maps） | 6,828 图（4,565 test 图）/ 1,144,617 comparisons |
| **P2-hold** | 仅 1,365 张 D2V-unseen 图，原 split | step1_1k_pairbest_d2v_holdout.parquet（23,905 行） | 同上 | 1,365 图（955 test 图）/ 54,721 comparisons |
| **P4** | 新玩家×新图 6×6/4×4，5 split seeds；off-diagonal 只给 ceiling | step1_1k_pairbest_p4_s{1..5}.parquet | 官方：--split test --ceiling-fit all --common-maps --strict-pairs --folds 2 --bootstrap 0 | 6,828 图；~2.1M comparisons/seed（total） |
| **P4-hold** | chart 侧固定为 1,365 张 D2V-unseen 图，玩家侧 6×6/4×4 | step1_1k_pairbest_p4_d2vhold_s{1..5}.parquet（23,902 行；由 **nogap 表**的 1,365 图子集生成，见 §6.5） | 同 P4 | 1,365 图；~31.9k comparisons/seed |
| **randommerge** | 4 个 random 快照合并、内容过滤后 453,406 对 / 20,388 图；P2 重采样 5 seeds、P4 双重冷启动 5 seeds | step1_randommerge_c_p2_s* / step1_randommerge_c_p4_s* | 同 P4 | P2 ~3.5M；P4 ~2.15–2.89M comparisons/seed |
| **set-P4** | P4 但 chart 侧按 beatmapset_id 整组留出（Experiment 2b） | step1_1k_pairbest_p4sets_s{1..5} [HPC-only] | 同 P4 | 分配图 4,091/2,728；~1.8–2.0 万 test 行 |
| **VSRG Dan** | 1,382 行 benchmark，5 折 chart-group OOF isotonic 的 Dan MAE | benchmark manifest + 单独编码的 1,382 图 content | benchmark_calibrate_eval.py | 1,382 行（1,246 RC / 102 LN / 34 course） |

### 1.2 D2V 8ch（route B，main_holdout，rev c29431f）—— 新臂的“同架构 8ch”基线

| 协议/方法 | test NLL | captured | 配对基线 | 来源 |
|---|---:|---:|---|---|
| P2-all D2V linear | **0.58632** | 0.8193 | CM3P canon linear 0.62404；Δ **-0.03772** [-0.04398,-0.03218] | results/d2v_main/eval/p2_all_d2v.json、p2_all_d2v_rank.json；README §6.4 |
| P2-all D2V linear+ext | **0.58103** | 0.8599 | CM3P canon linear+ext 0.60087；Δ **-0.01985** [-0.02487,-0.01540] | 同上（*_ext*）；paired Δ 由本调查用 paired_delta_nll.py 复核 |
| P2-hold D2V linear | **0.59715** | 0.7289 | CM3P canon linear 0.63704；Δ **-0.03990** [-0.05246,-0.02269] | results/d2v_main/eval/p2_hold_d2v.json |
| P2-hold D2V linear+ext | **0.59388** | 0.7538 | canon ext 0.61666；Δ **-0.02278** [-0.03579,-0.00508] | 同上 |
| P4 D2V linear | **0.59301 ± 0.00417** | 0.7859 ± 0.0475 | canon linear 0.63050 ± 0.00270；Δ **-0.03749 ± 0.00403（5/5）** | results/d2v_main/p4_d2v/ vs p4_cm3p/；README §6.5 |
| P4 D2V linear+ext | **0.58561 ± 0.00459** | 0.8437 ± 0.0468 | canon ext 0.60335 ± 0.00515；Δ **-0.01774 ± 0.00401（5/5）** | 同上 |
| P4 D2V MLP | 0.59681 ± 0.00281 | 0.7567 | legacy MLP 0.62540；Δ -0.02859 ± 0.00734 | results/d2v_main/p4_d2v_mlp/ vs p4_cm3p_legacy_mlp/ |
| P4 D2V MLP+ext | 0.58605 ± 0.00460 | 0.8403 | legacy MLP+ext 0.57343/0.57373；Δ **+0.01232 ± 0.00163（5/5 更差）** | 同上 |
| P2 D2V MLP（只有 3 seeds） | 0.58942 ± 0.00426 | 0.7868 ± 0.0242 | legacy MLP 0.61664（P2 5-seed runner） | results/d2v_main/p2_d2v_mlp/p2/s{1,2,3}/rank_eval.json |
| P2 D2V MLP+ext（3 seeds） | 0.57630 ± 0.00179 | 0.8867 ± 0.0248 | legacy MLP+ext 0.56602（P2）→ D2V +0.010 更差 | 同上 |
| P4-hold D2V linear | **0.60471 ± 0.00994** | 0.7793 ± 0.0678 | canon linear 0.65211；Δ **-0.04739 ± 0.00951（5/5）** | results/d2v_main/p4_hold_d2v/ vs p4_hold_cm3p/；README §6.6 |
| P4-hold D2V linear+ext | **0.60280 ± 0.01215** | 0.7940 ± 0.0708 | canon ext 0.62848；Δ **-0.02567 ± 0.00995（5/5）** | 同上 |
| P4-hold D2V MLP/MLP+ext | **没跑**（无干净基线） | — | — | results/d2v_main/p4_hold_d2v_mlp/ 不存在 |
| randommerge D2V | **没有本地数字** | — | CM3P legacy：P2 ext 0.58326 / P4 ext 0.58271 | 本地 run 于 20:04 中止（results/d2v_main/randommerge_logs/d2v.log）；HPC job 14076959 PENDING（data2vec_main/HPC_JOBS.md） |
| VSRG D2V-L（randommerge 训练，linear_extra+isotonic） | **MAE 0.4527**（CI [0.4270,0.4821]），Spearman 0.8534 | — | RM-L（CM3P legacy）**0.4216**；M1-L 0.7411；Sunny isotonic 0.3367 | docs/benchmark_d2v.md、results/benchmark/eval_d2v/summary.md |

**P4-holdout 的配对 Δ 因为实现细节，README/上表用 strict-pair 官方 NLL 差；
本调查用 all-pairs cluster bootstrap 复核得到 -0.0506…-0.0320/seed（5/5 方向一致，均值 ≈ -0.044），
两者口径不同但结论同向。**

### 1.3 其他基线（新臂也要报，但判据以 1.2 的 8ch-D2V 为主对照）

**CM3P legacy（results/p4_multiseed，5 seeds，ceiling 同 0.56540）：**

| 协议 | linear | linear_extra | MLP | MLP_extra |
|---|---:|---:|---:|---:|
| P2 | 0.61618 ± 0.00293 | 0.57786 ± 0.00335 | 0.61664 ± 0.00265 | **0.56602 ± 0.00307** |
| P4 | 0.62411 ± 0.00345 | 0.58479 ± 0.00494 | 0.62540 ± 0.00653 | **0.57343 ± 0.00433** |

（注意：p4_cm3p_legacy_mlp 重跑 MLP+ext 得 0.57373 ± 0.00382；同 split/ceiling 下
重跑 MLP 有 ~3e-4 的机器/线程非确定性，**必须用同一 run 的配对差**，见 §6.7。）

**CM3P canon（results/p4_multiseed_canon；P4 与 results/d2v_main/p4_cm3p/ 一致）：**

| 协议 | linear | linear_extra | MLP | MLP_extra |
|---|---:|---:|---:|---:|
| P2 | 0.62399 ± 0.00184 | 0.59758 ± 0.00171 | 0.62213 ± 0.00307 | 0.58574 ± 0.00153 |
| P4 | 0.63050 ± 0.00270 | 0.60335 ± 0.00515 | 0.63013 ± 0.00266 | 0.59425 ± 0.00413 |
| P4-hold | 0.65211 ± 0.00754 | 0.62848 ± 0.01047 | 没跑 | 没跑 |

**canon 消融（属性通道的直接先验）**：canon 删掉 hitsound/volume、kiai、mania SV 三类 token 后，
同一 runner/同 split 下 CM3P 变差：P4 linear_extra **+0.01855 ± 0.00385**、
MLP_extra **+0.02082 ± 0.00356**（0/5 seed 变好）；P2 同向 +0.0197。
图级余弦 legacy vs canon 0.8814/0.8844。→ 这三类属性在 CM3P token 空间值 ~0.02 nats，
D2V 连续通道上的效应预计在同一量级或更小；计划应按 ≥0.005 的可检测效应设计。

**randommerge CM3P legacy（docs/randommerge_p2p4_results.md，ceiling 0.59779/0.59769；captured 不可引用）：**

| 协议 | linear | linear_extra | MLP | MLP_extra |
|---|---:|---:|---:|---:|
| P2 | 0.62303 ± 0.00139 | **0.58326 ± 0.00145** | 0.61752 ± 0.00104 | 0.58824 ± 0.00153 |
| P4 | 0.62428 ± 0.00264 | **0.58271 ± 0.00514** | 0.62243 ± 0.00217 | 0.59178 ± 0.00502 |

**H4（行为监督线；full pairbest 1,363 张 D2V-unseen 图，test 3.68M comparisons；与 P4-holdout 不同 construct，不能互比）：**
ceiling 0.513480；D2V linear content 0.57547、D2V MLP 0.57699、CM3P+ext MLP 0.58034、Sunny 0.60390、
MinaCalc 0.62449、star 0.63810（docs/e1_behavior_supervised_encoder.md §5.2；split [HPC-only]）。

---

## 2. (b) 任意 --content-path 的精确命令矩阵

前置：content 表覆盖所有被测图表（8ch/新表都应覆盖 21,949 张 osu 缓存图，见 §3）。
所有命令在仓库根执行；OMP_NUM_THREADS/--threads 按机器核数。
**建议每个新臂一个独立 out-dir**（runner 有“已存在就跳过”逻辑，见 §6.6）。

### 2.1 P2-all（D2V-main 口径，单 split + 5 折 chart-disjoint；与 README §6.4 完全同口径）

    # linear / linear_extra（每张图 = 5 折中“没见过该图”的模型打分）
    python3 scripts/step2_cm3p_rank.py \
      --in data/processed/step1_1k_pairbest.parquet \
      --content-path <CONTENT.parquet> \
      --model linear --chart-only --chart-folds 5 --chart-seed 0 \
      --threads 8 \
      --out-signals results/<arm>/p2_all_signals.parquet \
      --out-json    results/<arm>/p2_all_rank.json
    # linear_extra 版本：--model linear_extra --extra-signals msd_overall,sunny_star
    # （--signal-table 默认 data/processed/rank_signals_1k.parquet）

    python3 scripts/step2_rank_eval.py \
      --in data/processed/step1_1k_pairbest.parquet \
      --signal-table results/<arm>/p2_all_signals.parquet --signals cm3p_rank \
      --split test --ceiling-fit all --folds 2 --bootstrap 300 \
      --cache-dir data/processed/rank_eval_cache \
      --out results/<arm>/eval/p2_all.json

实测成本（本机 8 线程）：signal 生成 linear **445 s**、linear_extra **491 s**；
1,365 图版本 23–26 s；step2_rank_eval（1 个信号，1.14M comparisons，ceiling 已缓存）
约 **200 s**（CPU 争用时可达 10 min+）。ceiling cache 首次生成另加 ~1–2 min。

### 2.2 P2-hold（1,365 张 D2V-unseen 图；唯一“编码器没预训练过”的 P2）

    python3 scripts/step2_cm3p_rank.py \
      --in data/processed/step1_1k_pairbest_d2v_holdout.parquet \
      --content-path <CONTENT.parquet> \
      --model linear_extra --extra-signals msd_overall,sunny_star \
      --chart-only --chart-folds 5 --chart-seed 0 --threads 8 \
      --out-signals results/<arm>/p2_hold_signals.parquet \
      --out-json    results/<arm>/p2_hold_rank.json
    python3 scripts/step2_rank_eval.py \
      --in data/processed/step1_1k_pairbest_d2v_holdout.parquet \
      --signal-table results/<arm>/p2_hold_signals.parquet --signals cm3p_rank \
      --split test --ceiling-fit all --folds 2 --bootstrap 300 \
      --out results/<arm>/eval/p2_hold.json

成本：signal 23–26 s；eval 54,721 comparisons ~10–30 s。**没有 5-seed 版本**：
现有 P2-hold 只跑了单 split（与 README §6.4 一致）；若要多 seed，需先
python3 scripts/step1_pairbest_resplit.py --in data/processed/step1_1k_pairbest_d2v_holdout.parquet
--out-prefix data/processed/step1_1k_pairbest_d2vhold_p2 --seeds 1,2,3,4,5（未落盘），
再走 §2.3 的 runner --protocols p2 --split-p2 data/processed/step1_1k_pairbest_d2vhold_p2。

### 2.3 P4（主确认口径；5 split seeds，模型 seed=0）

    python3 scripts/run_p2_p4_multiseed.py \
      --protocols p4 --seeds 1,2,3,4,5 \
      --models linear,linear_extra,mlp,mlp_extra \
      --content-path <CONTENT.parquet> \
      --out-dir results/<arm>/p4 --threads 8
    # 与 8ch-D2V 基线逐 seed 配对（脚本会断言两边 ceiling 完全一致，否则拒绝比较）
    python3 scripts/compare_multiseed_runs.py \
      --baseline results/d2v_main/p4_d2v \
      --variant  results/<arm>/p4 \
      --out results/<arm>/p4/vs_d2v8ch.md

成本实测（本机 8 线程）：linear+linear_extra 5 seeds ≈ **12.5 min**（~2–3 min/seed）；
mlp+mlp_extra 5 seeds ≈ **6 min**；eval 每 seed **5–10 s**（ceiling 已缓存）。
全 4 方法一次跑完约 20 min。5 seeds 为 split seed；模型 seed 固定 0。

### 2.4 P4-hold（1,365 图，干净口径；固定 checked-in split，勿重新生成）

    python3 scripts/run_p2_p4_multiseed.py \
      --protocols p4 --seeds 1,2,3,4,5 \
      --models linear,linear_extra \
      --content-path <CONTENT.parquet> \
      --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold \
      --out-dir results/<arm>/p4_hold --threads 8
    python3 scripts/compare_multiseed_runs.py \
      --baseline results/d2v_main/p4_hold_d2v \
      --variant  results/<arm>/p4_hold \
      --out results/<arm>/p4_hold/vs_d2v8ch.md

成本：5 seeds × 2 方法 ≈ **4.5 min**；eval 3–10 s/seed。**只有 linear/linear_extra 有基线**；
若要 MLP 行，必须补跑 CM3P legacy/canon MLP 同 split 作对照（每个 5 seeds ≈ 10–15 min）。

### 2.5 randommerge（P2+P4，大样本、高功效；当前 8ch-D2V 数字缺失）

    python3 scripts/run_p2_p4_multiseed.py \
      --protocols p2,p4 --seeds 1,2,3,4,5 \
      --models linear,linear_extra,mlp,mlp_extra \
      --content-path <CONTENT.parquet> \
      --signal-table data/processed/rank_signals_randommerge_od.parquet \
      --split-p2 data/processed/step1_randommerge_c_p2 \
      --split-p4 data/processed/step1_randommerge_c_p4 \
      --out-dir results/<arm>/randommerge --threads 8

成本：split 表本地已有（453,406 行 / 20,388 图）；P2 每 (seed, method) 的
chart-fold 拟合明显更重（~7.5M pairs/fold），P4 linear_extra 单次约 **438 s**（8 线程），
整格（5 seeds × 4 methods × 两协议）按 HPC array 拆并发跑，本地顺序跑需数小时。
**captured 不可引用（ceiling 不是上界，captured 1.10–1.23）**；只看 NLL/Pearson/Spearman。

### 2.6 set-P4（mapset 整组留出；Experiment 2b 协议）

    # 0) 生成 split（本地缺失，除非已有 HPC 产物；6,819 nogap 表是 Experiment 2b 口径）
    python3 scripts/step1_p4_split_sets.py \
      --in data/processed/step1_1k_pairbest_nogap.parquet \
      --meta data/processed/beatmap_meta_1k.csv \
      --out-prefix data/processed/step1_1k_pairbest_p4sets \
      --seeds 1,2,3,4,5
    # 1) 评测（示例：linear_extra）
    python3 scripts/run_p2_p4_multiseed.py \
      --protocols p4 --seeds 1,2,3,4,5 --models linear,linear_extra,mlp,mlp_extra \
      --split-p4 data/processed/step1_1k_pairbest_p4sets \
      --content-path <CONTENT.parquet> \
      --out-dir results/<arm>/setp4 --threads 8

期望：分配图 60/40（4,091/2,728），实际有行图 ~3,350–3,536 / 2,018–2,176，
group_overlap=0；test 比较数约 1.8–2.0 万行。Experiment 2b 已用该 split 跑过
audio/legacy 两臂，证明 runner 可直接消费。**用 6,828 canonical 表做也可以**，
但要换 --meta data/processed/beatmap_meta_full.csv（100% 覆盖）并重新生成 split；
不同 --in/meta 产生的 set-P4 数字不可与 Experiment 2b 的绝对值比，只比 Δ。

### 2.7 VSRG Dan benchmark（外部检查，非 DCR 主指标）

训练集必须与 RM-L 完全一致（randommerge 453,406 行 / 20,388 图，linear_extra λ=1e-3，fit-all）。
新编码器要先对 benchmark 的 1,382 个 .osu 重新编码（staging 已在 data/benchmark_d2v_osu/、
data/processed/benchmark_d2v_ids.txt + benchmark_d2v_map.parquet）：

    # 1) 用新 ckpt 编码 benchmark（输出按 row 序号 0..1381 作 beatmap_id）
    python3 -m d2v_main.extract --config data2vec_main/configs/<cfg>.yaml \
      --ckpt <NEW_CKPT> --osu-dir data/benchmark_d2v_osu \
      --ids-file data/processed/benchmark_d2v_ids.txt \
      --out data/processed/benchmark_content_<tag>_raw.parquet \
      --device cuda --batch-size 32 --num-workers 6 --tag benchmark_<tag>
    python3 scripts/finalize_benchmark_d2v.py \
      --inp data/processed/benchmark_content_<tag>_raw.parquet \
      --out data/processed/benchmark_content_<tag>.parquet   # 断言 512d，见 §6.10
    # 2) 训练（与 RM-L 同表同特征）
    python3 scripts/step2_cm3p_rank.py \
      --in data/processed/step1_randommerge_pairbest_content.parquet \
      --content-path <CONTENT.parquet> \
      --signal-table data/processed/rank_signals_1k.parquet \
      --extra-signals msd_overall,sunny_star \
      --model linear_extra --lam 1e-3 --val-frac 0 --seed 0 --threads 8 \
      --name cm3p_linear_extra --fit-all \
      --save-model results/benchmark/models/<TAG>.pt \
      --out-signals results/benchmark/<tag>_linear/train/signals.parquet \
      --out-json    results/benchmark/<tag>_linear/train/train.json
    # 3) 推理（content 是 row_id 键，不是 beatmap_id）
    python3 scripts/benchmark_infer.py --model results/benchmark/models/<TAG>.pt \
      --content data/processed/benchmark_content_<tag>.parquet \
      --signals data/processed/benchmark_signals.parquet \
      --out results/benchmark/pred/<TAG>.parquet
    # 4) 在 scripts/benchmark_calibrate_eval.py 的 RUNS dict 注册 <TAG>（需改源码），再
    python3 scripts/benchmark_calibrate_eval.py --pred-dir results/benchmark/pred \
      --out-dir results/benchmark/eval_<tag> --runs <TAG>,RM_L,M1_L --bootstrap 1000
    # 5) 可选对比
    python3 scripts/benchmark_d2v_compare.py --oof results/benchmark/eval_<tag>/oof.parquet ...

基线：D2V-L 0.4527 / RM-L **0.4216** / M1-L 0.7411 / Sunny isotonic 0.3367（isotonic MAE）。
注意 201 个训练 bid 与 benchmark 重叠，主表之外要报 clean 子集（1,172 行）敏感性
（docs/benchmark_d2v.md §3.2）。

### 2.8 配对 Δ 与 CI（不是 runner 自动产物）

5-seed 配对：scripts/compare_multiseed_runs.py（用 summary.csv 的逐 seed NLL，断言 ceiling 相同）；
单 split / 子集：scripts/paired_delta_nll.py（all-pairs，player_id_base cluster bootstrap）或
scripts/rank_paired_delta.py（strict-pairs，需要 ceiling cache v2 的 seen-fold mask；
--bootstrap 2000，同时给 player 与 chart 两套 CI）。ceiling cache 路径在
results/.../rank_eval.json 的 ceiling_cache 字段。

---

## 3. (c) Content parquet 契约（新内容表必须满足）

### 3.1 主表（P2/P4/P4-hold/randommerge 共用）

| 列 | 类型/形状 | 说明 |
|---|---|---|
| beatmap_id | int64（step1 里 int32，merge 前会转） | 与 data/osu_cache/<id>.osu 文件名一致；**21,949 个之一** |
| embedding | object（每行 np.ndarray(float32, (512,))） | step2_cm3p_rank._load_content 依次接受列名 embedding/element/vector，**任意 D 维**、必须 2-D finite；L2 norm≈1.0 |
| n_windows | int | 该图 16 s 窗口数（全表 mean 9.96，P50 9，max 319） |
| checksum_ok | bool | **D2V extract 恒为 True（未重算 md5）**，不能当完整性证据 |
| source | str | 如 d2v_main_holdout |
| d2v_rev（CM3P 表为 cm3p_rev） | str | git rev（如 c29431f） |

实测：data/processed/chart_content_d2v_holdout.parquet 21,949 行 × 512d，norm∈[1−1.2e-7,1+1.2e-7]；
legacy/canon CM3P 表同为 21,949×512。
下游消费者：step2_cm3p_rank.py（训练头）→ 线性/MLP 直接在 z-score 后特征上拟合；
benchmark_infer.py 要求 **row_id 键** + embedding（benchmark 表是独立的一次编码，
不是主表子集）；deploy/dan_predict.py 与 export_dcr_table.py 硬编码 CONTENT_DIM=512。

### 3.2 窗口池化（extract.py 的定义，必须保持）

- 训练/抽取窗口 16 s = 1,600 输入帧 @100 Hz → 800 个 50 Hz token；抽取时 16 s/16 s 不重叠，
  最后一块 padding；短于 min_valid_sec=1.0 s 的图在 dataset 阶段丢弃（21,949 全保留）。
- 单窗口向量 = 对窗口内**有效帧**的 token 均值；图向量 = 跨窗口按**有效帧数加权**的均值；
  最后 **L2 normalize**。所以 content 里没有绝对时长/长度信息（n_windows 单独给）。
- 新属性通道：加在 piano roll 输入通道（in_channels: 8 → 8+K，conv 首层），
  抽取/池化/契约不变，仍是 512d；若要改 512d（如 640d）需同步改 §3.1 的硬编码消费者。

### 3.3 覆盖的图表集（data/processed）

| 表 | 行（图） | 覆盖 |
|---|---:|---|
| chart_content_d2v_holdout.parquet | 21,949 | **全体 osu_cache（1k whitelist）**：6,828 pairbest ⊂、1,365 D2V holdout ⊂、20,388 randommerge ⊂，覆盖率 100% |
| chart_content_cm3p_local.parquet / ..._canon.parquet | 21,949 | 同上（21,949） |
| step1_1k_pairbest.parquet | 6,828 图 / 109,706 行 | P2/P4 主口径 |
| step1_randommerge_pairbest_content.parquet | 20,388 图 / 453,406 行 | randommerge（内容过滤后） |
| step1_full_pairbest.parquet | 20,945 图 / 1,898,076 行 | H4/full 线（本地有；H4 split 无） |
| benchmark_content_d2v.parquet（及 legacy/canon） | 1,382 行 | VSRG benchmark（按 row_id；bid 可能空/重复，rate 变体共用 bid），n_windows mean 11.2 |

随机 1,561 张 osu_cache 图不在 randommerge 的 pair 表里，但 content 有；训练时只保留
step1 表里出现的 20,388 图（_load_content 的 keep 过滤，log 会打 coverage 100%）。

---

## 4. (d) 分组数据可用性 verdict

### 4.1 21,949 缓存 id → beatmapset_id / mapper_id 的映射

- .osu 文件名 stem **就是 beatmap_id**；data/processed/beatmap_meta_full.csv
  （22,419 行，17 列，含 beatmap_id, beatmapset_id, mapper_id, star, bpm, ...）
  **100% 覆盖 21,949 张缓存图**；beatmapset_id 与 mapper_id 均 100% 非空。
  （beatmap_meta_1k.csv 覆盖 6,828/6,819 主表；其他 beatmap_meta_*.csv 是旧快照；
  whitelist_1k.csv 也带 beatmapset_id + checksum，可作交叉校验。）
- 覆盖统计（本机实测）：
  - 21,949 张 → 6,857 sets / 1,399 mappers；singleton set 2,073；max set 35 图；median 3。
  - 6,828 张（P2/P4）→ 3,810 sets / 955 mappers；singleton 2,458；max 35。
  - 1,365 张（D2V holdout）→ 1,146 sets / 519 mappers；singleton 978；max 6。
  - 20,388 张（randommerge）→ 6,684 sets / 1,369 mappers。
- 结论：**做 mapset- 或 mapper-grouped split 的数据完全够用，不需要额外抓取**。

### 4.2 已实现的划分（明确区分 implemented / planned）

| 划分 | 状态 | 位置/证据 | 本地文件 |
|---|---|---|---|
| 默认 P2/P4（chart-disjoint） | **已实现** | scripts/step1_p4_split.py、step1_pairbest_resplit.py；step1_1k_pairbest_p2_s*/p4_s* 本地在 | 有 |
| **beatmapset-grouped P4（set-P4，Experiment 2b）** | **已实现且已执行** | scripts/step1_p4_split_sets.py（group=beatmapset_id，图数贪心 60/40）；docs/audio_newset_results.md；results/audio_newset/p4sets/{audio,legacy}/p4/s*/rank_eval.json 本地在；scripts/slurm/audio_p4sets_split.slurm | **split parquet 本地缺（HPC-only**；docs 写的 data/processed/step1_1k_pairbest_p4sets_s{1..5}.parquet 本 checkout 没有）。用 §2.6 命令重新生成即可（几秒） |
| **H4 chart-holdout（D2V-unseen 1,365 图固定 test + 玩家 60/40）** | **已实现且已执行**（行为 E1/E2 线） | scripts/step1_holdout_split.py；scripts/slurm/behavior_h4_split.slurm；docs/e1_behavior_supervised_encoder.md | **step1_full_holdout_s{1..5}.parquet 本地缺（HPC-only）**；step1_full_pairbest.parquet 本地在，可用 1 条命令重生成 |
| **mapper-grouped split** | **未实现，也没有已写的计划** | 无脚本、无 doc 提及（全仓 grep mapper_id 只出现在 meta/QA，没有 split 使用） | 无（实现方式=复制 set-P4 的 _greedy_groups，把 group_id 换成 mapper_id，估 <1 天） |
| 同 set 隔离（P2/P4 层面排除同曲其他难度） | **未实现**（默认 P2/P4 是 chart-disjoint，不是 set-disjoint；同 set 兄弟图可跨 train/test） | step1_p4_split.py 无 group 参数 | — |

**对属性通道实验的 verdict**：
- 主口径应继续用 P4（与现有数字可比）；
- **加一条 set-P4（mapset-grouped）稳健性臂**：脚本现成、5 seeds、test 比较数足够（~1.8–2 万行）、
  Experiment 2b 已验证 runner 能跑；新内容表只需重新生成 split 并用 --split-p4。
- **H4 不建议作为属性通道主口径**：它是 full pairbest 的 behavior 线口径（1,363 图、3.68M
  comparisons、ceiling 0.513），与 D2V P4-holdout 不同 construct，且 split 本地缺；
  可以作“编码器未见图”的旁证。
- **mapper-grouped 只作为可选的下游稳健性检查**（数据够，代码要新写）；不建议进主 ablation grid。

---

## 5. (e) 统计功效：5 split seeds + paired per-seed ΔNLL

### 5.1 公式与判据

- 5 seeds 下均值的 95% CI 半宽 = t(0.975,4) × sd/√5 = **1.241 × sd**；这就是**最小可检测效应**
  MDE（只有 |mean Δ| ≳ MDE 时 95% CI 才排除 0）。
- 5/5 同向的符号检验单侧 p = 2⁻⁵ = **0.031**（双侧 0.0625）；8/8 → 0.0039；10/10 → 0.0010。
  **不要仅凭 5/5 宣称强确认**；把“5/5 + mean 达阈值 + 单 seed 配对 CI 方向”作为预注册判据。
- std 是 split 方差，不是测量噪声；5 seeds 共享同一批图/玩家，**不是独立样本**，
  不要把 5-seed std 当总体标准误做 t 检验。
- 每个 seed 的 per-seed 配对 CI（本调查用 paired_delta_nll.py 实测）：
  P4 content ≈ ±0.005–0.008、P4 ext ≈ ±0.004–0.006；
  P4-hold content ≈ ±0.010–0.013、P4-hold ext ≈ ±0.010–0.012；
  P2-all 单 split ±0.006（content）/±0.005（ext）；P2-hold 单 split ±0.015（content/ext）。

### 5.2 MDE 表（本仓库实测 sd）

| 协议/比较 | sd(paired Δ) | MDE(5 seeds, 95%) | 建议 |
|---|---:|---:|---|
| P4 D2V content vs canon linear | 0.00403 | **0.0050** | 主判据：Δ≤−0.005 |
| P4 D2V+ext vs canon linear+ext | 0.00401 | **0.0050** | 同上 |
| P4 D2V+ext vs legacy MLP+ext（最强基线） | 0.00163 | **0.0020** | 强基线差很小，容易检出“更差” |
| P4-hold content / +ext | 0.00951 / 0.00995 | **0.0118 / 0.0124** | 只用于干净口径；**<0.012 不得称 null** |
| canon 消融 P4 linear+ext / MLP+ext | 0.00385 / 0.00356 | 0.0048 / 0.0044 | 属性通道的 plausible 量级参考 |
| set-P4 audio（linear+ext） | 0.00366 | 0.0045 | set-P4 同样 ~0.005 可检测 |
| randommerge P4 mlp_extra−linear_extra | 0.00347 | 0.0043 | — |
| randommerge P2 mlp_extra−linear_extra | 0.00090 | **0.0011** | 大样本，可检 1e-3 级 |
| P2-all 单 split（无 5 seeds） | ½CI≈0.006 | ~0.006（单次） | 只作辅助；要 5 seeds 需重跑 runner |
| P2-hold 单 split | ½CI≈0.015 | ~0.015 | 小样本，不要单独下结论 |
| VSRG benchmark（D2V vs RM-L 配对 ΔMAE） | — | ~0.02–0.03 MAE | 外部粗门；不要作为小效应的功效来源 |

### 5.3 建议

1. **主功效来自 P4 主口径 + randommerge**：若预期属性通道单独效应 ~0.002–0.005，
   P4 的 0.005 MDE 刚好边缘；**randommerge 的 ~0.001–0.004 MDE 是唯一能稳定检出
   小效应的 1k 外口径**。P4-holdout 只能检出 ≥0.012。
2. 若属性 ablation 只关心“是否比 8ch 好”，预注册 **Δ ≤ −0.005（P4，5 seeds，5/5）**
   加 **randommerge 同向** 作为成功门槛；配 8–10 seeds 可把 MDE 分别降到 0.003/0.0025。
3. 报告至少：per-seed Δ、mean±sd、5/5（或 n/n）、per-seed 配对 CI（cluster bootstrap 400–2000）、
   ceiling、test pair 数；同一 pair 集才可配对（--common-maps/同 split）。
4. 属性通道是“加通道”消融：预期效应可能接近 canon 的 0.02（若整类属性缺失），
   但单通道（如 kiai）可能只有 0.002–0.005；**别用 P4-holdout 来否定单通道**。

---

## 6. (f) 坑（按危害排序）

1. **Transductive pretraining（最大混淆）**：D2V 编码器预训练在 21,949 中的 **20,584** 张图，
   覆盖 6,828 pairbest 的 80%；所以 **P2/P4 主口径的 D2V 数字带 transductive 成分**，
   README §6.5 已注明。唯一干净的是 P4-holdout（1,365 张）。新 arm 若“继续预训练”：
   **必须保持 pretrain 池不包含 data2vec_main/splits/holdout/eval_ids.txt 的 1,365 张**，
   否则 P4-holdout 的干净性被破坏、旧/新 arm 的 clean 数字不可比。
   反过来，CM3P 的预训练语料暴露未知 → D2V vs CM3P 的绝对差不是单变量结论。
2. **chart-disjoint ≠ beatmapset-disjoint**：P2/P4 只保证 chart 级不重叠，同曲其他难度可跨侧；
   6,828 图 / 3,810 sets（2,458 singleton）意味着不少兄弟图存在。要“新歌”结论就用 set-P4
   （已实现）或同时报两个口径；不要用 P2/P4 的结果声称“新谱面集泛化”。
3. **ceiling 不可比**：P2-all ceiling 0.56276、P4 0.56540±0.00859、P4-hold 0.57888±0.01712、
   set-P4 0.55947±0.01429、randommerge 0.5978（且 captured>1，ceiling 不是上界）。
   **绝对 NLL/captured 只能在同一 split、同一 chart 集、同一 ceiling 设置下比较**；
   compare_multiseed_runs.py 会断言两边 ceiling 逐 seed 相等再配对。
   新内容表若漏图 → map 集/ceiling/captured 都变；务必 100% 覆盖 + --common-maps。
4. **两种 P2 口径别混**：results/d2v_main/eval/*.json 是“原 split + 5 折 chart-disjoint”
   （76,567/33,139 行、n=1,144,617）；results/d2v_main/p2_d2v_mlp 是“5 个 pair-resplit seeds”
   （76,7xx/32,9xx 行、n≈1.05M）。两者的线性/MLP 数字不可混表；新 arm 选一个口径并重跑基线。
5. **P4-holdout split 的 provenance 缺口**：checked-in 的 5 个 parquet 是 1,365 图
   / 23,902 行 / 932 players，来自 **step1_1k_pairbest_nogap.parquet 的 1,365 图子集**
   （不是 6,828 的 step1_1k_pairbest.parquet：后者子集是 23,905 行/933 players），
   但仓库里没有生成脚本/JSON；用当前 step1_p4_split.py 在 nogap 子集上重跑得到
   8,847/3,688（现有 8,882/3,556），**不能 bit 级复现**。→ **直接复用现有 split 文件**，
   不要重新生成；新 arm 只要 --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold 即可。
6. **runner 的静默复用**：run_p2_p4_multiseed.py 发现 signals/<model>.parquet +
   rank_eval.json 已存在就跳过；assemble 只拼 --models 里列出的列（重跑新模型可能
   覆盖/漏列）；history 上 randommerge 因此出过 bug（docs/randommerge_p2p4_results.md §1）。
   **每个臂用全新 out-dir；重跑加 --force/--force-eval 或先删 rank_eval.json。**
7. **随机性/重跑漂移**：同 split 同 ceiling 重跑 legacy MLP+ext 得 0.57373 vs 旧 0.57343；
   MLP 有 ~3e-4 的 run-to-run 漂移（线程/初始化）。**paired Δ 必须同一次 run 内算**
   （或至少同 runner/同环境重跑基线），不要拿旧 run 的绝对值配新 run。
8. **randommerge 的 captured 不可用**（ceiling 欠拟合，captured 1.10–1.23）；
   只看 NLL/Pearson/Spearman/sign。
9. **VSRG 的域差与标签泄漏**：benchmark 是社区 Dan 标注（不是 ACC 序），1,382 行中
   201 个 bid 在 randommerge 训练集里出现过 → 必须报 clean 子集（1,172 行）敏感性；
   benchmark_infer.py 要求 benchmark 自己的 row_id content 表 + 新 RUNS key
   （scripts/benchmark_calibrate_eval.py 需改源码注册）；**不能把主 content 表直接喂进去**。
10. **512d 硬编码**：scripts/finalize_benchmark_d2v.py 断言 {512}；deploy/dan_predict.py
    与 deploy/export_dcr_table.py 有 CONTENT_DIM=512。属性通道保持 512d 最省事；
    若改维度，需同步所有消费者（probe 脚本本身支持任意 D）。
11. **content 标准化**：step2_cm3p_rank.py 在 **训练图**上逐维 z-score（sd+1e-8），
    再拟合头；接近常数的通道（如 kiai 稀疏）不会炸但也没信息；检查新通道的
    min/max、非零率、per-dim sd，必要时换窗口池化/加统计量（长度、密度）而不是只靠 mean。
12. **checksum_ok 恒 True**（D2V extract 未重算 md5）：新表最好真算 .osu md5，否则
    content 表与 .osu 版本失配无感知。source/d2v_rev 写清楚；不同 rev 的表不混比。
13. **P4-holdout MLP 基线缺失**：现有 clean 口径只有 linear/linear_extra；若新臂要在
    MLP 行比较，先在 1,365 图上补跑 CM3P legacy/canon MLP（5 seeds）作对照。
14. **H4 与 P4-holdout 不是一回事**：H4=full pairbest、1,363 图、3.68M comparisons、
    ceiling 0.513；P4-holdout=1k pairbest、1,365 图、~31.9k comparisons/seed、ceiling 0.579。
    两者的绝对 NLL/Δ 都不能互比。

---

## 7. 建议的 ablation grid（给计划用）

内容表臂（每臂 = 一个 --content-path；都从同一 8ch D2V ckpt 继续预训练、同 epoch/预算）：
C0 = 8ch D2V（基线，已有）；C1 = +hold-release pulse；C2 = +BPM/beat/SV/kiai；C3 = +hitsound/set/volume；
C4 = C1+C2+C3（全量）；C5 = CM3P legacy（跨 tokenizer 参照）；C6 = CM3P canon（属性类消融参照）。

| 协议 | 方法 | seeds | 判据（相对 C0） | 成本（本地 8 线程，估） |
|---|---|---|---|---|
| P4 主 | linear, linear_extra, mlp, mlp_extra | 1–5 | Δ≤−0.005，5/5，配对 CI 方向 | ~20 min/臂 |
| P4-hold | linear, linear_extra（必要时补 MLP 基线） | 1–5 | Δ≤−0.012 才能称方向；不以 null 否定 | ~5 min/臂（+补基线） |
| randommerge P4（+P2） | linear, linear_extra（主）；MLP 可选 | 1–5 | 高功效筛查：Δ≤−0.002~−0.004 | 数小时/臂（建议 HPC array） |
| P2-all | linear, linear_extra | 单 split | 辅助一致性 | ~12 min/臂 |
| set-P4 | linear, linear_extra | 1–5 | 同 P4 判据的稳健性复核 | ~15 min/臂 + split 生成 |
| VSRG | linear_extra（RM 表训练） | 1（fit-all） | 只作外部 gate，ΔMAE 与 RM-L 同向 | 训练 ~20–40 min + 编码/校准 |

优先级：**C4 vs C0（全量）→ 哪个属性组贡献（C1/C2/C3 单独）→ 与 C5/C6 的同一 runner 参照**；
先跑 P4 + randommerge 确认，再跑 P4-hold/set-P4 做干净性/稳健性；VSRG 只在 DCR 有正向信号后跑。

---

## 8. 产物与 provenance 索引（全部本地只读核对过）

| 内容 | 路径 |
|---|---|
| D2V README §6.4–6.6 数字/口径修正 | data2vec_main/README.md |
| P2 评测 JSON（linear/+ext、all/hold） | results/d2v_main/eval/p2_{all,hold}_{d2v,cm3p}{,_ext}.json |
| P2 signal/rank | results/d2v_main/p2_*_signals.parquet / p2_*_rank.json；复现脚本 data2vec_main/scripts/run_p2_probes.sh、run_p2_extra.sh |
| P4/P4-hold 结果 | results/d2v_main/p4_{d2v,cm3p}/、p4_{d2v,cm3p}_mlp/、p4_hold_{d2v,cm3p}/（summary.csv + 每 seed rank_eval.json + signals） |
| P4/P2 复现 shell | data2vec_main/scripts/run_p4_probes.sh、run_p4_hold_probes.sh、run_mlp_probes.sh、run_randommerge_d2v.sh |
| canon 消融 | docs/p4_multiseed_canon.md、results/p4_multiseed_canon/ |
| legacy multiseed | docs/p4_multiseed.md、results/p4_multiseed/ |
| randommerge 基线 | docs/randommerge_p2p4_results.md、results/randommerge_p2p4_hpc/ |
| VSRG | docs/benchmark_vsrg.md、docs/benchmark_d2v.md、results/benchmark/eval_d2v/summary.md |
| H4 行为线 | docs/e1_behavior_supervised_encoder.md、docs/plans/2026-11-24_behavior_supervision_plan.md §1.2 |
| mapset split | scripts/step1_p4_split_sets.py、docs/audio_newset_results.md、results/audio_newset/p4sets/、scripts/slurm/audio_p4sets_{split,train}.slurm |
| H4 split 脚本 | scripts/step1_holdout_split.py、scripts/slurm/behavior_h4_split.slurm |
| meta/分组 | data/processed/beatmap_meta_full.csv（22,419×17，100% 覆盖 21,949 缓存图） |
| ceiling cache | data/processed/rank_eval_cache/（52 个 npz，4.8 MB；key=md5(map_ids)+seed+folds+min_l+n_rows+split+fitall extra） |
| 配对 Δ | scripts/compare_multiseed_runs.py、scripts/paired_delta_nll.py、scripts/rank_paired_delta.py |
| 本次只读复核脚本输出 | /tmp/d2v_eval_proto/（临时，不落库）：P2 paired CI、P4/P4-hold per-seed paired CI、P4-hold split 重生成对照 |

**关键缺口（计划需注意）**：
- [HPC-only] data/processed/step1_full_holdout_s{1..5}.parquet（H4 split）、step1_1k_pairbest_p4sets_s{1..5}.parquet（set-P4 split）、D2V randommerge 结果（job 14076959 PENDING）、P2 MLP s4–s5、P4-holdout MLP；
- P4-holdout split 无生成脚本/JSON，不能 bit 级复现（§6.5）；
- 本 checkout 无 GPU/HPC，D2V 继续预训练与 re-extract 需在 A800/A40 或等价环境；
  8ch 全量预训练成本参考 README §6.2：20 epochs/25,720 steps/47 min on A800；抽取 27 min。
