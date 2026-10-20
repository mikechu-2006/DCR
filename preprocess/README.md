# preprocess/ —— step0 + step1 最小参考

三步管线里 step0/step1 的**可运行精简版**：去掉 I/O 分支与逐项校验，只保留数据契约
和清洗逻辑，便于对照阅读或单独跑。完整口径、漏斗与论证见
[docs/data.md](../docs/data.md)；更细的实测在 `legacy/docs/step0_preprocessing.md`、
`legacy/docs/step1_pairbest.md`、`legacy/docs/step1_cleaning.md`。

| 文件 | 对应主线 | 作用 |
|---|---|---|
| `step0_preprocess.py` | `scripts/step0_preprocess.py` | SQL dump → play 表 + 谱面元数据 |
| `step1_pairbest.py` | `scripts/step1_pairbest.py` | 默认口径 A：一对一行 |
| `step1_clean.py` | `scripts/step1_clean.py` | legacy play 级：划分后清洗 train 侧 |

## 数据契约

**step0 play 表**（5 列）：
`player_id, beatmap_id, timestamp, playcount_cur, loss`，其中
`loss = log(1 - ACC)`。

保留规则：

~~~text
4K:   playmode == 3 且 diff_size == 4
mod:  canon = mods - {MR, SD, PF};  keep <=> canon == {CL}
      NF 与其它 mod 一律拒绝；modern/legacy 重叠行只留 modern
~~~

1k：1,399,615 行 / 990 玩家 / 20,655 谱面；10k：13,359,355 / 9,812 / 21,899。

**step1/pairbest**（默认 A，9 列 = 5 列 + `split + player_id_base + player_seg + n_plays`）：

~~~text
0. 玩家记录数 < 20 整人剔除
1. 连续 >= 365 天没记录 -> 新玩家段
2. 每个 (玩家段, 谱面) 只留最高成绩
3. n_plays < 2 的对剔除
4. 最高 ACC > 0.998 的对剔除
5. 最高 ACC <= 0.90 的对剔除
6. 幸存对上 pair-random 70/30
~~~

1k：109,706 对（train 76,567 / test 33,139）；
10k：1,513,746 对（1,059,890 / 453,856）。

**step1/clean**（legacy，6 列 = 5 列 + `split`）：先按玩家记录数 < 200 整人剔除，
再划分；只对 train 行做 ACC <= 0.90 删除和每玩家最早 30% 热身截断；test 原样保留。

## 运行

~~~bash
python preprocess/step0_preprocess.py --dump-dir <dump> --out /tmp/step0_1k.parquet
python preprocess/step1_pairbest.py --in /tmp/step0_1k.parquet --out /tmp/step1_1k_pairbest.csv
python preprocess/step1_clean.py   --in /tmp/step0_1k.parquet --out /tmp/step1_1k.csv
~~~

参数与主线脚本基本一致（`--help` 可查）。管线入口见根 [README](../README.md) 与
[docs/procedure.md](../docs/procedure.md)。
