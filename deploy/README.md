# DCR Dan 部署接口（封装版）

> 2026-11-23。给负责封装/产品化的同事：这是一个**两个函数**的参考程序，
> 不依赖 torch / numpy / sklearn，可直接照抄成 TypeScript。
>
> 代码：[deploy/dan_predict.py](dan_predict.py)
> 权重：[deploy/models/](models/)
> 校准：与权重在同一个 JSON 里
> 一致性测试：[deploy/test_parity.py](test_parity.py)
> 分布/MAE 检查：[deploy/check_benchmark.py](check_benchmark.py)
> **出表脚本：[deploy/export_dcr_table.py](export_dcr_table.py)**（§9，把 21,945 张图一次性算成查表）

## 1. 快速使用

~~~python
from dan_predict import cm3p_to_dan, d2v_to_dan

# 514 维：512 维内容向量 + [msd_overall, sunny_star]
vector = list(C_512) + [msd_overall, sunny_star]
dan = cm3p_to_dan(vector)   # 或 d2v_to_dan(vector)
print(dan)                  # 一个实数，表示校准后的 Dan 估计
~~~

两个函数签名相同：

~~~python
cm3p_to_dan(vector: Sequence[float]) -> float
d2v_to_dan(vector: Sequence[float]) -> float
~~~

- 输入长度必须是 **514**；不足/超出、含 NaN/inf 会抛 ValueError。
- 返回的是**校准后的 Dan 实数**，不是 BT 潜变量 d。
- 同一个向量永远得到同一个数；没有随机性。

## 2. 向量契约

~~~text
vector = [ C[0..511], msd_overall, sunny_star ]
~~~

| 位置 | 含义 | 训练口径 |
|---|---|---|
| 0..511 | 内容向量 C | CM3P：OliBomby/CM3P@26cc694，audio off，crop legacy，512-d；D2V：audio-style data2vec holdout 编码器，512-d，L2-normalised |
| 512 | msd_overall | MinaCalc standalone calc_version 505，goal 0.93，rate 1.0 |
| 513 | sunny_star | osumania_map_analyser commit 448ed7c789af3d715acfb8370e41244cc3c00865 |

**必须传原始 C / 原始 msd / 原始 sunny**，不要预先标准化；函数内部用模型自带的
mu/sd 与 extra_mu/extra_sd 做标准化。向量表不要存标准化后的值，否则换模型不能复用。

## 3. 两个模型与固定校准

| content | JSON | 模型 | 训练数据 | 校准 |
|---|---|---|---|---|
| cm3p | models/cm3p_linear_extra.json | linear_extra，514 权重 | randommerge 全量：453,406 行 / 20,388 图 | 全部 1,382 行 VSRG benchmark，isotonic 全量拟合 |
| d2v | models/d2v_linear_extra.json | linear_extra，514 权重 | randommerge 全量：453,406 行 / 20,388 图 | 全部 1,382 行 VSRG benchmark，isotonic 全量拟合 |

每个 JSON 同时包含：

- mu / sd（512 维内容标准化，训练图统计）；
- extra_mu / extra_sd（2 维外部信号标准化）；
- w（514 维线性权重）；
- d_mu_raw / d_sd_raw（线性模型为 0 / 1）；
- calibration.x / calibration.y（isotonic knots，按 x 升序）；
- model_id、content_md5、train_git_commit、mae_in_sample、mae_oof。

**校准是固定的，不要在客户端按请求重拟合。** 它是在全部 1,382 行 VSRG benchmark
上拟合的 deployment 权重；5 折 OOF 的 MAE 只用于评测，不是这里用的校准。

当前 CM3P 数字（供 sanity check）：

| 项 | 值 |
|---|---:|
| 全量 isotonic 拟合的 in-sample MAE | 0.3795 |
| 5 折 OOF isotonic MAE（评测口径） | 0.4216 |
| 与 sklearn 参考的 raw d 最大差（1,382 行） | 3.6e-07 |
| 校准后 Dan 最大差（1,382 行） | 1.28e-04 |

D2V 的对应数字：

| 项 | 值 |
|---|---:|
| 全量 isotonic 拟合的 in-sample MAE | 0.4132 |
| 5 折 OOF isotonic MAE（评测口径） | 0.4527 |
| 与 sklearn 参考的 raw d 最大差（1,382 行） | 3.5e-07 |
| 校准后 Dan 最大差（1,382 行） | 8.15e-05 |

D2V benchmark 细节见 [docs/benchmark_d2v.md](../docs/benchmark_d2v.md)；
JSON 的 calibration.mae_in_sample / mae_oof 是权威值。

### 3.1 输出分布：为什么全表看起来像 star

函数输出的是 **Dan**，不是 star；但分位数会随**输入图集**变化：

| 输入 | n | min | p25 | p50 | p75 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| benchmark 1,382 行（CM3P） | 1,382 | 1.00 | 13.29 | 14.03 | 14.26 | 17.38 | 17.58 |
| 全 21,949 图表（CM3P） | 21,949 | 1.00 | 1.73 | 2.76 | 5.20 | 14.09 | 16.36 |
| benchmark 1,382 行（D2V） | 1,382 | 1.50 | 13.37 | 14.03 | 14.32 | 16.85 | 18.00 |
| 全 21,949 图表（D2V） | 21,949 | 1.50 | 1.50 | 2.76 | 5.31 | 14.12 | 16.85 |

你看到的 **min 1.00、p25 1.7252、p50 2.76、p75 5.20、p99 14.0878、max 16.36**
正好是 **CM3P 在全部 21,949 张图表上的分布**，不是 benchmark 的 1,382 行。

原因：benchmark 是社区高段图，expected 以 13–14 为主；而内容表里大多数图是低难图。
同一个单调 isotonic 映射作用在低难图上，分位数自然低。min=1.00 是 isotonic 的第一个
knot（benchmark expected 的最小值）；max=16.36 只是全表里最大 d_m 落到的 y 值，
模型的 y 上限是 17.58（CM3P）/18.0（D2V），benchmark 上才碰到上限。
它们不是 star 的 0–10 尺度。

**VSRG MAE 必须在 benchmark 的 1,382 行上算**。在这 1,382 行上，函数输出与
isotonic got 一致：CM3P in-sample MAE 0.3795、D2V 0.4132；OOF 分别是 0.4216 / 0.4527。
可以直接跑 python deploy/test_parity.py 复现（它会打印 mae_vs_expected）；
也可以只跑 python deploy/check_benchmark.py（只用 pandas/numpy，不需要 sklearn，
会打印 min/p25/p50/p75/p99/max 与 MAE）。

## 4. 前向公式（TS/JS 照抄）

~~~text
d = sum_i w[i] * (C[i] - mu[i]) / sd[i]
  + sum_j w[512+j] * (S[j] - extra_mu[j]) / extra_sd[j]
d = (d - d_mu_raw) / d_sd_raw

dan = isotonic_interp(d, calibration.x, calibration.y)
~~~

isotonic_interp 与 sklearn 的 IsotonicRegression(out_of_bounds="clip").predict 一致：
低于第一个 knot 取 y[0]，高于最后一个 knot 取 y[-1]，中间线性插值。

~~~ts
function isotonic(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
}
~~~

## 5. 一致性测试

~~~bash
python deploy/test_parity.py
~~~

它会对 1,382 行 benchmark 逐行：
1. 用 [C, msd, sunny] 调两个函数；
2. 用 sklearn 在相同 raw d 上拟合同样的全量 isotonic；
3. 比较 raw d 与最终 Dan。

CM3P 当前结果：max|Δd| = 3.6e-07，max|Δdan| = 1.28e-04。
后者主要是 isotonic 在陡峭段的斜率放大，属于可接受的浮点差；raw d 的 parity 更严格。

## 6. 封装侧需要的东西

1. **两份 JSON**（models/cm3p_linear_extra.json、models/d2v_linear_extra.json），
   合计约 86 KB；不需要 torch。
2. **内容向量表**：客户端/服务端按谱面 MD5 查 C，并同时存 msd_overall、sunny_star。
   已知图完全复现训练特征；未知图没有 C 时不能用这两个函数，需要 external-only 兜底
   （另一个模型，不在本接口内）。
3. **版本合同**：Sunny commit、MinaCalc calc_version 505、CM3P rev、D2V rev 必须与
   JSON 里的 content_md5 / train_git_commit / 特征合同一致；换版要重训或做一致性测试。
4. **UI 标签**：显示“DCR Dan 估计”，不要标成官方 Dan/star。校准的 MAE 在 0.38–0.42，
   不是官方段位系统。

## 7. 重新生成

~~~bash
# CM3P（已有 pred）
python scripts/export_dan_deploy.py \
  --model results/benchmark/models/RM_L_full.pt \
  --pred results/benchmark/pred/RM_L.parquet \
  --content-type cm3p \
  --out deploy/models/cm3p_linear_extra.json \
  --oof results/benchmark/eval_rm/oof.parquet --oof-col RM_L_isotonic

# D2V（D2V_L_full.pt + pred 生成后）
python scripts/export_dan_deploy.py \
  --model results/benchmark/models/D2V_L_full.pt \
  --pred results/benchmark/pred/D2V_L.parquet \
  --content-type d2v \
  --out deploy/models/d2v_linear_extra.json \
  --oof results/benchmark/eval_d2v/oof.parquet --oof-col D2V_L_isotonic
~~~

## 8. 已知限制

- 校准只在 VSRG benchmark 的 1,382 张 4K 谱面上得到；它不是官方 Dan，也不能保证
  在其他谱面总体/其他键数上同尺度。
- CM3P/D2V 的 C 都是 4K、特定编码器/rev 的产物；换编码器等于换输入分布。
- msd/sunny 的版本敏感；MMA 自带的 MSD 与 MinaCalc 505 不等价。
- 输入缺 C 时不要静默用 0 向量；应走 external-only 路径并标 approx。
- 这是线性 + 固定单调校准，不含玩家能力/OD；不要把它解释成“这个玩家能打多少 ACC”。
- **模型选择**：VSRG benchmark 上 CM3P/RM-L 的 MAE 0.4216 优于 D2V 的 0.4527；
  D2V 在 expected>15 的高段（0.6232）甚至比旧 M1-L（0.5548）差。
  如果目标人群集中在高段，优先 CM3P；D2V 适合作为第二路 content provider/对照。

## 9. 出表：把 21,945 张图一次算完

插件侧（ManaMapAnalyser）不装 512 维向量表，只装一张 `md5 → Dan` 的定长表。
这张表由 `deploy/export_dcr_table.py` 生成，**在你这边跑**（需要 `rank_signals_1k.parquet`，
它被 `.gitignore` 的 `data/**` 与 `*.parquet` 挡住，所以不在仓库里）。

~~~bash
python deploy/export_dcr_table.py \
  --content data/processed/chart_content_cm3p_local.parquet \
  --signals data/processed/rank_signals_1k.parquet \
  --whitelist manifests/whitelist_1k.csv
~~~

默认读 `deploy/models/cm3p_linear_extra.json`、输出到 `deploy/table/`：

~~~text
deploy/table/dcr_content.bin              600 KB（21,945 行 × 28 B + 16 B 头）
deploy/table/dcr_content.manifest.json    sha256 / count / 覆盖 / Dan 分布 / model_id
~~~

只想要两列信号（给没有本仓库的人）：

~~~bash
python deploy/export_dcr_table.py --signals data/processed/rank_signals_1k.parquet \
    --dump-signals signals_1k.csv
~~~

### 9.1 表格式（客户端按这个解析，不要改）

~~~text
header 16 B : "DCRT" | u16 format_version=1 | u16 record_size=28 | u32 count | u32 reserved
record 28 B : md5[16] | f32 dan | f32 d | f32 sunny      按 md5 字节升序，二分查找
~~~

- `dan`：校准后的 Reform Dan，**这是 UI 显示的数**；
- `d`：BT 潜变量，仅用于对拍/诊断；
- `sunny`：训练期的 `sunny_star`，仅用于对拍/诊断。

### 9.2 脚本会拦住的四类错

1. **内容表与模型不匹配**：`md5(parquet 字节)` ≠ 模型 `content_md5` → 直接退出 2，
   不产出任何文件（这条最容易静默出错：拿 d2v 模型配 cm3p 向量，数字照样出得来，但全错）。
2. **向量维度/信号缺失**：`embedding` 不是 512、或 `msd_overall`/`sunny_star` 为 NaN → 退出 2。
3. **重复键**：内容表或信号表里 `beatmap_id` 重复 → 退出 2。
4. **写出后回读**：magic / 记录长度 / 总数 / md5 有序性逐项复核。

另外会抽 200 行用公开 API `cm3p_to_dan()` 复算，确认导出值与参考实现一致（阈值 1e-9）。

### 9.3 已验证的数字（合成信号，仅验链路，不是真实读数）

| 项 | 值 |
|---|---:|
| 行数 / 体积 | 21,945 / 614,476 B |
| 覆盖 4K 白名单 | 21,945 / 21,945（100%） |
| 落盘 f32 与 Python 参考的最大差 | 4.5e-07（f32 精度极限） |
| 与插件侧独立实现的字节比对 | **完全一致**（sha256 相同） |

真实 `dan` 分布要等真实 `msd/sunny` 跑出来才有效；当前用合成信号时中位数偏低（3.43），
因为合成的 `sunny_star` 均值 3.89 覆盖不到高段。

表更新后，插件侧要同步 bump `CACHE_KEY_STAR_UNIFIED_VERSION`（`js/app/core/analysis.js`），
否则玩家看到的是旧缓存里的数。
